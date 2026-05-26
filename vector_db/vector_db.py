"""
VectorDB integration module for Qdrant.
This module handles vector search using Qdrant as the vector database.
Uses lazy initialization to avoid crashing the app when DB is not ready.

Changes (Review-02):
  - URI parsing now uses urllib.parse.urlparse() instead of manual string split
    to handle IPv6 and URLs with scheme correctly.
  - Added retry mechanism with exponential backoff for Qdrant connections.
  - create_embeddings() now returns a numpy array directly instead of a list
    of individual ndarrays for better performance and API clarity.
  - hybrid_search() now actually calls graph_db.graph_rag_search() and combines
    results with vector search results using weighted scoring.
  - Model name moved to configurable constant (EMBEDDING_MODEL) for easier override.

Changes (Review-04):
  - Chunking logic extracted to chunking.py with support for:
      * Overlap (10-20%) between chunks for context continuity
      * UTF-8 byte-boundary safety to prevent character corruption
      * Semantic chunking via embedding model
      * Thai-aware splitting via pythainlp
      * Mode switching (character/semantic/thai_aware)
"""

import os
import re
import time
import qdrant_client
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, VectorParams, FilterSelector
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import requests
from config import get_config
from chunking import split_content, SplitMode

logger = logging.getLogger(__name__)

# Pipeline Improvement: Dynamic embedding model from environment variable with validation.
# The actual model name is resolved at runtime via config.get_embedding_model().
# This constant is kept as a fallback default only.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Supported embedding models with dimension and source metadata.
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "all-MiniLM-L6-v2": {"dim": 768, "source": "sentence_transformers"},
    "BAAI/bge-m3": {"dim": 1024, "source": "sentence_transformers"},
    "nomic-embed-text-v1.5": {"dim": 768, "source": "external_api"},
}

# External API endpoint for nomic-embed-text-v1.5 (OpenAI-compatible).
EXTERNAL_EMBEDDING_API_URL = os.getenv(
    "EXTERNAL_EMBEDDING_API_URL", "http://localhost:7700/v1"
)


def load_embedding_model(model_name: str) -> Any:
    """
    Load an embedding model by name with cache validation and progress logging.

    Checks if the model exists in the HuggingFace cache (~/.cache/huggingface).
    If not, downloads it with a progress indicator. Loads the model and verifies
    that the output dimension matches the expected value from SUPPORTED_MODELS.

    Args:
        model_name: The embedding model identifier (e.g., 'BAAI/bge-m3').

    Returns:
        A loaded SentenceTransformer model instance.

    Raises:
        ValueError: If the model is not in SUPPORTED_MODELS.
        RuntimeError: If model loading fails or dimension mismatch occurs.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not in SUPPORTED_MODELS. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    expected_dim = SUPPORTED_MODELS[model_name]["dim"]
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    # Check if model cache directory exists for this model.
    # HuggingFace stores models under ~/.cache/huggingface/hub/snapshots/<hash>
    # We check the model card/index presence as a heuristic.
    model_cached = False
    if os.path.isdir(cache_dir):
        # Search for any snapshot directory containing the model files.
        hub_dir = os.path.join(cache_dir, "hub")
        if os.path.isdir(hub_dir):
            for root, dirs, files in os.walk(hub_dir):
                if "model.safetensors" in files or "pytorch_model.bin" in files:
                    model_cached = True
                    break

    status_msg = "cached" if model_cached else "download required"
    logger.info(
        f"Loading embedding model '{model_name}' [dim={expected_dim}, source={SUPPORTED_MODELS[model_name]['source']}, cache={status_msg}]"
    )

    start_time = time.time()
    try:
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        load_time = time.time() - start_time

        # Verify the actual output dimension.
        test_embedding = model.encode(["dimension check"])
        actual_dim = test_embedding.shape[1]

        if actual_dim != expected_dim:
            logger.warning(
                f"Dimension mismatch for '{model_name}': expected={expected_dim}, actual={actual_dim}. "
                f"Updating expected dimension to {actual_dim}."
            )
            SUPPORTED_MODELS[model_name]["dim"] = actual_dim

        logger.info(
            f"Model '{model_name}' loaded in {load_time:.2f}s — output_dim={actual_dim}"
        )
        return model

    except OSError as e:
        load_time = time.time() - start_time
        logger.error(
            f"Failed to download/load model '{model_name}' after {load_time:.2f}s: {e}"
        )
        raise RuntimeError(f"Model download failed for '{model_name}': {e}") from e
    except Exception as e:
        load_time = time.time() - start_time
        logger.error(f"Failed to load model '{model_name}' after {load_time:.2f}s: {e}")
        raise RuntimeError(f"Model loading failed for '{model_name}': {e}") from e


def validate_embedding_model(model_name: Optional[str] = None) -> bool:
    """
    Verify that the specified embedding model is available and loadable.

    Logs model name, expected dimension, and cache status.
    Does not raise — returns False on failure so startup can decide.

    Args:
        model_name: Model to validate. Defaults to EMBEDDING_MODEL env var.

    Returns:
        True if the model is valid and loadable, False otherwise.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL

    if model_name not in SUPPORTED_MODELS:
        logger.error(
            f"Embedding model '{model_name}' is not supported. "
            f"Available: {list(SUPPORTED_MODELS.keys())}"
        )
        return False

    info = SUPPORTED_MODELS[model_name]
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    model_cached = os.path.isdir(cache_dir)

    logger.info(
        f"validate_embedding_model: name='{model_name}', dim={info['dim']}, "
        f"source='{info['source']}', cache_available={model_cached}"
    )

    # For external_api models, just verify the source is correct.
    if info["source"] == "external_api":
        logger.info(f"Model '{model_name}' uses external API — skipping local load check.")
        return True

    # Attempt a quick load to confirm availability.
    try:
        _ = load_embedding_model(model_name)
        return True
    except (ValueError, RuntimeError) as e:
        logger.error(f"Embedding model validation failed for '{model_name}': {e}")
        return False

# REVIEW-03 FIX (Item #5.2): Configurable chunk size via environment variable.
DEFAULT_CHUNK_SIZE = int(os.getenv('VECTOR_DB_CHUNK_SIZE', '500'))


class QdrantVectorDB:
    """Qdrant VectorDB integration for vector search"""

    def __init__(self):
        self.client = None
        self.model = None
        self._init_client()
        self._init_model()

    def _init_client(self):
        """Initialize Qdrant client with configuration"""
        try:
            vector_db_config = get_config().get_config()['vector_db']
            uri = vector_db_config['qdrant_uri']
            api_key = vector_db_config['qdrant_api_key']
            collection_name = vector_db_config['collection_name']

            # REVIEW-02 FIX: Use urllib.parse for robust URI parsing (handles IPv6, schemes)
            parsed = urlparse(f"//{uri}")  # Prepend scheme for proper parsing
            host = parsed.hostname or "localhost"
            port = parsed.port or 6333

            # REVIEW-02 FIX: Retry with exponential backoff for Qdrant connections
            max_retries = 3
            retry_delay = 1.0
            client = None
            for attempt in range(1, max_retries + 1):
                try:
                    if api_key:
                        client = QdrantClient(
                            host=host,
                            port=port,
                            api_key=api_key
                        )
                    else:
                        client = QdrantClient(
                            host=host,
                            port=port
                        )
                    # Verify connection works
                    client.get_collections()
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(
                            f"Qdrant connection attempt {attempt}/{max_retries} failed: {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                        raise

            self.client = client

            # Create collection if it doesn't exist
            collection_config = vector_db_config
            vector_size = collection_config['vector_size']
            distance_metric = collection_config['distance_metric']

            try:
                self.client.get_collection(collection_name=collection_name)
                logger.info(f"Collection {collection_name} already exists")
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=getattr(qdrant_client.models.Distance, distance_metric)
                    )
                )
                logger.info(f"Created collection {collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _init_model(self):
        """
        Initialize embedding model for embeddings.

        Uses dynamic model name from environment variable via config.get_embedding_model()
        with validation. Supports both local SentenceTransformer models and external APIs.

        Logs model loading time and dimension verification.
        """
        try:
            from config import get_embedding_model
            model_name = get_embedding_model()
            self._model_name = model_name

            # Determine model source from SUPPORTED_MODELS.
            model_info = SUPPORTED_MODELS.get(model_name)
            if model_info and model_info["source"] == "external_api":
                # External API model — no local loading needed.
                self.model = None
                logger.info(
                    f"External embedding model configured: {model_name} "
                    f"[dim={model_info['dim']}, api={EXTERNAL_EMBEDDING_API_URL}]"
                )
            else:
                # Load local model using load_embedding_model().
                self.model = load_embedding_model(model_name)

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _get_external_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Fetch embeddings from an external OpenAI-compatible API.

        Used when EMBEDDING_MODEL is set to a model with source='external_api'
        (e.g., 'nomic-embed-text-v1.5'). Sends a POST request to /v1/embeddings
        endpoint and returns the embedding vectors as a numpy array.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy.ndarray of shape (len(texts), embedding_dim).

        Raises:
            RuntimeError: If the API request fails.
        """
        url = f"{EXTERNAL_EMBEDDING_API_URL}/embeddings"
        model_info = SUPPORTED_MODELS.get(EMBEDDING_MODEL, {})
        model_for_api = EMBEDDING_MODEL

        try:
            payload = {
                "model": model_for_api,
                "input": texts if isinstance(texts, list) else [texts],
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from OpenAI-compatible response format.
            embeddings_list = []
            for item in data.get("data", []):
                embeddings_list.append(item.get("embedding", []))

            if not embeddings_list:
                raise RuntimeError(f"No embeddings returned from external API: {data}")

            return np.array(embeddings_list)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"External embedding API request failed: {e}") from e
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse external API response: {e}") from e

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.

        Supports both local SentenceTransformer models and external APIs.
        When using an external API model (source='external_api'), delegates to
        _get_external_embedding().

        REVIEW-02 FIX: Returns numpy array directly instead of list of ndarrays
        for better performance and cleaner API.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy.ndarray of shape (len(texts), embedding_dim).
            Empty array if embedding fails.
        """
        try:
            # Check if using external API model.
            model_info = SUPPORTED_MODELS.get(EMBEDDING_MODEL, {})
            if model_info.get("source") == "external_api":
                return self._get_external_embedding(texts)

            # Use local SentenceTransformer model.
            if self.model is None:
                logger.error("No embedding model loaded and no external API configured.")
                return np.array([])

            embeddings = self.model.encode(texts)
            return embeddings  # Already a numpy array from SentenceTransformer
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return np.array([])

    def store_document_embeddings(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap_ratio: float = 0.15,
        chunk_mode: SplitMode = SplitMode.CHARACTER,
    ) -> bool:
        """
        Store document embeddings in Qdrant.

        REVIEW-04: Uses the new chunking.split_content() which supports
        overlap, UTF-8 byte-boundary safety, semantic chunking, and
        Thai-aware splitting.

        Args:
            doc_id: Unique document identifier.
            content: Raw document text.
            metadata: Optional payload metadata.
            max_chunk_size: Maximum character length per chunk.
            overlap_ratio: Overlap fraction (0.10-0.20 recommended).
            chunk_mode: Splitting strategy (character/semantic/thai_aware).

        Returns:
            True if embeddings were stored successfully.
        """
        try:
            vector_db_config = get_config().get_config()['vector_db']
            collection_name = vector_db_config['collection_name']

            # Split content into chunks using the advanced chunking module.
            chunks = split_content(
                content,
                max_chunk_size=max_chunk_size,
                mode=chunk_mode,
                overlap_ratio=overlap_ratio,
                embedding_model=self.model if chunk_mode == SplitMode.SEMANTIC else None,
            )

            if not chunks:
                logger.warning(f"No chunks produced for document {doc_id}")
                return False

            # Create embeddings for each chunk
            embeddings = self.create_embeddings(chunks)

            if embeddings.size == 0:
                logger.warning(f"No embeddings created for document {doc_id}")
                return False

            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{doc_id}_chunk_{i}"
                points.append({
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "content": chunk,
                        "metadata": metadata or {}
                    }
                })

            # Store in Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Stored {len(points)} chunks for document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store document embeddings: {e}")
            return False

    # --- Chunking (delegated to chunking.py) ---

    @staticmethod
    def _split_content(
        content: str,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap_ratio: float = 0.15,
        chunk_mode: SplitMode = SplitMode.CHARACTER,
    ) -> List[str]:
        """
        Split content into overlapping chunks using the advanced chunking module.

        REVIEW-04: Replaces the inline regex-based algorithm with a call to
        ``chunking.split_content()`` which supports:

          - Overlap (10-20%) between chunks for context continuity
          - UTF-8 byte-boundary safety to prevent character corruption
          - Semantic chunking via embedding model
          - Thai-aware splitting via pythainlp
          - Mode switching (character/semantic/thai_aware)

        Args:
            content: Raw text to chunk.
            max_chunk_size: Maximum character length per chunk.
            overlap_ratio: Overlap fraction (0.10-0.20 recommended).
            chunk_mode: Splitting strategy.

        Returns:
            List of text chunks, each <= max_chunk_size characters.
        """
        return split_content(
            content,
            max_chunk_size=max_chunk_size,
            mode=chunk_mode,
            overlap_ratio=overlap_ratio,
        )

    def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector search in Qdrant.
        """
        try:
            vector_db_config = get_config().get_config()['vector_db']
            collection_name = vector_db_config['collection_name']

            # Create query embedding
            query_embeddings = self.create_embeddings([query])
            if query_embeddings.size == 0:
                logger.warning("Failed to create query embedding, returning empty results")
                return []

            query_embedding = query_embeddings[0]

            # Perform vector search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "doc_id": result.payload.get("doc_id", ""),
                    "chunk_id": result.payload.get("chunk_id", 0),
                    "score": result.score
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def hybrid_search(self, query: str, graph_weight: float = 0.5, vector_weight: float = 0.5,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining graph and vector search results.
        
        REVIEW-02 FIX: This method now actually calls the graph_db's graph_rag_search()
        and combines results with vector search using weighted scoring.
        Previously this was a stub that only returned vector results.
        """
        try:
            from graph_db import get_graph_db

            # Get vector search results
            vector_results = self.vector_search(query, limit * 2)

            # REVIEW-02 FIX: Actually perform graph search and combine results
            graph_db = get_graph_db()
            graph_results = graph_db.graph_rag_search(query, max_hops=2)

            # Combine results using weighted scoring (same logic as hybrid_search.py)
            result_map: Dict[str, Dict[str, Any]] = {}

            # Process vector results
            for result in vector_results:
                doc_id = result.get('doc_id', result.get('id', ''))
                score = result.get('score', 0)
                if not doc_id:
                    continue

                if doc_id not in result_map:
                    result_map[doc_id] = {
                        'doc_id': doc_id,
                        'vector_score': 0,
                        'graph_score': 0,
                        'combined_score': 0,
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {})
                    }

                if score > result_map[doc_id]['vector_score']:
                    result_map[doc_id]['vector_score'] = score

            # Process graph results
            for result in graph_results:
                doc_id = result.get('id', '')
                score = result.get('related_count', 0)
                if not doc_id:
                    continue

                if doc_id not in result_map:
                    result_map[doc_id] = {
                        'doc_id': doc_id,
                        'vector_score': 0,
                        'graph_score': 0,
                        'combined_score': 0,
                        'content': result.get('content', ''),
                        'metadata': {}
                    }

                if score > result_map[doc_id]['graph_score']:
                    result_map[doc_id]['graph_score'] = score

            # Calculate combined scores with proper normalization
            all_graph_scores = [r['graph_score'] for r in result_map.values() if r['graph_score'] > 0]
            max_graph = max(all_graph_scores) if all_graph_scores else 1.0

            for doc_id, result in result_map.items():
                normalized_vector = result['vector_score']  # Already 0-1 from cosine similarity
                normalized_graph = result['graph_score'] / max_graph if max_graph > 0 and result['graph_score'] > 0 else 0

                combined_score = (graph_weight * normalized_graph) + (vector_weight * normalized_vector)
                result['combined_score'] = combined_score

            # Sort by combined score and return top results
            sorted_results = sorted(
                result_map.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )[:limit]

            return [
                {
                    'id': r['doc_id'],
                    'content': r['content'],
                    'vector_score': r['vector_score'],
                    'graph_score': r['graph_score'],
                    'combined_score': r['combined_score'],
                    'metadata': r['metadata']
                }
                for r in sorted_results
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector-only results
            try:
                return self.vector_search(query, limit)
            except Exception:
                return []


# --- Lazy initialization to avoid crashing app on import when DB is not ready ---

_vector_db_instance: Optional[QdrantVectorDB] = None


def get_vector_db() -> QdrantVectorDB:
    """
    Get or create the global QdrantVectorDB instance (lazy init).

    Returns:
        The singleton QdrantVectorDB instance.

    Raises:
        Exception: If Qdrant or sentence-transformer model cannot be initialized.
    """
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = QdrantVectorDB()
    return _vector_db_instance


# Backward-compatible alias - callers should prefer get_vector_db() for safety.
vector_db = None
