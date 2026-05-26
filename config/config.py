"""
Configuration module for hybrid search system.
This module handles configuration for both GraphDB (Neo4j) and VectorDB (Qdrant) parameters.
Uses lazy initialization to avoid crashing the app when DB credentials are unavailable.

Changes (Pipeline Improvement):
  - EMBEDDING_MODEL now configurable via environment variable with validation.
  - Neo4j connection pool size, timeout, and retry settings added.
  - Redis-based rate limiting configuration added.
  - Metrics configuration added for Prometheus-compatible /metrics endpoint.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)


# --- .env file support (loads once at module import) ---
def _load_dotenv() -> None:
    """Attempt to load environment variables from a .env file using python-dotenv."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv()
        logger.debug("Loaded .env file successfully")
    except ImportError:
        logger.debug(
            "python-dotenv not installed; skipping .env loading. "
            "Install with: pip install python-dotenv"
        )


_load_dotenv()


# --- Safe parsing helpers for environment variables ---

def _safe_int(key: str, default: int) -> int:
    """
    Safely parse an integer from an environment variable.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is missing or invalid.

    Returns:
        The parsed integer or the default value.
    """
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


def _safe_float(key: str, default: float) -> float:
    """
    Safely parse a float from an environment variable.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is missing or invalid.

    Returns:
        The parsed float or the default value.
    """
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


# --- Supported embedding models for validation ---
SUPPORTED_EMBEDDING_MODELS: Set[str] = {
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-roberta-large-v1",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-mpnet-base-v2",
    "multi-qa-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-m3",
    "nomic-embed-text-v1.5",
}


def validate_embedding_model(model_name: str) -> bool:
    """
    Validate that the given embedding model name is supported.

    Args:
        model_name: The model name to validate.

    Returns:
        True if the model is in the supported list or follows a valid HuggingFace pattern.
    """
    # Allow exact matches from the supported set
    if model_name in SUPPORTED_EMBEDDING_MODELS:
        return True
    # Allow HuggingFace-style paths (org/model or model_name)
    if "/" in model_name or (model_name.isidentifier() or all(c.isalnum() or c in "-_./" for c in model_name)):
        logger.info(f"Embedding model '{model_name}' not in predefined list but has valid format. Proceeding.")
        return True
    return False


# --- Embedding Dimension Mapping ---
# Dynamic mapping of embedding model names to their vector dimensions.
# Falls back to 1024 for unknown models (e.g., bge-m3).
EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "all-MiniLM-L6-v2": 768,
    "bge-m3": 1024,
    "nomic-embed-text-v1.5": 768,
}


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    """
    Get the embedding dimension for a given model name.

    Args:
        model_name: The embedding model name. Defaults to EMBEDDING_MODEL env var.

    Returns:
        The embedding dimension in integers, defaulting to 1024 for unknown models.
    """
    if model_name is None:
        model_name = get_embedding_model()
    dim = EMBEDDING_DIMENSIONS.get(model_name, 1024)
    logger.debug(f"Embedding dimension for '{model_name}': {dim}")
    return dim


# --- Vector Backend Selection ---
# Selects the vector database backend: "qdrant" (default), "neo4j", or "dual".
#   - "qdrant": Read from Qdrant only.
#   - "neo4j":  Read from Neo4j vector index only.
#   - "dual":   Compare results from both backends (DualCompareBackend).
# The value is re-read on each call to get_vector_backend_mode() so that
# changing the environment variable does not require a service restart.
vector_backend = os.getenv("VECTOR_BACKEND", "qdrant")  # "qdrant" | "neo4j" | "dual"

_VALID_VECTOR_BACKENDS: Set[str] = {"qdrant", "neo4j", "dual"}


def get_vector_backend_mode() -> str:
    """Return the current vector backend mode, re-reading from the environment.

    This allows runtime switching without a service restart -- simply update
    the VECTOR_BACKEND environment variable (or call
    HybridSearchEngine.switch_backend() which sets it under the hood).

    Returns:
        One of "qdrant", "neo4j", or "dual".
        Falls back to "qdrant" if the value is unrecognised.
    """
    mode = os.getenv("VECTOR_BACKEND", "qdrant")
    if mode not in _VALID_VECTOR_BACKENDS:
        logger.warning(
            "Unknown VECTOR_BACKEND=%r, falling back to 'qdrant'", mode
        )
        return "qdrant"
    return mode


# --- Domain Detection Mode ---
# Controls how document domains are detected: "auto" (heuristic), "manual" (user-set), or "llm" (AI-assisted).
domain_detection_mode = os.getenv("DOMAIN_DETECTION_MODE", "auto")  # "auto", "manual", "llm"


# --- LLM Server Configuration ---
llm_config = {
    # Base URL of the LLM inference server (e.g., Ollama, vLLM, custom endpoint).
    "server_url": os.getenv("LLM_SERVER_URL", "http://localhost:8080"),
    # Name of the model to use for extraction and reasoning tasks.
    "model": os.getenv("LLM_MODEL", "Qwen3.6-27B-MTP-GGUF"),
    # Sampling temperature: 0.0 for deterministic output, higher for creativity.
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    # Enable extended reasoning (Chain-of-Thought) mode when supported by the model.
    "enable_thinking": os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
    # Maximum number of tokens the LLM can generate in a single response.
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
    # Request timeout in seconds for LLM API calls.
    "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
}


# --- VRAM Management Configuration ---
vram_config = {
    # Sequential mode: load models one at a time to avoid GPU memory exhaustion.
    "sequential_mode": os.getenv("VRAM_SEQUENTIAL_MODE", "true").lower() == "true",
    # Safe VRAM usage limit in GB — triggers model unload when exceeded.
    "safe_limit_gb": int(os.getenv("VRAM_SAFE_LIMIT_GB", "22")),
    # Timeout in seconds for loading a model into GPU memory.
    "model_load_timeout": int(os.getenv("VRAM_LOAD_TIMEOUT", "300")),
    # Timeout in seconds for unloading a model from GPU memory.
    "model_unload_timeout": int(os.getenv("VRAM_UNLOAD_TIMEOUT", "60")),
}


# --- Embedding Model Configuration ---
def get_embedding_model() -> str:
    """
    Get the embedding model name from environment variable with validation.

    Returns:
        The validated embedding model name string.

    Raises:
        ValueError: If the model name is not supported or has an invalid format.
    """
    model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    if not validate_embedding_model(model_name):
        raise ValueError(
            f"Unsupported embedding model: '{model_name}'. "
            f"Supported models include: {', '.join(sorted(SUPPORTED_EMBEDDING_MODELS))}. "
            f"Set EMBEDDING_MODEL environment variable to change."
        )
    logger.info(f"Using embedding model: {model_name}")
    return model_name


class HybridSearchConfig:
    """Configuration class for hybrid search system"""

    def __init__(self, require_neo4j_password: bool = False):
        """
        Initialize the hybrid search configuration.

        Args:
            require_neo4j_password: If True and NEO4J_PASSWORD is not set,
                raises ValueError. Defaults to False for graceful degradation.
        """
        # GraphDB (Neo4j) configuration
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        if not neo4j_password:
            if require_neo4j_password:
                raise ValueError(
                    "NEO4J_PASSWORD environment variable is required. "
                    "Do not store passwords in code."
                )
            logger.warning(
                "NEO4J_PASSWORD not set. GraphDB operations will fail until "
                "credentials are provided via environment variables or .env file."
            )
        self.graph_db_config = {
            'neo4j_uri': os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
            'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
            'neo4j_password': neo4j_password,
            'neo4j_database': os.getenv('NEO4J_DATABASE', 'neo4j'),
            # Pipeline Improvement: Connection pool and timeout settings
            'max_connection_pool_size': _safe_int('NEO4J_MAX_CONNECTION_POOL_SIZE', 50),
            'connection_timeout': _safe_float('NEO4J_CONNECTION_TIMEOUT', 30.0),
            'connection_retry_attempts': _safe_int('NEO4J_CONNECTION_RETRY_ATTEMPTS', 3),
            'connection_retry_delay': _safe_float('NEO4J_CONNECTION_RETRY_DELAY', 1.0),
            'graph_rag_config': {
                'max_hops': _safe_int('GRAPH_MAX_HOPS', 2),
                'max_nodes': _safe_int('GRAPH_MAX_NODES', 100),
                'relationship_threshold': _safe_float('GRAPH_RELATIONSHIP_THRESHOLD', 0.5)
            }
        }

        # VectorDB (Qdrant) configuration
        self.vector_db_config = {
            'qdrant_uri': os.getenv('QDRANT_URI', 'localhost:6333'),
            'qdrant_api_key': os.getenv('QDRANT_API_KEY', None),
            'collection_name': os.getenv('QDRANT_COLLECTION', 'markdown_embeddings'),
            'vector_size': _safe_int('VECTOR_SIZE', 768),
            'distance_metric': os.getenv('DISTANCE_METRIC', 'Cosine'),
            'vector_db_config': {
                'batch_size': _safe_int('VECTOR_BATCH_SIZE', 100),
                'max_retries': _safe_int('VECTOR_MAX_RETRIES', 3)
            }
        }

        # Hybrid search configuration
        self.hybrid_config = {
            'graph_weight': _safe_float('GRAPH_WEIGHT', 0.5),
            'vector_weight': _safe_float('VECTOR_WEIGHT', 0.5),
            'search_limit': _safe_int('SEARCH_LIMIT', 10),
            'min_score_threshold': _safe_float('MIN_SCORE_THRESHOLD', 0.3)
        }

        # Chunking configuration (REVIEW-04)
        self.chunking_config = {
            'max_chunk_size': _safe_int('VECTOR_DB_CHUNK_SIZE', 500),
            'overlap_ratio': _safe_float('CHUNK_OVERLAP_RATIO', 0.15),
            'overlap_chars': _safe_int('CHUNK_OVERLAP_CHARS', 75),
            'chunk_mode': os.getenv('CHUNK_MODE', 'character'),
            'similarity_threshold': _safe_float('SEMANTIC_SIMILARITY_THRESHOLD', 0.6),
        }

        # Redis-based rate limiting configuration (Pipeline Improvement)
        self.rate_limit_config = {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': _safe_int('REDIS_PORT', 6379),
            'redis_password': os.getenv('REDIS_PASSWORD', None),
            'redis_db': _safe_int('REDIS_DB', 0),
            'rate_limit_max_requests': _safe_int('RATE_LIMIT_MAX_REQUESTS', 100),
            'rate_limit_window_seconds': _safe_int('RATE_LIMIT_WINDOW_SECONDS', 60),
            'use_redis_rate_limiting': os.getenv('USE_REDIS_RATE_LIMITING', 'true').lower() == 'true',
        }

        # Metrics configuration (Pipeline Improvement)
        self.metrics_config = {
            'enable_metrics': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'metrics_port': _safe_int('METRICS_PORT', 9090),
            'metrics_path': os.getenv('METRICS_PATH', '/metrics'),
        }

        # Parallel processing configuration (Pipeline Improvement)
        self.parallel_config = {
            'max_workers': _safe_int('PARALLEL_MAX_WORKERS', 0),  # 0 = auto-detect
            'use_process_pool': os.getenv('USE_PROCESS_POOL', 'auto').lower(),  # 'auto', 'true', 'false'
            'chunk_batch_size': _safe_int('PARALLEL_CHUNK_BATCH_SIZE', 10),
        }

        # LLM server configuration
        self.llm_config = {
            "server_url": os.getenv("LLM_SERVER_URL", "http://localhost:8080"),
            "model": os.getenv("LLM_MODEL", "Qwen3.6-27B-MTP-GGUF"),
            "temperature": _safe_float("LLM_TEMPERATURE", 0.0),
            "enable_thinking": os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
            "max_tokens": _safe_int("LLM_MAX_TOKENS", 4096),
            "timeout": _safe_int("LLM_TIMEOUT", 120),
        }

        # VRAM management configuration
        self.vram_config = {
            "sequential_mode": os.getenv("VRAM_SEQUENTIAL_MODE", "true").lower() == "true",
            "safe_limit_gb": _safe_int("VRAM_SAFE_LIMIT_GB", 22),
            "model_load_timeout": _safe_int("VRAM_LOAD_TIMEOUT", 300),
            "model_unload_timeout": _safe_int("VRAM_UNLOAD_TIMEOUT", 60),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return {
            'graph_db': self.graph_db_config,
            'vector_db': self.vector_db_config,
            'hybrid': self.hybrid_config,
            'chunking': self.chunking_config,
            'rate_limit': self.rate_limit_config,
            'metrics': self.metrics_config,
            'parallel': self.parallel_config,
            'llm': self.llm_config,
            'vram': self.vram_config,
        }


# --- Lazy initialization to avoid crashing app on import when DB is not ready ---

_config_instance: Optional[HybridSearchConfig] = None


def get_config() -> HybridSearchConfig:
    """
    Get or create the global HybridSearchConfig instance (lazy init).

    Returns:
        The singleton HybridSearchConfig instance.

    Raises:
        ValueError: If required environment variables are missing and
            require_neo4j_password is True.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = HybridSearchConfig()
    return _config_instance


def is_config_ready() -> bool:
    """Check whether the configuration can be created without raising."""
    return bool(os.getenv('NEO4J_PASSWORD'))
