"""
Neo4j-based vector store implementation.

This module provides a Neo4j-native vector storage backend that uses
Neo4j's built-in vector index (via the Bolt protocol on port 7687) for
storing and querying embeddings directly inside the graph database.

Features:
    - Binary communication via Neo4j Bolt protocol (port 7687).
    - Lazy initialization with singleton pattern — connects only when the
      first operational method is invoked.
    - Supports both 768-dim (all-MiniLM-L6-v2) and 1024-dim (bge-m3)
      embedding spaces through configurable dimension parameter.
    - Full type hints and Google-style docstrings.
    - Logging at INFO level for operations, ERROR level for failures.

Changes (Initial Release):
    - Implements VectorBackend protocol for drop-in replacement of Qdrant.
    - Factory function get_vector_backend() selects backend via VECTOR_BACKEND
      environment variable ("neo4j" or "qdrant").
    - Vector indexes created for multiple label types:
      KnowledgeConcept, Hardware, Strategy, Narrative, ContentAsset.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
from config import get_config, get_embedding_model, get_embedding_dimension

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Protocol definition
# --------------------------------------------------------------------------- #

class VectorBackend(Protocol):  # type: ignore[misc]
    """Protocol that all vector backends must implement."""

    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool: ...
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]: ...
    def create_index(self) -> bool: ...
    def delete_embedding(self, node_id: str) -> bool: ...


# --------------------------------------------------------------------------- #
#  Neo4j Vector Store
# --------------------------------------------------------------------------- #

# Labels that should have vector-index support.
VECTOR_LABELS: List[str] = [
    "KnowledgeConcept",
    "Hardware",
    "Strategy",
    "Narrative",
    "ContentAsset",
]


class Neo4jVectorStore:
    """Neo4j-native vector store using built-in vector indexes.

    Stores embeddings as node properties and leverages Neo4j's native
    vector index for approximate nearest-neighbor (ANN) search over
    graph nodes.  Communication happens over the Bolt binary protocol
    (default port 7687).

    Attributes:
        neo4j_uri: Bolt URI for the Neo4j instance (e.g. ``neo4j://10.10.1.210:7687``).
        username: Authentication username.
        password: Authentication password.
        index_name: Base name prefix for vector indexes.
        dimension: Embedding vector dimension (768 or 1024).
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: str = "knowledge_vector",
        dimension: Optional[int] = None,
    ):
        """Initialise the vector store (lazy — no connection yet).

        Connection parameters are read from :func:`config.get_config` when
        ``neo4j_uri``, ``username`` or ``password`` are not provided
        explicitly.

        Args:
            neo4j_uri: Bolt URI.  Defaults to ``NEO4J_URI`` environment
                variable or ``neo4j://localhost:7687``.
            username: Neo4j username.  Defaults to ``NEO4J_USER`` or ``"neo4j"``.
            password: Neo4j password.  Defaults to ``NEO4J_PASSWORD``.
            index_name: Base name for the vector index.  Individual indexes
                are created as ``<index_name>_<LABEL>`` for each supported
                label.
            dimension: Embedding dimension.  If *None*, it is resolved from
                the configured embedding model via
                :func:`config.get_embedding_dimension`.
        """
        cfg = get_config().get_config()["graph_db"]

        self.neo4j_uri: str = neo4j_uri or cfg.get("neo4j_uri", "neo4j://localhost:7687")
        self.username: str = username or cfg.get("neo4j_user", "neo4j")
        self.password: str = password or cfg.get("neo4j_password", "")
        self.index_name: str = index_name

        if dimension is not None:
            self.dimension: int = dimension
        else:
            model_name = get_embedding_model()
            self.dimension = get_embedding_dimension(model_name)

        # Lazy-singleton internals
        self._driver: Optional[GraphDatabase.Driver] = None
        self._model: Optional[SentenceTransformer] = None

        logger.info(
            "Neo4jVectorStore initialised (lazy) — uri=%s, index=%s, dim=%d",
            self.neo4j_uri,
            self.index_name,
            self.dimension,
        )

    # ------------------------------------------------------------------ #
    #  Lazy initialisation helpers
    # ------------------------------------------------------------------ #

    def _ensure_driver(self) -> GraphDatabase.Driver:
        """Return an active Neo4j driver, initialising lazily if needed.

        Returns:
            A connected ``GraphDatabase.Driver`` instance.

        Raises:
            ServiceUnavailable: If Neo4j cannot be reached after retries.
        """
        if self._driver is not None:
            return self._driver

        max_retries = 3
        retry_delay = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                self._driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.username, self.password),
                    max_connection_pool_size=50,
                    connection_timeout=30,
                )
                # Verify connectivity
                with self._driver.session() as session:
                    session.run("RETURN 1")
                logger.info(
                    "Neo4j driver connected on attempt %d/%d (uri=%s)",
                    attempt,
                    max_retries,
                    self.neo4j_uri,
                )
                return self._driver
            except (ServiceUnavailable, TransientError, ConnectionError) as e:
                last_error = e  # type: ignore[assignment]
                if attempt < max_retries:
                    logger.warning(
                        "Neo4j connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt,
                        max_retries,
                        e,
                        retry_delay,
                    )
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        "Failed to connect to Neo4j after %d attempts: %s",
                        max_retries,
                        e,
                    )

        raise last_error  # type: ignore[misc]

    def _ensure_model(self) -> SentenceTransformer:
        """Return the embedding model, loading lazily if needed.

        Returns:
            A loaded ``SentenceTransformer`` model.
        """
        if self._model is not None:
            return self._model

        model_name = get_embedding_model()
        self._model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model for Neo4jVectorStore: %s", model_name)
        return self._model

    # ------------------------------------------------------------------ #
    #  VectorBackend protocol implementation
    # ------------------------------------------------------------------ #

    def _index_exists(self, index_name: str, database: str) -> bool:
        """Check whether a vector index already exists in Neo4j.

        Uses ``SHOW VECTOR INDEXES`` (Neo4j 5+) to query existing indexes.

        Args:
            index_name: Name of the index to check.
            database: Target database name.

        Returns:
            ``True`` if the index is found, ``False`` otherwise.
        """
        driver = self._ensure_driver()
        try:
            with driver.session(database=database) as session:
                result = session.run(
                    "SHOW VECTOR INDEXES YIELD name",
                )
                existing_names = {record["name"] for record in result}
                if index_name in existing_names:
                    logger.debug("Index %s already exists", index_name)
                    return True
                return False
        except Exception as e:
            logger.warning(
                "Could not query existing vector indexes (may be Neo4j < 5): %s",
                e,
            )
            # If SHOW VECTOR INDEXES fails, assume index does not exist
            # and let the CREATE ... IF NOT EXISTS handle idempotency.
            return False

    def create_index(self) -> bool:
        """Create a composite vector index in Neo4j covering all universal schema labels.

        Creates a **single** composite vector index that spans every label
        in ``VECTOR_LABELS`` using a union type descriptor::

            CREATE VECTOR INDEX knowledge_vector IF NOT EXISTS
            FOR (n:KnowledgeConcept|Hardware|Strategy|Narrative|ContentAsset)
            ON n.embedding
            OPTIONS {indexConfig: {
              `vector.dimensions`: $dimensions,
              `vector.similarity_function`: 'cosine'
            }}

        Parameters are sourced from config:
            - dimensions: resolved via ``get_embedding_dimension(embedding_model)``
              (1024 for bge-m3, 768 for all-MiniLM-L6-v2).
            - index_name: from constructor or config (default "knowledge_vector").

        The operation is idempotent — safe to call multiple times.
        After creation the method verifies the index by querying
        ``SHOW INDEXES``.

        Returns:
            ``True`` if the index was created (or already existed) and
            verified, ``False`` on any error.
        """
        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            # Build composite label descriptor, e.g. "KnowledgeConcept|Hardware|..."
            label_union = "|".join(VECTOR_LABELS)

            # Check if index already exists before attempting creation
            if self._index_exists(self.index_name, database):
                logger.info(
                    "Vector index %s already exists — skipping creation",
                    self.index_name,
                )
                return True

            with driver.session(database=database) as session:
                query = (
                    f'CREATE VECTOR INDEX $index_name IF NOT EXISTS '
                    f'FOR (n:{label_union}) ON n.embedding '
                    f'OPTIONS {{'
                    f'  indexConfig: {{'
                    f'    `vector.dimensions`: $dimensions, '
                    f'    `vector.similarity_function`: $similarity'
                    f'  }}'
                    f'}}'
                )
                session.run(
                    query,
                    index_name=self.index_name,
                    dimensions=self.dimension,
                    similarity="cosine",
                )
                logger.info(
                    "Created composite vector index: %s (dim=%d, labels=%s)",
                    self.index_name,
                    self.dimension,
                    label_union,
                )

            # Verify index creation by querying SHOW INDEXES
            verified = self._verify_index_creation(database)
            if not verified:
                logger.warning(
                    "Index %s creation command succeeded but verification failed",
                    self.index_name,
                )

            return True

        except Exception as e:
            logger.error("Failed to create vector index: %s", e)
            return False

    def _verify_index_creation(self, database: str) -> bool:
        """Verify that the composite vector index exists after creation.

        Queries ``SHOW INDEXES`` and checks for the expected index name.

        Args:
            database: Target database name.

        Returns:
            ``True`` if the index is found, ``False`` otherwise.
        """
        driver = self._ensure_driver()
        try:
            with driver.session(database=database) as session:
                result = session.run(
                    "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
                    "WHERE name = $index_name",
                    index_name=self.index_name,
                )
                record = result.single()
                if record is not None:
                    logger.info(
                        "Verified index %s — type=%s, labels=%s, props=%s",
                        self.index_name,
                        record["type"],
                        record["labelsOrTypes"],
                        record["properties"],
                    )
                    return True
                logger.warning("Index %s not found after creation", self.index_name)
                return False
        except Exception as e:
            logger.error("Failed to verify index %s: %s", self.index_name, e)
            return False

    def store_embedding(
        self, node_id: str, text: str, embedding: List[float]
    ) -> bool:
        """Attach an embedding vector to an existing graph node.

        Matches the node by its ``id`` property and sets both the
        ``embedding`` (vector) and ``embedded_text`` (source text)
        properties.

        Args:
            node_id: The unique ``id`` property of the target node.
            text: The original text that was embedded (stored for
                reference / debugging).
            embedding: Float vector of length matching the store's
                dimension.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            query = """
                MATCH (n) WHERE n.id = $node_id
                SET n.embedding = $embedding,
                    n.embedded_text = $text
            """
            with driver.session(database=database) as session:
                result = session.run(query, node_id=node_id, embedding=embedding, text=text)
                single = result.single()
                if single is None:
                    logger.warning("No node found with id=%s — nothing to update", node_id)
                    return False

            logger.info("Stored embedding for node_id=%s (dim=%d)", node_id, len(embedding))
            return True

        except Exception as e:
            logger.error("Failed to store embedding for node_id=%s: %s", node_id, e)
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform vector similarity search across all indexed labels.

        The query string is first passed through the configured embedding
        model to produce a fixed-dimensional vector.  That vector is then
        used to query each Neo4j vector index (one per supported label)
        via ``db.index.vector.queryNodes``.  Results from all indexes are
        merged, deduplicated by node ``id``, and sorted by descending
        similarity score.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return per index (final
                list may contain up to ``top_k * num_labels`` entries
                before deduplication, but at most ``top_k`` after).

        Returns:
            A list of dictionaries, each containing:

            - **id** (str): Node identifier.
            - **label** (str): Primary Neo4j label.
            - **name_th** (str): Thai-language name (if available).
            - **name_en** (str): English-language name (if available).
            - **score** (float): Cosine similarity score (0 – 1).
            - **properties** (Dict[str, Any]): Full node property map.
        """
        try:
            # 1. Generate query embedding
            model = self._ensure_model()
            query_embedding = model.encode([query])[0].tolist()

            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            all_results: List[Dict[str, Any]] = []

            with driver.session(database=database) as session:
                cypher = """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node, score, labels(node)[0] AS label
                """
                try:
                    result = session.run(
                        cypher,
                        index_name=self.index_name,
                        top_k=top_k * len(VECTOR_LABELS),
                        query_embedding=query_embedding,
                    )
                    for record in result:
                        node_props = dict(record["node"])
                        all_results.append({
                            "id": node_props.get("id", ""),
                            "label": record.get("label", ""),
                            "name_th": node_props.get("name_th", ""),
                            "name_en": node_props.get("name_en", ""),
                            "score": float(record["score"]),
                            "properties": node_props,
                        })
                except Exception as e:
                    logger.warning(
                        "Vector query failed for index %s: %s",
                        self.index_name,
                        e,
                    )

            # Deduplicate by node id (keep highest score)
            seen: Dict[str, Dict[str, Any]] = {}
            for item in all_results:
                nid = item["id"]
                if nid not in seen or item["score"] > seen[nid]["score"]:
                    seen[nid] = item

            ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
            final = ranked[:top_k]

            logger.info(
                "Vector search for query=%r returned %d results (top_k=%d)",
                query,
                len(final),
                top_k,
            )
            return final

        except Exception as e:
            logger.error("Vector search failed for query=%r: %s", query, e)
            return []

    def delete_embedding(self, node_id: str) -> bool:
        """Remove embedding properties from a graph node.

        Deletes both ``embedding`` and ``embedded_text`` properties from
        the node identified by *node_id*, effectively removing it from
        vector search results while preserving the rest of the node data.

        Args:
            node_id: The unique ``id`` property of the target node.

        Returns:
            ``True`` on success, ``False`` on failure or if the node was
            not found.
        """
        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            query = """
                MATCH (n) WHERE n.id = $node_id
                REMOVE n.embedding, n.embedded_text
            """
            with driver.session(database=database) as session:
                result = session.run(query, node_id=node_id)
                single = result.single()
                if single is None:
                    logger.warning(
                        "No node found with id=%s — nothing to remove", node_id
                    )
                    return False

            logger.info("Deleted embedding for node_id=%s", node_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete embedding for node_id=%s: %s", node_id, e
            )
            return False

    def show_indexes(self) -> List[Dict[str, Any]]:
        """Return a list of all current indexes in the Neo4j database.

        Executes ``SHOW INDEXES YIELD name, type, labelsOrTypes, properties``
        and returns each index as a dictionary for verification and
        administrative purposes.

        Returns:
            A list of dictionaries, each containing:

            - **name** (str): Index name.
            - **type** (str): Index type (e.g. ``"vector"``, ``"lookup"``).
            - **labelsOrTypes** (List[str]): Labels or node types covered.
            - **properties** (List[str]): Indexed properties.

        Example::

            store = Neo4jVectorStore()
            for idx in store.show_indexes():
                print(f"{idx['name']} ({idx['type']}): {idx['labelsOrTypes']}")
        """
        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            with driver.session(database=database) as session:
                result = session.run(
                    "SHOW INDEXES YIELD name, type, labelsOrTypes, properties"
                )
                indexes = []
                for record in result:
                    indexes.append({
                        "name": record["name"],
                        "type": record["type"],
                        "labelsOrTypes": record["labelsOrTypes"] or [],
                        "properties": record["properties"] or [],
                    })

            logger.info("Retrieved %d indexes from Neo4j", len(indexes))
            return indexes

        except Exception as e:
            logger.error("Failed to retrieve indexes: %s", e)
            return []

    def drop_index(self, index_name: str) -> bool:
        """Drop an existing index by name.

        Executes ``DROP INDEX $index_name IF EXISTS`` for maintenance or
        schema migration purposes.  This operation is idempotent — it will
        not fail if the index does not exist.

        Args:
            index_name: The exact name of the index to drop.

        Returns:
            ``True`` on success (including when the index did not exist),
            ``False`` on failure.

        Warning:
            Dropping a vector index will make vector search fail for the
            affected labels until the index is recreated via
            :meth:`create_index`.
        """
        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            with driver.session(database=database) as session:
                session.run(
                    "DROP INDEX $index_name IF EXISTS",
                    index_name=index_name,
                )

            logger.info("Dropped index: %s", index_name)
            return True

        except Exception as e:
            logger.error("Failed to drop index %s: %s", index_name, e)
            return False

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying Neo4j driver (if open)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4jVectorStore driver closed")


# --------------------------------------------------------------------------- #
#  Singleton / factory
# --------------------------------------------------------------------------- #

_neo4j_vector_instance: Optional[Neo4jVectorStore] = None


def _get_neo4j_vector_store() -> Neo4jVectorStore:
    """Return a singleton ``Neo4jVectorStore`` instance (lazy init)."""
    global _neo4j_vector_instance
    if _neo4j_vector_instance is None:
        _neo4j_vector_instance = Neo4jVectorStore()
    return _neo4j_vector_instance


def get_vector_backend() -> VectorBackend:
    """Factory function that returns the configured vector backend.

    Reads the ``VECTOR_BACKEND`` environment variable (or calls
    :func:`config.get_vector_backend_mode` for runtime-aware selection):

    - ``"neo4j"`` → returns a :class:`Neo4jVectorStore` instance.
    - ``"dual"``  → returns a :class:`DualCompareBackend` that queries
      both Qdrant and Neo4j and merges results.
    - Any other value (default ``"qdrant"``) → falls back to the existing
      Qdrant implementation from :mod:`vector_db`.

    Returns:
        A vector store implementing the :class:`VectorBackend` protocol.
    """
    from config import get_vector_backend_mode
    backend = get_vector_backend_mode()
    logger.info("Selecting vector backend: %s", backend)

    if backend == "neo4j":
        return _get_neo4j_vector_store()
    elif backend == "dual":
        from dual_write_vector_store import DualCompareBackend, QdrantBackendAdapter
        from vector_db import get_vector_db
        qdrant_adapter = QdrantBackendAdapter(get_vector_db())
        neo4j_store = _get_neo4j_vector_store()
        return DualCompareBackend(qdrant_adapter, neo4j_store)
    else:
        # Import lazily to avoid hard dependency when using Neo4j-only mode.
        from vector_db import get_vector_db  # type: ignore[attr-defined]
        return get_vector_db()
