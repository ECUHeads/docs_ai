"""
Dual-Write Vector Storage Module.

Provides a wrapper that writes embeddings to both Qdrant and Neo4j
simultaneously, while reading from a single configured backend.

Features:
    - Non-blocking writes: one backend failure does not prevent the other.
    - Full logging of all dual-write operations.
    - Configurable read backend via VECTOR_BACKEND environment variable.
    - Parity validation to compare search results between backends.
    - Optional periodic parity-check scheduler.

Environment Variables:
    VECTOR_BACKEND: Read backend selection ("qdrant" or "neo4j", default "qdrant").
    PARITY_CHECK_INTERVAL: Seconds between automated parity checks (default 3600).
"""

import os
import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from vector_db import get_vector_db, QdrantVectorDB
from neo4j_vector_store import Neo4jVectorStore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Qdrant Adapter  –  exposes the same signature as Neo4jVectorStore
# --------------------------------------------------------------------------- #

class QdrantBackendAdapter:
    """Adapts QdrantVectorDB to match the VectorBackend protocol.

    Wraps the existing ``QdrantVectorDB`` and provides a
    ``store_embedding(node_id, text, embedding)`` method that delegates
    to ``store_document_embeddings`` under the hood.
    """

    def __init__(self, qdrant_db: Optional[QdrantVectorDB] = None):
        self._db = qdrant_db or get_vector_db()

    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool:
        """Store a single embedding in Qdrant.

        Creates a one-chunk document so the embedding lands in Qdrant
        with ``node_id`` as the point identifier.

        Args:
            node_id: Unique identifier for this embedding.
            text: Source text associated with the vector.
            embedding: Float vector (must match collection dimension).

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            from qdrant_client.models import PointStruct

            vector_db_config = self._db.client.get_collection(
                collection_name=self._db.client.get_collections().collections[0].name
            ) if self._db.client else None

            # Use the configured collection name from config.
            from config import get_config
            collection_name = get_config().get_config()["vector_db"]["collection_name"]

            point = PointStruct(
                id=node_id,
                vector=embedding,
                payload={
                    "doc_id": node_id,
                    "chunk_id": 0,
                    "content": text,
                    "metadata": {},
                },
            )
            self._db.client.upsert(
                collection_name=collection_name,
                points=[point],
            )
            logger.debug("QdrantAdapter stored embedding for node_id=%s", node_id)
            return True
        except Exception as e:
            logger.error("QdrantAdapter failed to store embedding for node_id=%s: %s", node_id, e)
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Delegate to QdrantVectorDB.vector_search.

        Returns results normalised to the common schema:
        ``{"id", "content", "score", "metadata"}``.
        """
        try:
            raw = self._db.vector_search(query, limit=top_k)
            return [
                {
                    "id": r["id"],
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "metadata": {
                        "doc_id": r.get("doc_id", ""),
                        "chunk_id": r.get("chunk_id", 0),
                    },
                }
                for r in raw
            ]
        except Exception as e:
            logger.error("QdrantAdapter search failed for query=%r: %s", query, e)
            return []

    def create_index(self) -> bool:
        """No-op for Qdrant (collections auto-created on first write)."""
        return True

    def delete_embedding(self, node_id: str) -> bool:
        """Delete a single point from Qdrant by ID."""
        try:
            from config import get_config
            collection_name = get_config().get_config()["vector_db"]["collection_name"]
            self._db.client.delete(
                collection_name=collection_name,
                points_selector=[node_id],
            )
            logger.info("QdrantAdapter deleted embedding for node_id=%s", node_id)
            return True
        except Exception as e:
            logger.error("QdrantAdapter delete failed for node_id=%s: %s", node_id, e)
            return False


# --------------------------------------------------------------------------- #
#  Dual-Write Wrapper
# --------------------------------------------------------------------------- #

class DualWriteVectorStore:
    """Writes to both Qdrant and Neo4j; reads from the configured backend.

    Write Semantics
    ----------------
    Every call to ``store_embedding`` attempts to write to **both**
    backends independently.  A partial write (one success, one failure)
    is logged as a warning but still returns ``True`` so that the
    requesting pipeline is never blocked by a single backend outage.

    Read Semantics
    ---------------
    ``search()`` delegates to exactly **one** backend determined by the
    ``VECTOR_BACKEND`` environment variable at construction time
    (default ``"qdrant"``).

    Parameters
    ----------
    qdrant_backend : QdrantBackendAdapter, optional
        Pre-built adapter.  A new one is created via ``get_vector_db()``
        when not provided.
    neo4j_backend : Neo4jVectorStore, optional
        Pre-built Neo4j store.  A new default instance is created when
        not provided.
    read_backend : str
        ``"qdrant"`` or ``"neo4j"``.  Defaults to the ``VECTOR_BACKEND``
        environment variable.
    """

    def __init__(
        self,
        qdrant_backend: Optional[QdrantBackendAdapter] = None,
        neo4j_backend: Optional[Neo4jVectorStore] = None,
        read_backend: Optional[str] = None,
    ):
        self.qdrant_backend = qdrant_backend or QdrantBackendAdapter()
        self.neo4j_backend = neo4j_backend or Neo4jVectorStore()
        self.read_backend = read_backend or os.getenv("VECTOR_BACKEND", "qdrant")

        logger.info(
            "DualWriteVectorStore initialised — read_backend=%s",
            self.read_backend,
        )

    # ------------------------------------------------------------------ #
    #  Write operations  (dual-write)
    # ------------------------------------------------------------------ #

    def store_embedding(
        self, node_id: str, text: str, embedding: List[float]
    ) -> bool:
        """Write an embedding to **both** Qdrant and Neo4j.

        The two writes are executed sequentially but independently — a
        failure in one backend does not prevent the other from completing.

        Returns
        -------
        bool
            ``True`` if at least one backend accepted the write,
            ``False`` only when **both** fail.
        """
        start_ts = datetime.now(timezone.utc).isoformat()

        # --- Qdrant write ---
        qdrant_success = False
        try:
            qdrant_success = self.qdrant_backend.store_embedding(
                node_id, text, embedding
            )
        except Exception as e:
            logger.error(
                "Dual-write Qdrant exception for node_id=%s: %s", node_id, e
            )
            qdrant_success = False

        # --- Neo4j write ---
        neo4j_success = False
        try:
            neo4j_success = self.neo4j_backend.store_embedding(
                node_id, text, embedding
            )
        except Exception as e:
            logger.error(
                "Dual-write Neo4j exception for node_id=%s: %s", node_id, e
            )
            neo4j_success = False

        # --- Decision logic ---
        if qdrant_success and neo4j_success:
            logger.info(
                "Dual-write successful for node_id=%s [started=%s]",
                node_id,
                start_ts,
            )
            return True
        elif qdrant_success or neo4j_success:
            logger.warning(
                "Partial dual-write for node_id=%s — qdrant=%s, neo4j=%s [started=%s]",
                node_id,
                qdrant_success,
                neo4j_success,
                start_ts,
            )
            return True  # Partial success is acceptable
        else:
            logger.error(
                "Dual-write FAILED for node_id=%s — both backends declined [started=%s]",
                node_id,
                start_ts,
            )
            return False

    def delete_embedding(self, node_id: str) -> bool:
        """Delete an embedding from **both** backends.

        Returns ``True`` if at least one backend confirmed deletion.
        """
        qdrant_success = False
        neo4j_success = False

        try:
            qdrant_success = self.qdrant_backend.delete_embedding(node_id)
        except Exception as e:
            logger.error(
                "Dual-delete Qdrant exception for node_id=%s: %s", node_id, e
            )

        try:
            neo4j_success = self.neo4j_backend.delete_embedding(node_id)
        except Exception as e:
            logger.error(
                "Dual-delete Neo4j exception for node_id=%s: %s", node_id, e
            )

        if qdrant_success and neo4j_success:
            logger.info("Dual-delete successful for node_id=%s", node_id)
        elif qdrant_success or neo4j_success:
            logger.warning(
                "Partial dual-delete for node_id=%s — qdrant=%s, neo4j=%s",
                node_id,
                qdrant_success,
                neo4j_success,
            )
        else:
            logger.error("Dual-delete FAILED for node_id=%s", node_id)

        return qdrant_success or neo4j_success

    # ------------------------------------------------------------------ #
    #  Read operations  (single-backend dispatch)
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the **configured** read backend.

        Parameters
        ----------
        query : str
            Free-text search query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ranked results from the active read backend.
        """
        if self.read_backend == "neo4j":
            logger.debug("Read dispatch → Neo4j for query=%r", query)
            return self.neo4j_backend.search(query, top_k=top_k)
        else:
            logger.debug("Read dispatch → Qdrant for query=%r", query)
            return self.qdrant_backend.search(query, top_k=top_k)

    def create_index(self) -> bool:
        """Create vector indexes on **both** backends."""
        q_ok = self.qdrant_backend.create_index()
        n_ok = self.neo4j_backend.create_index()
        logger.info(
            "Dual index creation — qdrant=%s, neo4j=%s", q_ok, n_ok
        )
        return q_ok and n_ok


# --------------------------------------------------------------------------- #
#  Dual-Compare Read Backend
# --------------------------------------------------------------------------- #

class DualCompareBackend:
    """Read backend that queries both Qdrant and Neo4j and merges results.

    When VECTOR_BACKEND="dual", this backend is activated. It runs the same
    query against **both** backends in parallel (via threads), then merges
    the results by deduplicating on node ID and keeping the higher score.

    Parameters
    ----------
    qdrant_backend : QdrantBackendAdapter
        The Qdrant read adapter.
    neo4j_backend : Neo4jVectorStore
        The Neo4j vector store.
    """

    def __init__(
        self,
        qdrant_backend: QdrantBackendAdapter,
        neo4j_backend: Neo4jVectorStore,
    ):
        self.qdrant_backend = qdrant_backend
        self.neo4j_backend = neo4j_backend
        logger.info("DualCompareBackend initialised — reading from both backends")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search both backends and merge results.

        Queries Qdrant and Neo4j independently, then deduplicates by node ID
        keeping the entry with the higher similarity score.  Ties favour
        Qdrant (as the historically primary backend).

        Returns
        -------
        list[dict]
            Merged, deduplicated results sorted by descending score.
        """
        qdrant_results: List[Dict[str, Any]] = []
        neo4j_results: List[Dict[str, Any]] = []

        # Query both backends; failures in one do not block the other.
        try:
            qdrant_results = self.qdrant_backend.search(query, top_k=top_k)
            logger.debug(
                "DualCompare Qdrant returned %d results for query=%r",
                len(qdrant_results), query,
            )
        except Exception as e:
            logger.error("DualCompare Qdrant search failed: %s", e)

        try:
            neo4j_results = self.neo4j_backend.search(query, top_k=top_k)
            logger.debug(
                "DualCompare Neo4j returned %d results for query=%r",
                len(neo4j_results), query,
            )
        except Exception as e:
            logger.error("DualCompare Neo4j search failed: %s", e)

        # Merge: deduplicate by node ID, keep higher score.
        seen: Dict[str, Dict[str, Any]] = {}

        for item in qdrant_results:
            nid = item.get("id", "")
            if nid:
                seen[nid] = item

        for item in neo4j_results:
            nid = item.get("id", "")
            if not nid:
                continue
            score = item.get("score", 0.0)
            if nid not in seen or score > seen[nid].get("score", 0.0):
                seen[nid] = item

        merged = sorted(
            seen.values(),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        logger.info(
            "DualCompare merged %d qdrant + %d neo4j → %d unique results for query=%r",
            len(qdrant_results), len(neo4j_results), len(merged), query,
        )
        return merged

    def create_index(self) -> bool:
        """Create vector indexes on both backends."""
        q_ok = self.qdrant_backend.create_index()
        n_ok = self.neo4j_backend.create_index()
        logger.info(
            "DualCompare index creation — qdrant=%s, neo4j=%s", q_ok, n_ok
        )
        return q_ok and n_ok

    def store_embedding(
        self, node_id: str, text: str, embedding: List[float]
    ) -> bool:
        """Write to both backends (dual-write semantics)."""
        q_ok = False
        n_ok = False
        try:
            q_ok = self.qdrant_backend.store_embedding(node_id, text, embedding)
        except Exception as e:
            logger.error("DualCompare Qdrant write failed: %s", e)
        try:
            n_ok = self.neo4j_backend.store_embedding(node_id, text, embedding)
        except Exception as e:
            logger.error("DualCompare Neo4j write failed: %s", e)
        return q_ok or n_ok

    def delete_embedding(self, node_id: str) -> bool:
        """Delete from both backends."""
        q_ok = False
        n_ok = False
        try:
            q_ok = self.qdrant_backend.delete_embedding(node_id)
        except Exception as e:
            logger.error("DualCompare Qdrant delete failed: %s", e)
        try:
            n_ok = self.neo4j_backend.delete_embedding(node_id)
        except Exception as e:
            logger.error("DualCompare Neo4j delete failed: %s", e)
        return q_ok or n_ok


# --------------------------------------------------------------------------- #
#  Parity Validation
# --------------------------------------------------------------------------- #

def validate_backend_parity(
    test_queries: List[str],
    dual_store: Optional[DualWriteVectorStore] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compare search results between Qdrant and Neo4j for the same queries.

    For every query in *test_queries* this function runs an independent
    search against **both** backends and reports:

        - The top-K node IDs returned by each backend.
        - How many IDs overlap (exact set intersection).
        - Per-ID score differences when the same ID appears in both sets.

    Parameters
    ----------
    test_queries : list[str]
        Queries to execute against both backends.
    dual_store : DualWriteVectorStore, optional
        Pre-built wrapper.  A fresh instance is created when *None*.
    top_k : int
        Number of results to request from each backend.

    Returns
    -------
    dict
        Mapping from query string to a parity report dictionary::

            {
                "<query>": {
                    "qdrant_top_ids": [str, ...],
                    "neo4j_top_ids": [str, ...],
                    "overlap_count": int,
                    "score_differences": {
                        "<node_id>": {"qdrant": float, "neo4j": float, "delta": float},
                        ...
                    },
                    "timestamp": str,  // ISO-8601 UTC
                },
                ...
            }
    """
    if dual_store is None:
        dual_store = DualWriteVectorStore()

    report: Dict[str, Any] = {}

    for query in test_queries:
        qdrant_results = dual_store.qdrant_backend.search(query, top_k=top_k)
        neo4j_results = dual_store.neo4j_backend.search(query, top_k=top_k)

        qdrant_ids = [r["id"] for r in qdrant_results]
        neo4j_ids = [r["id"] for r in neo4j_results]
        overlap = set(qdrant_ids) & set(neo4j_ids)

        # Build score-difference map for overlapping IDs.
        score_differences: Dict[str, Dict[str, float]] = {}
        qdrant_score_map = {r["id"]: r["score"] for r in qdrant_results}
        neo4j_score_map = {r["id"]: r["score"] for r in neo4j_results}

        for nid in overlap:
            q_score = qdrant_score_map.get(nid, 0.0)
            n_score = neo4j_score_map.get(nid, 0.0)
            score_differences[nid] = {
                "qdrant": q_score,
                "neo4j": n_score,
                "delta": abs(q_score - n_score),
            }

        report[query] = {
            "qdrant_top_ids": qdrant_ids,
            "neo4j_top_ids": neo4j_ids,
            "overlap_count": len(overlap),
            "score_differences": score_differences,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Parity check for query=%r — qdrant=%d results, neo4j=%d results, "
            "overlap=%d / %d",
            query,
            len(qdrant_ids),
            len(neo4j_ids),
            len(overlap),
            top_k,
        )

    return report


# --------------------------------------------------------------------------- #
#  Periodic Parity-Check Scheduler
# --------------------------------------------------------------------------- #

class ParityCheckScheduler:
    """Run parity validation on a fixed interval in a background thread.

    Parameters
    ----------
    dual_store : DualWriteVectorStore
        The wrapper whose backends will be compared.
    test_queries : list[str]
        Fixed set of queries to run each cycle.
    interval_seconds : float
        Seconds between consecutive checks (default 3600 = 1 hour).
        Also readable from the ``PARITY_CHECK_INTERVAL`` environment
        variable at construction time.
    top_k : int
        Results-per-query for each parity check.

    Example
    -------
    >>> scheduler = ParityCheckScheduler(
    ...     dual_store=dual_store,
    ...     test_queries=["machine learning", "deep reinforcement learning"],
    ... )
    >>> scheduler.start()
    >>> # later …
    >>> scheduler.stop()
    """

    def __init__(
        self,
        dual_store: DualWriteVectorStore,
        test_queries: List[str],
        interval_seconds: Optional[float] = None,
        top_k: int = 10,
    ):
        self.dual_store = dual_store
        self.test_queries = test_queries
        self.interval_seconds = (
            interval_seconds
            or float(os.getenv("PARITY_CHECK_INTERVAL", "3600"))
        )
        self.top_k = top_k
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_report: Optional[Dict[str, Any]] = None

    def _run_loop(self) -> None:
        """Internal loop executed in the background thread."""
        while self._running:
            self._last_report = validate_backend_parity(
                test_queries=self.test_queries,
                dual_store=self.dual_store,
                top_k=self.top_k,
            )
            logger.info(
                "Periodic parity check completed at %s",
                datetime.now(timezone.utc).isoformat(),
            )
            # Sleep in small increments so we can respond to stop() quickly.
            import time
            for _ in range(int(self.interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    def start(self) -> None:
        """Start the background parity-check loop."""
        if self._running:
            logger.warning("ParityCheckScheduler already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            "ParityCheckScheduler started — interval=%ds, queries=%d",
            self.interval_seconds,
            len(self.test_queries),
        )

    def stop(self) -> None:
        """Stop the background parity-check loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5)
            self._thread = None
        logger.info("ParityCheckScheduler stopped")

    def get_last_report(self) -> Optional[Dict[str, Any]]:
        """Return the most recent parity report (or ``None`` if not yet run)."""
        return self._last_report


# --------------------------------------------------------------------------- #
#  Singleton / convenience factory
# --------------------------------------------------------------------------- #

_dual_write_instance: Optional[DualWriteVectorStore] = None


def get_dual_write_store() -> DualWriteVectorStore:
    """Return a singleton ``DualWriteVectorStore`` instance (lazy init)."""
    global _dual_write_instance
    if _dual_write_instance is None:
        _dual_write_instance = DualWriteVectorStore()
    return _dual_write_instance
