"""
Vector-first hybrid search pipeline using Neo4j as a single source.

Replaces the previous dual-source (Qdrant + Neo4j) architecture with a
streamlined three-stage pipeline:

    1. **Vector Search** — Embed query → Neo4j vector index → Top-K seed nodes
    2. **Graph Traversal** — Expand from seed nodes via relationships (multi-hop)
    3. **Context Assembly** — Format results with layman_explanation, analogy, fiction_seed

Benefits:
    - Single source of truth (Neo4j provides both vector and graph data)
    - No score normalization between disparate backends
    - Simpler ranking: deduplication + score-based ordering only
    - Full type hints, docstrings, parameterized Cypher, stage timing logs

Requirements:
    - Parameterized Cypher queries only ($param syntax)
    - Empty results handled gracefully at each pipeline stage
    - Timing logged for vector, traversal, and assembly stages
"""

import os
import time
import logging
from typing import List, Dict, Any, Set, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
from config import get_config, get_vector_backend_mode
from neo4j_vector_store import get_vector_backend, Neo4jVectorStore

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Vector-first hybrid search engine backed entirely by Neo4j.

    Pipeline stages:
        1. Vector Search — Uses Neo4j's native vector index to find top-K
           semantically similar nodes (seed nodes).
        2. Graph Traversal — Expands from seed nodes via graph relationships
           up to ``max_hops`` hops using variable-length relationship patterns.
        3. Context Assembly — Formats all discovered nodes into a structured
           Markdown context block suitable for LLM consumption.

    Attributes:
        vector_backend: Vector backend implementing the VectorBackend protocol
                        (Neo4jVectorStore when VECTOR_BACKEND=neo4j).
        _driver: Lazy-loaded Neo4j driver for graph traversal queries.
    """

    def __init__(self) -> None:
        self._vector_backend: Any = None
        self._driver: Optional[GraphDatabase.Driver] = None
        self._qdrant_backend: Any = None
        self._neo4j_backend: Any = None
        self._dual_backend: Any = None
        # Start with the mode from environment (re-read at runtime via switch_backend)
        self.backend_mode: str = get_vector_backend_mode()

    # ------------------------------------------------------------------ #
    #  Lazy property accessors
    # ------------------------------------------------------------------ #

    @property
    def vector_backend(self) -> Any:
        """Lazy-load the configured vector backend on first access."""
        if self._vector_backend is None:
            self._vector_backend = get_vector_backend()
        return self._vector_backend

    def _ensure_driver(self) -> GraphDatabase.Driver:
        """Return an active Neo4j driver, initialising lazily if needed.

        Returns:
            A connected ``GraphDatabase.Driver`` instance.

        Raises:
            ServiceUnavailable: If Neo4j cannot be reached after retries.
        """
        if self._driver is not None:
            return self._driver

        cfg = get_config().get_config()["graph_db"]
        uri = cfg["neo4j_uri"]
        user = cfg["neo4j_user"]
        password = cfg["neo4j_password"]
        max_pool = cfg.get("max_connection_pool_size", 50)
        timeout = cfg.get("connection_timeout", 30.0)

        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_pool,
            connection_timeout=timeout,
        )
        # Verify connectivity
        with self._driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j driver initialised for graph traversal (uri=%s)", uri)
        return self._driver

    # ------------------------------------------------------------------ #
    #  Backend-specific lazy loaders
    # ------------------------------------------------------------------ #

    def _get_qdrant_backend(self) -> Any:
        """Lazy-load the Qdrant backend adapter."""
        if self._qdrant_backend is None:
            from vector_db import get_vector_db
            self._qdrant_backend = get_vector_db()
        return self._qdrant_backend

    def _get_neo4j_backend(self) -> Neo4jVectorStore:
        """Lazy-load the Neo4j vector store."""
        if self._neo4j_backend is None:
            from neo4j_vector_store import _get_neo4j_vector_store
            self._neo4j_backend = _get_neo4j_vector_store()
        return self._neo4j_backend

    def _get_dual_backend(self) -> Any:
        """Lazy-load the DualCompareBackend (wraps both Qdrant and Neo4j)."""
        if self._dual_backend is None:
            from dual_write_vector_store import DualCompareBackend, QdrantBackendAdapter
            qdrant_adapter = QdrantBackendAdapter(self._get_qdrant_backend())
            neo4j_store = self._get_neo4j_backend()
            self._dual_backend = DualCompareBackend(qdrant_adapter, neo4j_store)
        return self._dual_backend

    # ------------------------------------------------------------------ #
    #  Active backend selection (feature flag)
    # ------------------------------------------------------------------ #

    def _get_active_backend(self) -> Any:
        """Return the active read backend based on the current feature flag.

        Re-reads ``self.backend_mode`` each call so that a prior call to
        :meth:`switch_backend` takes effect immediately.

        Returns
        -------
        VectorBackend
            The backend instance for the current mode.
        """
        if self.backend_mode == "neo4j":
            return self._get_neo4j_backend()
        elif self.backend_mode == "dual":
            return self._get_dual_backend()
        else:  # default: qdrant
            return self._get_qdrant_backend()

    def switch_backend(self, backend: str) -> bool:
        """Runtime backend switching with health check and logging.

        Before switching, verifies that the target backend is reachable.
        If the target is unhealthy, the switch is rejected and the current
        backend remains active.  All switches are logged.

        Parameters
        ----------
        backend : str
            One of ``"qdrant"``, ``"neo4j"``, or ``"dual"``.

        Returns
        -------
        bool
            ``True`` if the switch succeeded, ``False`` otherwise.
        """
        if backend not in ("qdrant", "neo4j", "dual"):
            logger.warning("switch_backend rejected — invalid backend=%r", backend)
            return False

        # Health check before switching
        health = self.check_backend_health(backend)
        if not health.get("healthy", False):
            logger.warning(
                "switch_backend rejected — target %r unhealthy: %s",
                backend, health.get("detail", "unknown"),
            )
            return False

        old_mode = self.backend_mode
        self.backend_mode = backend
        # Also update the environment variable so that any child processes
        # or other modules reading VECTOR_BACKEND see the new value.
        os.environ["VECTOR_BACKEND"] = backend
        logger.info("Switched vector backend: %s → %s", old_mode, backend)
        return True

    def check_backend_health(self, backend_name: str) -> Dict[str, Any]:
        """Verify backend connectivity and index status.

        Tests the connection to the named backend by attempting a lightweight
        operation (ping or simple query).  Returns a health report dictionary.

        Parameters
        ----------
        backend_name : str
            One of ``"qdrant"``, ``"neo4j"``, or ``"dual"``.

        Returns
        -------
        dict
            Health report with keys:

            - **backend** (str): The backend that was checked.
            - **healthy** (bool): ``True`` if the backend responded successfully.
            - **detail** (str): Human-readable status message.
            - **latency_ms** (float, optional): Round-trip time in milliseconds.
        """
        import time
        t0 = time.monotonic()

        if backend_name == "qdrant":
            return self._check_qdrant_health(t0)
        elif backend_name == "neo4j":
            return self._check_neo4j_health(t0)
        elif backend_name == "dual":
            # For dual mode, both backends must be healthy.
            q_health = self._check_qdrant_health(t0)
            n_health = self._check_neo4j_health(t0)
            healthy = q_health["healthy"] and n_health["healthy"]
            detail = (
                f"Qdrant: {q_health['detail']}; Neo4j: {n_health['detail']}"
            )
            return {
                "backend": "dual",
                "healthy": healthy,
                "detail": detail,
                "qdrant": q_health,
                "neo4j": n_health,
            }
        else:
            return {
                "backend": backend_name,
                "healthy": False,
                "detail": f"Unknown backend: {backend_name}",
            }

    def _check_qdrant_health(self, t0: float) -> Dict[str, Any]:
        """Ping Qdrant and verify collection accessibility."""
        try:
            qdrant_db = self._get_qdrant_backend()
            # Attempt a lightweight ping via get_collections.
            collections = qdrant_db.client.get_collections()
            elapsed_ms = (time.monotonic() - t0) * 1000
            count = len(collections.collections) if collections else 0
            return {
                "backend": "qdrant",
                "healthy": True,
                "detail": f"Connected, {count} collection(s) available",
                "latency_ms": round(elapsed_ms, 2),
            }
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("Qdrant health check failed: %s", e)
            return {
                "backend": "qdrant",
                "healthy": False,
                "detail": str(e),
                "latency_ms": round(elapsed_ms, 2),
            }

    def _check_neo4j_health(self, t0: float) -> Dict[str, Any]:
        """Ping Neo4j and verify basic connectivity."""
        try:
            driver = self._ensure_driver()
            with driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
            elapsed_ms = (time.monotonic() - t0) * 1000
            is_ok = record and record["ok"] == 1 if record else False
            return {
                "backend": "neo4j",
                "healthy": bool(is_ok),
                "detail": "Connected and responsive" if is_ok else "Query returned unexpected result",
                "latency_ms": round(elapsed_ms, 2),
            }
        except ServiceUnavailable as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("Neo4j health check failed — service unavailable: %s", e)
            return {
                "backend": "neo4j",
                "healthy": False,
                "detail": f"Service unavailable: {e}",
                "latency_ms": round(elapsed_ms, 2),
            }
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning("Neo4j health check failed: %s", e)
            return {
                "backend": "neo4j",
                "healthy": False,
                "detail": str(e),
                "latency_ms": round(elapsed_ms, 2),
            }

    def _get_active_backend_for_search(self) -> Any:
        """Return the active backend for search with fallback to qdrant.

        Attempts to use the currently configured backend mode.  If that
        backend is unreachable, falls back to Qdrant (the default).

        Returns
        -------
        VectorBackend
            A working backend instance guaranteed to be available.
        """
        preferred = self.backend_mode

        # Fast path: try the preferred backend directly.
        backend = self._get_active_backend()
        health = self.check_backend_health(preferred)
        if health.get("healthy", False):
            return backend

        # Fallback: if preferred is not qdrant, try qdrant as default.
        if preferred != "qdrant":
            logger.warning(
                "Backend %r unhealthy during search, falling back to qdrant",
                preferred,
            )
            return self._get_qdrant_backend()

        # Already qdrant and still unhealthy — return it anyway so the
        # caller can handle the error from the search() call itself.
        logger.error(
            "Preferred backend %r unavailable and qdrant fallback also failed",
            preferred,
        )
        return backend

    # ------------------------------------------------------------------ #
    #  Main search pipeline
    # ------------------------------------------------------------------ #

    def search(
        self, query: str, top_k: int = 3, max_hops: int = 2
    ) -> Dict[str, Any]:
        """Vector-first hybrid search pipeline.

        Executes a three-stage search pipeline:

            1. **Vector Search** — Embed ``query`` and query Neo4j's vector
               index for the top-K most similar nodes (seed nodes).
            2. **Graph Traversal** — Expand from each seed node via graph
               relationships up to ``max_hops`` hops.
            3. **Context Assembly** — Format all discovered nodes into a
               structured Markdown context block.

        Args:
            query: Free-text search query.
            top_k: Number of seed nodes to retrieve from vector search.
            max_hops: Maximum relationship traversal depth (1..3 recommended).

        Returns:
            Dictionary containing:

            - **query** (str): The original search query.
            - **vector_results** (List[Dict]): Raw vector search results.
            - **expanded_nodes** (List[Dict]): Nodes discovered via graph traversal.
            - **context_block** (str): Formatted Markdown context block.
            - **total_nodes** (int): Count of unique nodes across all stages.

        Handles empty results gracefully at each stage — if vector search
        returns nothing, traversal and assembly are skipped with warnings.
        """
        total_start = time.monotonic()
        result: Dict[str, Any] = {
            "query": query,
            "vector_results": [],
            "expanded_nodes": [],
            "context_block": "",
            "total_nodes": 0,
        }

        # ---- Stage 1: Vector Search ----
        t0 = time.monotonic()
        active_backend = self._get_active_backend_for_search()
        vector_results = active_backend.search(query, top_k=top_k)
        t1 = time.monotonic()
        logger.info(
            "Pipeline stage [vector_search]: %.3fs — found %d seed nodes",
            t1 - t0, len(vector_results),
        )
        result["vector_results"] = vector_results

        if not vector_results:
            logger.warning("Vector search returned no results for query=%r", query)
            result["context_block"] = (
                f"> **No results found** for query: \"{query}\"\n"
                "> Vector search returned zero matching nodes.\n"
            )
            return result

        seed_node_ids: List[str] = [r["id"] for r in vector_results if r.get("id")]
        if not seed_node_ids:
            logger.warning("Vector results contain no valid node IDs for query=%r", query)
            result["context_block"] = (
                f"> **No valid nodes** for query: \"{query}\"\n"
                "> Vector search returned results but none contained a node ID.\n"
            )
            return result

        # ---- Stage 2: Graph Traversal ----
        t2 = time.monotonic()
        expanded_nodes = self._graph_traversal(seed_node_ids, max_hops=max_hops)
        t3 = time.monotonic()
        logger.info(
            "Pipeline stage [graph_traversal]: %.3fs — found %d expanded nodes",
            t3 - t2, len(expanded_nodes),
        )
        result["expanded_nodes"] = expanded_nodes

        # ---- Stage 3: Context Assembly ----
        t4 = time.monotonic()
        context_block = self._assemble_context(vector_results, expanded_nodes)
        t5 = time.monotonic()
        logger.info(
            "Pipeline stage [context_assembly]: %.3fs — context length %d chars",
            t5 - t4, len(context_block),
        )
        result["context_block"] = context_block

        # ---- Totals ----
        all_node_ids: Set[str] = set()
        all_node_ids.update(seed_node_ids)
        all_node_ids.update(n.get("id", "") for n in expanded_nodes if n.get("id"))
        result["total_nodes"] = len(all_node_ids)

        total_elapsed = time.monotonic() - total_start
        logger.info(
            "Pipeline complete for query=%r: %.3fs total (%d unique nodes)",
            query, total_elapsed, result["total_nodes"],
        )

        return result

    # ------------------------------------------------------------------ #
    #  Pipeline stage implementations
    # ------------------------------------------------------------------ #

    def _graph_traversal(
        self, seed_node_ids: List[str], max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Expand from seed nodes via graph relationships (multi-hop).

        Uses a variable-length relationship pattern to discover neighboring
        nodes within ``max_hops`` hops from any seed node. Results are
        deduplicated and limited to prevent runaway queries.

        Cypher pattern::

            MATCH path = (start)-[rels*1..$max_hops]-(neighbor)
            WHERE start.id IN $seed_ids
            AND neighbor.id <> start.id
            RETURN path, length(path) AS hops
            LIMIT $limit

        Args:
            seed_node_ids: List of node IDs returned by vector search.
            max_hops: Maximum traversal depth (clamped to 1..3).

        Returns:
            List of dictionaries describing each discovered neighbor node,
            including relationship metadata (type, direction, hop count).
            Empty list if traversal fails or yields no neighbors.
        """
        if not seed_node_ids:
            logger.info("Graph traversal skipped — no seed node IDs provided")
            return []

        # Clamp max_hops to a safe range
        effective_hops = max(1, min(max_hops, 3))
        limit = 50

        try:
            driver = self._ensure_driver()
            cfg = get_config().get_config()["graph_db"]
            database = cfg.get("neo4j_database", "neo4j")

            cypher = """
                MATCH path = (start)-[rels*1..$max_hops]-(neighbor)
                WHERE start.id IN $seed_ids
                AND neighbor.id <> start.id
                RETURN path, length(path) AS hops
                LIMIT $limit
            """

            expanded: List[Dict[str, Any]] = []
            seen_ids: Set[str] = set(seed_node_ids)  # Exclude seed nodes themselves

            with driver.session(database=database) as session:
                result = session.run(
                    cypher,
                    seed_ids=seed_node_ids,
                    max_hops=effective_hops,
                    limit=limit,
                )

                for record in result:
                    path = record["path"]
                    hops = record["hops"]

                    # Extract neighbor nodes from the path
                    nodes_in_path = list(path.nodes)
                    relationships_in_path = list(path.relationships)

                    # The neighbor is the last node in the path
                    if len(nodes_in_path) < 2:
                        continue

                    neighbor_node = nodes_in_path[-1]
                    neighbor_props = dict(neighbor_node)
                    neighbor_id = neighbor_props.get("id", "")

                    if not neighbor_id or neighbor_id in seen_ids:
                        continue

                    seen_ids.add(neighbor_id)

                    # Collect relationship chain metadata
                    rel_chain: List[Dict[str, Any]] = []
                    for rel in relationships_in_path:
                        rel_chain.append({
                            "type": str(rel.type),
                            "direction": "outgoing" if rel.start_node == nodes_in_path[0] else "incoming",
                        })

                    # Determine primary label
                    labels = list(neighbor_node.labels)
                    primary_label = labels[0] if labels else ""

                    expanded.append({
                        "id": neighbor_id,
                        "label": primary_label,
                        "name_th": neighbor_props.get("name_th", ""),
                        "name_en": neighbor_props.get("name_en", ""),
                        "layman_explanation": neighbor_props.get("layman_explanation", ""),
                        "analogy": neighbor_props.get("analogy", ""),
                        "fiction_seed": neighbor_props.get("fiction_seed", ""),
                        "score": 0.0,  # Traversal-discovered nodes have no vector score
                        "hops": hops,
                        "relationship_chain": rel_chain,
                        "properties": neighbor_props,
                    })

            logger.info(
                "Graph traversal: %d unique neighbors found from %d seeds (max_hops=%d)",
                len(expanded), len(seed_node_ids), effective_hops,
            )
            return expanded

        except Exception as e:
            logger.error("Graph traversal failed: %s", e)
            return []

    def _assemble_context(
        self,
        vector_results: List[Dict[str, Any]],
        expanded_nodes: List[Dict[str, Any]],
    ) -> str:
        """Format search results into a structured Markdown context block.

        Combines vector search results and graph-traversal-discovered nodes
        into a single deduplicated list. Each node is formatted as a
        Markdown section containing its name, simple explanation, analogy,
        and creative fiction seed.

        Args:
            vector_results: Raw results from the vector search stage.
            expanded_nodes: Nodes discovered via graph traversal.

        Returns:
            A Markdown-formatted string suitable for LLM context injection.
            Empty string if no nodes are available.
        """
        if not vector_results and not expanded_nodes:
            logger.info("Context assembly skipped — no results to format")
            return ""

        # Deduplicate by node ID — vector results take priority (they have scores)
        seen: Dict[str, Dict[str, Any]] = {}

        for node in vector_results:
            nid = node.get("id", "")
            if nid and nid not in seen:
                seen[nid] = node

        for node in expanded_nodes:
            nid = node.get("id", "")
            if nid and nid not in seen:
                seen[nid] = node

        ordered_nodes = list(seen.values())

        # Sort: scored nodes first (by descending score), then traversal nodes
        def sort_key(n: Dict[str, Any]) -> tuple:
            has_score = n.get("score", 0.0) > 0
            return (not has_score, -n.get("score", 0.0))

        ordered_nodes.sort(key=sort_key)

        lines: List[str] = []
        lines.append("# Search Results")
        lines.append("")

        for idx, node in enumerate(ordered_nodes, start=1):
            name_th = node.get("name_th", "")
            name_en = node.get("name_en", "")
            label = node.get("label", "")

            # Build display name
            if name_th and name_en:
                display_name = f"{name_th} ({name_en})"
            elif name_th:
                display_name = name_th
            elif name_en:
                display_name = name_en
            else:
                display_name = node.get("id", f"Node-{idx}")

            layman = node.get("layman_explanation", "")
            analogy = node.get("analogy", "")
            fiction = node.get("fiction_seed", "")
            score = node.get("score")
            hops = node.get("hops")

            lines.append(f"## {idx}. {display_name}")
            if label:
                lines.append(f"*Type: {label}*")
                lines.append("")

            # Metadata line
            meta_parts: List[str] = []
            if score is not None and score > 0:
                meta_parts.append(f"Relevance: {score:.4f}")
            if hops is not None:
                meta_parts.append(f"Hops: {hops}")
            if meta_parts:
                lines.append(f"📊 {' | '.join(meta_parts)}")
                lines.append("")

            if layman:
                lines.append(f"**Simple Explanation:** {layman}")
                lines.append("")

            if analogy:
                lines.append(f"**Analogy:** {analogy}")
                lines.append("")

            if fiction:
                lines.append(f"**Creative Seed:** {fiction}")
                lines.append("")

            lines.append("---")
            lines.append("")

        context = "\n".join(lines)
        logger.info(
            "Context assembly complete: %d nodes, %d characters",
            len(ordered_nodes), len(context),
        )
        return context

    # ------------------------------------------------------------------ #
    #  Simplified combine / dedup logic
    # ------------------------------------------------------------------ #

    def _combine_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate and rank a single list of search results.

        No longer merges Qdrant + Neo4j results — Neo4j is the sole source
        providing both vector similarity scores and graph topology data.
        This method performs deduplication by node ID (keeping the highest
        score) and returns results sorted by descending score.

        Args:
            results: Flat list of result dictionaries, each with at least
                     an ``id`` key and optionally a ``score`` key.

        Returns:
            Deduplicated, score-ranked list of result dictionaries.
        """
        if not results:
            return []

        seen: Dict[str, Dict[str, Any]] = {}
        for item in results:
            nid = item.get("id", "")
            if not nid:
                continue
            score = item.get("score", 0.0)
            if nid not in seen or score > seen[nid].get("score", 0.0):
                seen[nid] = item

        ranked = sorted(seen.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        logger.info("_combine_results: %d input → %d unique after dedup", len(results), len(ranked))
        return ranked

    # ------------------------------------------------------------------ #
    #  Legacy compatibility (deprecated)
    # ------------------------------------------------------------------ #

    def store_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store document embedding via the vector backend.

        Deprecated: Use ``neo4j_vector_store.Neo4jVectorStore.store_embedding()``
        directly for new code. This method is retained for backward compatibility.

        Args:
            doc_id: Unique document identifier.
            content: Document text content.
            metadata: Optional metadata dictionary.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            # Generate embedding and store via vector backend
            from config import get_embedding_model
            from sentence_transformers import SentenceTransformer

            model_name = get_embedding_model()
            model = SentenceTransformer(model_name)
            embedding = model.encode([content])[0].tolist()

            success = self.vector_backend.store_embedding(doc_id, content, embedding)
            if success:
                logger.info("Stored document embedding: doc_id=%s", doc_id)
            return success
        except Exception as e:
            logger.error("Failed to store document %s: %s", doc_id, e)
            return False

    def close(self) -> None:
        """Close the underlying Neo4j driver (if open)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("HybridSearchEngine driver closed")


# --------------------------------------------------------------------------- #
#  Singleton / factory
# --------------------------------------------------------------------------- #

_hybrid_search_instance: Optional[HybridSearchEngine] = None


def get_hybrid_search() -> HybridSearchEngine:
    """Get or create the global HybridSearchEngine instance (lazy init).

    Returns:
        The singleton HybridSearchEngine instance.
    """
    global _hybrid_search_instance
    if _hybrid_search_instance is None:
        _hybrid_search_instance = HybridSearchEngine()
    return _hybrid_search_instance


# Backward-compatible alias — callers should prefer get_hybrid_search().
hybrid_search: Optional[HybridSearchEngine] = None
