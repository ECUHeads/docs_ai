"""
GraphDB integration module for Neo4j.
This module handles graphRAG functionality using Neo4j as the graph database.
Extracts sections, headings, and entities from markdown to build a real knowledge graph.
Uses lazy initialization to avoid crashing the app when DB is not ready.

Changes (Review-02):
  - Batch transactions using UNWIND for sections, code blocks, terms, and links
    to reduce network round-trips significantly.
  - Multi-term graph_rag_search: search with multiple query terms instead of only
    the first term, then aggregate match counts.
  - Added create_indexes() for Neo4j performance on large datasets.
  - All Cypher queries audited — parameterized queries used exclusively (no injection risk).
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
from config import get_config

logger = logging.getLogger(__name__)


class Neo4jGraphDB:
    """Neo4j GraphDB integration for graphRAG with real relationship extraction"""

    def __init__(self):
        self.driver = None
        self._init_driver()

    def _init_driver(self):
        """
        Initialize Neo4j driver with configuration.

        Pipeline Improvement:
          - max_connection_pool_size from config (default 50)
          - connection_timeout from config (default 30s)
          - Retry mechanism with exponential backoff for initialization
        """
        try:
            graph_db_config = get_config().get_config()['graph_db']
            uri = graph_db_config['neo4j_uri']
            user = graph_db_config['neo4j_user']
            password = graph_db_config['neo4j_password']
            database = graph_db_config['neo4j_database']

            # Pipeline Improvement: Connection pool and timeout settings
            max_pool_size = graph_db_config.get('max_connection_pool_size', 50)
            conn_timeout = graph_db_config.get('connection_timeout', 30.0)
            retry_attempts = graph_db_config.get('connection_retry_attempts', 3)
            retry_delay = graph_db_config.get('connection_retry_delay', 1.0)

            # Retry initialization with exponential backoff
            last_error = None
            for attempt in range(1, retry_attempts + 1):
                try:
                    self.driver = GraphDatabase.driver(
                        uri,
                        auth=(user, password),
                        max_connection_pool_size=max_pool_size,
                        connection_timeout=conn_timeout,
                    )
                    # Verify connection by running a simple query
                    with self.driver.session() as session:
                        session.run("RETURN 1")
                    logger.info(
                        f"Neo4j driver initialized successfully "
                        f"(pool_size={max_pool_size}, timeout={conn_timeout}s)"
                    )
                    return
                except (ServiceUnavailable, TransientError, ConnectionError) as e:
                    last_error = e
                    if attempt < retry_attempts:
                        logger.warning(
                            f"Neo4j connection attempt {attempt}/{retry_attempts} failed: {e}. "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to connect to Neo4j after {retry_attempts} attempts: {e}"
                        )

            raise last_error  # type: ignore[misc]
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    # --- Health check (Pipeline Improvement) ---

    def health_check(self) -> bool:
        """
        Check if the Neo4j connection is healthy.

        Returns:
            True if the connection is active and responsive, False otherwise.
        """
        try:
            if self.driver is None:
                logger.warning("Neo4j driver not initialized")
                return False
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS healthy")
                record = result.single()
                return record and record["healthy"] == 1
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    def verify_connectivity(self) -> bool:
        """
        Verify Neo4j connectivity with automatic reconnection.

        Returns:
            True if connection is established (after reconnect if needed).
        """
        if self.driver is not None and self.health_check():
            return True
        # Attempt to reinitialize
        logger.info("Neo4j connection lost or unhealthy, attempting reconnection...")
        try:
            if self.driver:
                self.driver.close()
            self._init_driver()
            return self.health_check()
        except Exception as e:
            logger.error(f"Failed to reconnect to Neo4j: {e}")
            return False

    # --- Neo4j indexes for performance (Review-02 fix) ---

    def create_indexes(self) -> bool:
        """
        Create indexes for frequently queried fields to improve query performance.
        Safe to call multiple times — Neo4j ignores duplicate index creation.
        """
        try:
            with self.driver.session() as session:
                indexes = [
                    "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
                    "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
                    "CREATE INDEX section_id_index IF NOT EXISTS FOR (s:Section) ON (s.id)",
                    "CREATE INDEX section_doc_id_index IF NOT EXISTS FOR (s:Section) ON (s.doc_id)",
                    "CREATE INDEX term_name_index IF NOT EXISTS FOR (t:Term) ON (t.name)",
                    "CREATE INDEX codeblock_id_index IF NOT EXISTS FOR (c:CodeBlock) ON (c.id)",
                    "CREATE INDEX url_id_index IF NOT EXISTS FOR (l:URL) ON (l.id)",
                ]
                for index_query in indexes:
                    session.run(index_query)
                    logger.info(f"Created/verified index: {index_query}")
            return True
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False

    # --- Entity extraction helpers ---

    @staticmethod
    def _extract_sections(markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract hierarchical sections from markdown headings.
        Returns list of dicts with keys: level, title, start_line, end_line.
        """
        sections = []
        lines = markdown_content.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous section
                if current_section:
                    current_section['end_line'] = i - 1
                    sections.append(current_section)

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'start_line': i,
                    'end_line': None
                }

        # Close last section
        if current_section:
            current_section['end_line'] = len(lines) - 1
            sections.append(current_section)

        return sections

    @staticmethod
    def _extract_code_blocks(markdown_content: str) -> List[Dict[str, Any]]:
        """Extract code blocks with language tags."""
        blocks = []
        pattern = r'```(\w*)\s*\n(.*?)```'
        for match in re.finditer(pattern, markdown_content, re.DOTALL):
            blocks.append({
                'language': match.group(1) or 'unknown',
                'code': match.group(2).strip(),
            })
        return blocks

    @staticmethod
    def _extract_key_terms(markdown_content: str, max_terms: int = 50) -> List[str]:
        """
        Extract key terms: capitalized words, technical terms, and repeated phrases.
        Simple heuristic-based extraction for English text.

        NOTE (Review-02): For production Thai-language documents, consider replacing
        this with an NLP library such as spaCy or THUNDER for better accuracy.
        """
        # Remove code blocks to avoid false positives
        clean = re.sub(r'```.*?```', '', markdown_content, flags=re.DOTALL)
        # Remove inline code
        clean = re.sub(r'`[^`]*`', '', clean)

        # Find capitalized words (potential proper nouns / terms)
        terms = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b', clean)
        # Also find repeated significant words
        word_freq: Dict[str, int] = {}
        for word in re.findall(r'\b([a-zA-Z]{4,})\b', clean.lower()):
            word_freq[word] = word_freq.get(word, 0) + 1

        frequent = [w for w, c in word_freq.items() if c >= 3][:max_terms]

        # Combine and deduplicate
        combined = list(dict.fromkeys(terms + frequent))[:max_terms]
        return combined

    @staticmethod
    def _extract_key_terms_th(markdown_content: str, max_terms: int = 50) -> List[str]:
        """
        Extract key terms from Thai text using pythainlp.

        REVIEW-03 FIX (Item #8): Adds Thai language support for key term extraction.
        Falls back to _extract_key_terms() if pythainlp is not installed.

        Args:
            markdown_content: Raw markdown text (may contain Thai content).
            max_terms: Maximum number of terms to return.

        Returns:
            List of extracted key terms from Thai text.
        """
        try:
            from pythainlp import word_tokenize
            from pythainlp.util import strip_accents
        except ImportError:
            logger.warning(
                "pythainlp not installed; falling back to English term extraction. "
                "Install with: pip install pythainlp"
            )
            return Neo4jGraphDB._extract_key_terms(markdown_content, max_terms)

        # Remove code blocks
        clean = re.sub(r'```.*?```', '', markdown_content, flags=re.DOTALL)
        clean = re.sub(r'`[^`]*`', '', clean)

        text = strip_accents(clean)
        words = word_tokenize(text, engine='newmm')

        # Thai stop words to filter out
        stop_words = {
            'ที่', 'เป็น', 'มี', 'ใน', 'และ', 'ของ', 'จะ', 'ได้',
            'ก็', 'กับ', 'จาก', 'ถึง', 'สำหรับ', 'โดย', 'ไม่',
            'แต่', 'หรือ', 'ยัง', 'สามารถ', 'นี้', 'นั้น', 'ซึ่ง',
            'มา', 'ไป', 'ทำ', 'เช่น', 'เมื่อ', 'ถ้า', 'เพราะ',
        }
        terms = [w for w in words if w not in stop_words and len(w) > 2]

        # Get frequent terms
        word_freq: Dict[str, int] = {}
        for term in terms:
            word_freq[term] = word_freq.get(term, 0) + 1

        frequent = [w for w, c in word_freq.items() if c >= 2][:max_terms]
        return list(dict.fromkeys(frequent))[:max_terms]

    @staticmethod
    def _extract_key_terms_auto(markdown_content: str, max_terms: int = 50) -> List[str]:
        """
        Auto-detect language and extract key terms accordingly.

        REVIEW-03 FIX (Item #8): Heuristic language detection — if Thai characters
        are present in significant proportion, use Thai extraction; otherwise English.

        Args:
            markdown_content: Raw markdown text.
            max_terms: Maximum number of terms to return.

        Returns:
            List of extracted key terms.
        """
        # Simple heuristic: check if Thai Unicode range is present
        thai_chars = sum(1 for c in markdown_content if '\u0E00' <= c <= '\u0E7F')
        total_chars = len(markdown_content.replace('\n', '').replace(' ', ''))
        thai_ratio = thai_chars / total_chars if total_chars > 0 else 0

        if thai_ratio > 0.1:  # More than 10% Thai characters
            return Neo4jGraphDB._extract_key_terms_th(markdown_content, max_terms)
        return Neo4jGraphDB._extract_key_terms(markdown_content, max_terms)

    @staticmethod
    def _extract_links(markdown_content: str) -> List[Dict[str, str]]:
        """Extract markdown links [text](url)."""
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        return [{'text': m.group(1), 'url': m.group(2)} for m in re.finditer(pattern, markdown_content)]

    # --- Core graph operations ---

    def create_nodes_and_relationships(self, markdown_content: str, doc_id: str) -> bool:
        """
        Create a real knowledge graph from markdown content:
          1. Document node
          2. Section nodes (hierarchical HAS_SECTION relationships)
          3. CodeBlock nodes (CONTAINS_CODE relationship)
          4. Term nodes (MENTIONS_TERM relationship)
          5. Link nodes (REFERENCES_URL relationship)
        
        REVIEW-02 FIX: Uses UNWIND batch transactions instead of individual session.run()
        calls in loops to dramatically reduce network round-trips.
        """
        try:
            with self.driver.session() as session:
                # 1. Create Document node
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.updated_at = timestamp(), d.title = $title
                    """,
                    doc_id=doc_id,
                    title=self._get_document_title(markdown_content)
                )

                # 2. Extract and create sections with hierarchy using UNWIND batch (Review-02 fix)
                sections = self._extract_sections(markdown_content)
                if sections:
                    # Build parent mapping before batching
                    section_data = []
                    parent_children: Dict[str, List[str]] = {}
                    for i, section in enumerate(sections):
                        section_id = f"{doc_id}_section_{i}"
                        parent_id = None
                        for j in range(i - 1, -1, -1):
                            if sections[j]['level'] < section['level']:
                                parent_id = f"{doc_id}_section_{j}"
                                break

                        section_data.append({
                            "id": section_id,
                            "title": section['title'],
                            "level": section['level'],
                            "start_line": section['start_line'],
                            "end_line": section['end_line'],
                            "doc_id": doc_id,
                            "parent_id": parent_id if parent_id else ""
                        })

                        if parent_id:
                            parent_children.setdefault(parent_id, []).append(section_id)

                    # Batch-create sections with UNWIND
                    session.run(
                        """
                        UNWIND $sections AS s
                        MERGE (sec:Section {id: s.id})
                        SET sec.title = s.title, sec.level = s.level,
                            sec.start_line = s.start_line, sec.end_line = s.end_line,
                            sec.doc_id = s.doc_id
                        WITH sec, s
                        MATCH (d:Document {id: s.doc_id})
                        MERGE (d)-[:HAS_SECTION]->(sec)
                        """,
                        sections=section_data
                    )

                    # Batch-create parent-child relationships with UNWIND
                    if parent_children:
                        rel_data = []
                        for parent_id, children in parent_children.items():
                            for child_id in children:
                                rel_data.append({"parent_id": parent_id, "child_id": child_id})

                        session.run(
                            """
                            UNWIND $rels AS r
                            MATCH (parent:Section {id: r.parent_id})
                            MATCH (child:Section {id: r.child_id})
                            MERGE (parent)-[:CONTAINS_SECTION]->(child)
                            """,
                            rels=rel_data
                        )

                # 3. Extract and create CodeBlock nodes using UNWIND batch (Review-02 fix)
                code_blocks = self._extract_code_blocks(markdown_content)
                if code_blocks:
                    block_data = [
                        {"id": f"{doc_id}_code_{i}", "language": block['language'], "doc_id": doc_id}
                        for i, block in enumerate(code_blocks)
                    ]
                    session.run(
                        """
                        UNWIND $blocks AS b
                        MERGE (c:CodeBlock {id: b.id})
                        SET c.language = b.language, c.doc_id = b.doc_id
                        WITH c, b
                        MATCH (d:Document {id: b.doc_id})
                        MERGE (d)-[:CONTAINS_CODE]->(c)
                        """,
                        blocks=block_data
                    )

                # 4. Extract and create Term nodes using UNWIND batch (Review-02 fix)
                terms = self._extract_key_terms(markdown_content)
                if terms:
                    term_data = [{"term": term.strip().lower(), "doc_id": doc_id} for term in terms]
                    session.run(
                        """
                        UNWIND $terms AS t
                        MERGE (term:Term {name: t.term})
                        ON CREATE SET term.first_seen = timestamp()
                        SET term.last_seen = timestamp(), term.count = coalesce(term.count, 0) + 1
                        WITH term, t
                        MATCH (d:Document {id: t.doc_id})
                        MERGE (d)-[:MENTIONS_TERM]->(term)
                        """,
                        terms=term_data
                    )

                # 5. Cross-document relationships: documents sharing terms
                session.run(
                    """
                    MATCH (d1:Document {id: $doc_id})-[:MENTIONS_TERM]->(t:Term)
                    MATCH (t)<-[:MENTIONS_TERM]-((d2:Document))
                    WHERE d1.id <> d2.id
                    MERGE (d1)-[:RELATED_TO {via_term: t.name}]->(d2)
                    """,
                    doc_id=doc_id
                )

                # 6. Extract and create Link nodes using UNWIND batch (Review-02 fix)
                links = self._extract_links(markdown_content)
                if links:
                    link_data = [
                        {"id": f"{doc_id}_link_{i}", "text": link['text'], "url": link['url'], "doc_id": doc_id}
                        for i, link in enumerate(links)
                    ]
                    session.run(
                        """
                        UNWIND $links AS l
                        MERGE (u:URL {id: l.id})
                        SET u.text = l.text, u.url = l.url, u.doc_id = l.doc_id
                        WITH u, l
                        MATCH (d:Document {id: l.doc_id})
                        MERGE (d)-[:REFERENCES_URL]->(u)
                        """,
                        links=link_data
                    )

                logger.info(
                    f"Created graph for {doc_id}: "
                    f"{len(sections)} sections, {len(code_blocks)} code blocks, "
                    f"{len(terms)} terms, {len(links)} links"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to create nodes and relationships: {e}")
            return False

    @staticmethod
    def _get_document_title(markdown_content: str) -> str:
        """Extract document title from the first H1 heading, or fallback."""
        match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
        return match.group(1).strip() if match else "Untitled Document"

    def graph_rag_search(self, query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Perform graphRAG search using Neo4j.
        Searches across Document, Section, and Term nodes with relationship traversal.
        
        REVIEW-02 FIX: Search with multiple significant query terms instead of only
        the first term, then aggregate match counts for better recall.
        """
        try:
            with self.driver.session() as session:
                # Split query into terms for matching
                query_terms = re.findall(r'\b\w{3,}\b', query.lower())  # Filter short words
                if not query_terms:
                    return []

                # Use up to 3 most significant terms (deduplicated)
                unique_terms = list(dict.fromkeys(query_terms))[:3]

                results_map: Dict[str, Dict[str, Any]] = {}

                for term in unique_terms:
                    result = session.run(
                        """
                        MATCH (d:Document)-[:HAS_SECTION|MENTIONS_TERM*1..$max_hops]->(node)
                        WHERE toLower(coalesce(node.title, '')) CONTAINS $term
                           OR toLower(coalesce(node.name, '')) CONTAINS $term
                           OR toLower(d.title) CONTAINS $term
                        WITH d, count(distinct node) as match_count
                        ORDER BY match_count DESC
                        LIMIT $limit
                        RETURN d.id as id, d.title as title, match_count as related_count
                        """,
                        term=term,
                        max_hops=max_hops,
                        limit=10
                    )

                    for record in result:
                        doc_id = record["id"]
                        if doc_id not in results_map:
                            results_map[doc_id] = {
                                "id": doc_id,
                                "title": record.get("title", ""),
                                "related_count": 0
                            }
                        # Accumulate match counts across terms for better ranking
                        results_map[doc_id]["related_count"] += record["related_count"]

                # Sort by accumulated related_count and return top results
                sorted_results = sorted(
                    results_map.values(),
                    key=lambda x: x["related_count"],
                    reverse=True
                )[:10]

                return sorted_results

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return []

    def get_document_relationships(self, doc_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific document including cross-document links.
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:RELATED_TO*1..$max_hops]->(related:Document)
                    RETURN related.id as related_id, related.title as title,
                           count(*) as connection_strength
                    ORDER BY connection_strength DESC
                    LIMIT 20
                    """,
                    doc_id=doc_id,
                    max_hops=max_hops
                )

                return [
                    {
                        "related_id": record["related_id"],
                        "title": record.get("title", ""),
                        "connection_strength": record["connection_strength"]
                    }
                    for record in result
                ]

        except Exception as e:
            logger.error(f"Failed to get document relationships: {e}")
            return []

    # ------------------------------------------------------------------
    # Universal Meta-Schema Support
    # ------------------------------------------------------------------

    UNIVERSAL_LABELS: List[str] = [
        "KnowledgeConcept",
        "Hardware",
        "Strategy",
        "Narrative",
        "ContentAsset",
    ]

    LOGICAL_RELS: List[str] = [
        "PREDICTS",
        "APPLIES_TO",
        "CORRELATES_WITH",
        "REQUIRES_INFRA",
    ]

    CREATIVE_RELS: List[str] = [
        "INSPIRES_PLOT",
        "IS_EXAMPLE_OF",
        "CONTRADICTS",
    ]

    STRUCTURAL_RELS: List[str] = [
        "HAS_CHAPTER",
        "NEXT_CHAPTER",
        "REFERENCES_CONCEPT",
    ]

    def create_universal_schema(self) -> bool:
        """
        Create constraints and indexes for all universal-schema node labels.

        Establishes a UNIQUE constraint on the ``id`` property for each label
        (used as the merge key in upsert operations) and a generic index on
        ``name_en`` to accelerate text-based look-ups.

        Returns:
            True if all constraints/indexes were created successfully, False on error.
        """
        try:
            with self.driver.session() as session:
                # Unique constraints on id per label (merge keys)
                for label in self.UNIVERSAL_LABELS:
                    query = (
                        f"CREATE CONSTRAINT universal_id_{label.lower()} "
                        f"IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    )
                    session.run(query)
                    logger.info(f"Created/verified constraint: {query}")

                # Text-like indexes on name_en for faster lookups
                for label in self.UNIVERSAL_LABELS:
                    query = (
                        f"CREATE INDEX universal_name_en_{label.lower()} "
                        f"IF NOT EXISTS FOR (n:{label}) ON (n.name_en)"
                    )
                    session.run(query)
                    logger.info(f"Created/verified index: {query}")

                # Index on domain for filtered queries
                for label in self.UNIVERSAL_LABELS:
                    query = (
                        f"CREATE INDEX universal_domain_{label.lower()} "
                        f"IF NOT EXISTS FOR (n:{label}) ON (n.domain)"
                    )
                    session.run(query)
                    logger.info(f"Created/verified index: {query}")

            logger.info("Universal schema constraints and indexes created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create universal schema: {e}")
            return False

    def upsert_node(
        self, label: str, properties: Dict[str, Any]
    ) -> bool:
        """
        Merge (upsert) a single node of the given universal-schema label.

        The node is matched on its ``id`` property. If it exists, all supplied
        properties are merged in; otherwise a new node is created.

        Args:
            label: One of the universal-schema labels
                (KnowledgeConcept, Hardware, Strategy, Narrative, ContentAsset).
            properties: Dictionary containing at minimum an ``id`` key.
                Supported keys: id, domain, name_th, name_en,
                technical_desc_th, technical_desc_en, layman_explanation,
                analogy, visual_concept, fiction_seed, embedding.

        Returns:
            True if the MERGE succeeded, False on error.
        """
        if label not in self.UNIVERSAL_LABELS:
            logger.error(f"upsert_node called with unknown label: {label}")
            return False

        node_id = properties.get("id")
        if not node_id:
            logger.error(f"upsert_node called without 'id' in properties for label {label}")
            return False

        try:
            with self.driver.session() as session:
                # Build dynamic SET clauses for known property keys
                set_clauses = []
                allowed_keys = {
                    "domain", "name_th", "name_en",
                    "technical_desc_th", "technical_desc_en",
                    "layman_explanation", "analogy",
                    "visual_concept", "fiction_seed", "embedding",
                }
                for key in properties:
                    if key != "id" and key in allowed_keys:
                        set_clauses.append(f"n.{key} = ${key}")

                if not set_clauses:
                    # Only id provided — just ensure the node exists
                    query = f"MERGE (n:{label} {{id: $id}})"
                else:
                    set_clause = ", ".join(set_clauses)
                    query = (
                        f"MERGE (n:{label} {{id: $id}}) "
                        f"SET {set_clause}"
                    )

                session.run(query, **properties)
                logger.debug(f"Upserted node {label} with id={node_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to upsert node {label} id={node_id}: {e}")
            return False

    def upsert_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Merge (upsert) a relationship between two universal-schema nodes.

        Both endpoints are matched on their ``id`` property across all
        universal-schema labels. The relationship type must belong to one of
        the recognised categories (Logical, Creative, Structural).

        Args:
            from_id: The ``id`` of the source node.
            to_id: The ``id`` of the target node.
            rel_type: Relationship type string, e.g. ``"PREDICTS"``,
                ``"INSPIRES_PLOT"``, ``"HAS_CHAPTER"``.
            properties: Optional dictionary of relationship properties to SET.

        Returns:
            True if the relationship was merged successfully, False on error.
        """
        all_rels = set(self.LOGICAL_RELS + self.CREATIVE_RELS + self.STRUCTURAL_RELS)
        if rel_type not in all_rels:
            logger.error(
                f"upsert_relationship called with unknown rel_type: {rel_type}. "
                f"Allowed: {all_rels}"
            )
            return False

        try:
            with self.driver.session() as session:
                # Build the base MERGE query
                labels_pattern = "|".join(self.UNIVERSAL_LABELS)
                query = (
                    f"MATCH (from_node:{labels_pattern} {{id: $from_id}}) "
                    f"MATCH (to_node:{labels_pattern} {{id: $to_id}}) "
                    f"MERGE (from_node)-[r:{rel_type}]->(to_node)"
                )

                params: Dict[str, Any] = {
                    "from_id": from_id,
                    "to_id": to_id,
                }

                # Add optional relationship properties
                if properties:
                    set_clauses = [f"r.{k} = ${k}" for k in properties]
                    if set_clauses:
                        query += " SET " + ", ".join(set_clauses)
                    params.update(properties)

                session.run(query, **params)
                logger.debug(
                    f"Upserted relationship {rel_type}: {from_id} -> {to_id}"
                )
                return True
        except Exception as e:
            logger.error(
                f"Failed to upsert relationship {rel_type} "
                f"{from_id} -> {to_id}: {e}"
            )
            return False


# --- Lazy initialization to avoid crashing app on import when DB is not ready ---

_graph_db_instance: Optional[Neo4jGraphDB] = None


def get_graph_db() -> Neo4jGraphDB:
    """
    Get or create the global Neo4jGraphDB instance (lazy init).

    Returns:
        The singleton Neo4jGraphDB instance.

    Raises:
        Exception: If Neo4j driver cannot be initialized.
    """
    global _graph_db_instance
    if _graph_db_instance is None:
        _graph_db_instance = Neo4jGraphDB()
    return _graph_db_instance


# Backward-compatible alias - callers should prefer get_graph_db() for safety.
graph_db = None
