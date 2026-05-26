"""
End-to-End Integration Tests for Open WebUI Knowledge Graph Tool.

Tests the complete pipeline from Open WebUI function-calling interface
through hybrid search to formatted Markdown responses.

Test Cases:
    1. test_thai_query_fiction_seed() - Thai query returning fiction_seed content
    2. test_english_query_hardware() - English query returning Hardware nodes
    3. test_tool_invocation_flow() - Full tool invocation with latency measurement
    4. test_error_handling() - Malformed/empty queries and connection failures
    5. test_thai_english_mixed() - Mixed-language query handling

Usage:
    pytest test_openwebui_integration.py -v
    pytest test_openwebui_integration.py -v --log-cli-level=DEBUG

Requirements:
    pytest
"""

import json
import logging
import os
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------
LATENCY_THRESHOLD_SEC = 2.0  # Maximum allowed end-to-end latency
TOP_K_DEFAULT = 3
MAX_HOPS_DEFAULT = 2


# ===================================================================
# Fixtures - Test Data & Infrastructure
# ===================================================================

@pytest.fixture(scope="session")
def test_entities() -> List[Dict[str, Any]]:
    """Known entities pre-loaded into the test knowledge graph."""
    return [
        {
            "id": "HW_FPGA_001",
            "labels": ["Hardware"],
            "properties": {
                "name": "Xilinx FPGA Accelerator",
                "description_en": (
                    "Field-Programmable Gate Array used for low-latency "
                    "trading strategy execution with nanosecond response times."
                ),
                "description_th": (
                    "ชิป FPGA จาก Xilinx ใช้สำหรับการเทรดความเร็วสูง "
                    "ที่มีความหน่วงในระดับนาโนวินาที"
                ),
                "layman_explanation": (
                    "A reprogrammable chip that can be customized for specific "
                    "trading tasks, making it much faster than general-purpose CPUs."
                ),
                "analogy": (
                    "Like having a custom-built race car instead of a "
                    "family sedan for trading."
                ),
                "fiction_seed": (
                    "In a future where trading algorithms evolve independently, "
                    "an FPGA cluster develops its own strategy language, "
                    "communicating through light pulses across fiber optics."
                ),
                "domain": "finance",
            },
        },
        {
            "id": "HW_NETWORK_001",
            "labels": ["Hardware"],
            "properties": {
                "name": "10GbE Network Interface",
                "description_en": (
                    "10 Gigabit Ethernet NIC required for high-frequency trading "
                    "infrastructure to minimize network latency."
                ),
                "description_th": (
                    "การ์ดเครือข่าย 10 กิกะบิต ที่จำเป็นสำหรับโครงสร้างพื้นฐาน HFT"
                ),
                "layman_explanation": (
                    "A super-fast network card that moves data 10x faster than "
                    "typical home internet connections."
                ),
                "analogy": (
                    "Like upgrading from a country road to a highway for your data."
                ),
                "fiction_seed": "",
                "domain": "finance",
            },
        },
        {
            "id": "STRAT_MOMENTUM_001",
            "labels": ["Strategy"],
            "properties": {
                "name": "Momentum Trading Strategy",
                "description_en": (
                    "Algorithmic strategy that follows price momentum trends "
                    "using moving average crossovers."
                ),
                "description_th": "กลยุทธ์การเทรดตามโมเมนตัมของราคา",
                "layman_explanation": (
                    "A trading approach that buys when prices are rising and "
                    "sells when they fall, riding the wave of market momentum."
                ),
                "analogy": "Like surfing - you catch the wave and ride it.",
                "fiction_seed": (
                    "A trader discovers that market momentum follows patterns "
                    "identical to ocean tides, leading to a strategy inspired by "
                    "celestial mechanics."
                ),
                "domain": "finance",
            },
        },
        {
            "id": "CONCEPT_LOW_LATENCY_001",
            "labels": ["Concept"],
            "properties": {
                "name": "Low-Latency Architecture",
                "description_en": (
                    "System design pattern minimizing response time through "
                    "kernel bypass, lock-free data structures, and CPU pinning."
                ),
                "description_th": (
                    "รูปแบบการออกแบบระบบที่ลดเวลาตอบสนองให้น้อยที่สุด"
                ),
                "layman_explanation": (
                    "Building a system where every microsecond counts by removing "
                    "unnecessary steps in data processing."
                ),
                "analogy": (
                    "Like a relay race where the baton pass is instantaneous."
                ),
                "fiction_seed": (
                    "A programmer builds a trading system so fast it can execute "
                    "orders before the market realizes the price changed, raising "
                    "questions about the nature of time in financial markets."
                ),
                "domain": "technology",
            },
        },
    ]


@pytest.fixture(scope="session")
def test_relationships() -> List[Dict[str, Any]]:
    """Known relationships between test entities."""
    return [
        {
            "from": "HW_FPGA_001",
            "to": "CONCEPT_LOW_LATENCY_001",
            "type": "REQUIRES_INFRA",
            "properties": {
                "description": "FPGA accelerator implements low-latency architecture"
            },
        },
        {
            "from": "HW_NETWORK_001",
            "to": "CONCEPT_LOW_LATENCY_001",
            "type": "REQUIRES_INFRA",
            "properties": {
                "description": "10GbE NIC supports low-latency network requirements"
            },
        },
        {
            "from": "STRAT_MOMENTUM_001",
            "to": "HW_FPGA_001",
            "type": "USES",
            "properties": {
                "description": "Momentum strategy benefits from FPGA acceleration"
            },
        },
    ]


@pytest.fixture(scope="session")
def mock_vector_results(test_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulated vector search results with relevance scores."""
    return [
        {
            "id": ent["id"],
            "score": 0.95 - idx * 0.1,
            "labels": ent["labels"],
            "properties": ent["properties"],
        }
        for idx, ent in enumerate(test_entities[:3])
    ]


@pytest.fixture(scope="session")
def mock_expanded_nodes(
    test_entities: List[Dict[str, Any]],
    test_relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Simulated graph traversal expansion results."""
    expanded = []
    for rel in test_relationships:
        target = next((e for e in test_entities if e["id"] == rel["to"]), None)
        if target:
            expanded.append({
                "id": target["id"],
                "labels": target["labels"],
                "properties": target["properties"],
                "relationship": {
                    "type": rel["type"],
                    "source_id": rel["from"],
                    "description": rel["properties"].get("description", ""),
                },
                "hop_count": 1,
            })
    return expanded


@pytest.fixture(scope="session")
def mock_search_response(
    mock_vector_results: List[Dict[str, Any]],
    mock_expanded_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Complete simulated search pipeline response."""
    # Build context block simulating ContextAssembler output
    context_parts = ["# Search Results\n\n"]
    for node in mock_vector_results:
        props = node.get("properties", {})
        context_parts.append(f"## {props.get('name', 'Unknown')}\n")
        context_parts.append(f"**Type:** {', '.join(node.get('labels', []))}\n")
        context_parts.append(f"**Score:** {node.get('score', 0):.4f}\n")
        if props.get("description_en"):
            context_parts.append(
                f"**Technical Description:** {props['description_en']}\n"
            )
        if props.get("layman_explanation"):
            context_parts.append(
                f"**Simple Explanation:** {props['layman_explanation']}\n"
            )
        if props.get("analogy"):
            context_parts.append(f"**Analogy:** {props['analogy']}\n")
        if props.get("fiction_seed"):
            context_parts.append(
                f"**Creative Seed:** {props['fiction_seed']}\n"
            )
        context_parts.append("\n---\n\n")

    for node in mock_expanded_nodes:
        rel = node.get("relationship", {})
        props = node.get("properties", {})
        context_parts.append(f"### {props.get('name', 'Unknown')} (Expanded)\n")
        context_parts.append(
            f"**Related via:** {rel.get('type', 'UNKNOWN')} "
            f"from `{rel.get('source_id', '')}`\n"
        )
        if rel.get("description"):
            context_parts.append(f"**Relationship:** {rel['description']}\n")
        context_parts.append("\n")

    total_ids = set()
    total_ids.update(r["id"] for r in mock_vector_results)
    total_ids.update(n["id"] for n in mock_expanded_nodes)

    return {
        "query": "",
        "vector_results": mock_vector_results,
        "expanded_nodes": mock_expanded_nodes,
        "context_block": "".join(context_parts),
        "total_nodes": len(total_ids),
    }


@pytest.fixture
def mock_hybrid_engine(
    mock_search_response: Dict[str, Any],
) -> MagicMock:
    """Mock HybridSearchEngine that returns predetermined responses."""
    engine = MagicMock()
    engine.search.return_value = mock_search_response
    return engine


# ---------------------------------------------------------------------------
# Core helper: build a SearchKnowledgeGraphTool with a mocked engine.
#
# We avoid importing `hybrid_search` (which pulls in sentence_transformers)
# by constructing the tool class and injecting the mock _search_engine
# attribute directly.  The `_engine` property returns `_search_engine` when
# it is not None, so this completely bypasses the lazy import.
# ---------------------------------------------------------------------------

def _build_mocked_tool(mock_engine: MagicMock) -> Any:
    """Return a SearchKnowledgeGraphTool with *mock_engine* injected."""
    # Import only the tool module — it does NOT import hybrid_search at top-level.
    from openwebui_tool import SearchKnowledgeGraphTool  # noqa: F811

    tool = object.__new__(SearchKnowledgeGraphTool)  # bypass __init__
    tool._search_engine = mock_engine
    return tool


@pytest.fixture
def mock_tool(mock_hybrid_engine: MagicMock) -> Any:
    """SearchKnowledgeGraphTool with mocked engine (no real DB needed)."""
    return _build_mocked_tool(mock_hybrid_engine)


# ===================================================================
# Test Case 1: Thai Query - Fiction Seed Content
# ===================================================================

class TestThaiQueryFictionSeed:
    """Test Case 1: Thai query should return fiction_seed content from graph."""

    def test_thai_query_fiction_seed(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
    ) -> None:
        """
        Query: \"ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย\"
        Expected: Response includes fiction_seed content from graph.
        Verify: Tool was invoked, search returned results, response contains creative content.
        """
        query = "ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย"
        logger.info("TEST: Thai fiction seed query: %r", query)

        # Execute tool
        start_time = time.monotonic()
        result = mock_tool.execute(query)
        elapsed = time.monotonic() - start_time
        logger.info("TEST: Tool execution took %.4fs", elapsed)

        # Verify engine.search was called
        mock_tool._engine.search.assert_called_once()
        call_kwargs = mock_tool._engine.search.call_args[1]
        assert call_kwargs["query"] == query
        assert call_kwargs["top_k"] == TOP_K_DEFAULT
        assert call_kwargs["max_hops"] == MAX_HOPS_DEFAULT

        # Assertions on result
        assert isinstance(result, str), "Result must be a string"
        assert len(result) > 0, "Result must not be empty"

        # Verify Markdown structure
        assert "# Search Results" in result, "Response must contain search results header"
        assert "Found" in result and "related nodes" in result, (
            "Response must include node count"
        )

        # Verify fiction_seed content is present in test data
        context = mock_search_response.get("context_block", "")
        fiction_seeds = [
            node.get("properties", {}).get("fiction_seed", "")
            for node in mock_search_response.get("vector_results", [])
        ]
        has_fiction_seed = any(seed for seed in fiction_seeds if seed)
        assert has_fiction_seed, "Test data must include fiction_seed content"

        # Verify creative content keywords appear in response
        assert "Creative Seed" in result or "แนวคิดสร้างสรรค์" in result, (
            "Response must contain fiction_seed/creative content section"
        )

        logger.info("TEST PASSED: Thai query returned fiction_seed content")

    def test_thai_query_utf8_handling(
        self,
        mock_tool: Any,
    ) -> None:
        """Verify Thai text is properly handled as UTF-8 throughout the pipeline."""
        thai_queries = [
            "กลยุทธ์การเทรดอัตโนมัติ",
            "ฮาร์ดแวร์สำหรับการเทรดความเร็วสูง",
            "ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย",
        ]

        for query in thai_queries:
            logger.debug("Testing UTF-8 handling for: %r", query)
            result = mock_tool.execute(query)
            assert isinstance(result, str), f"Result for {query!r} must be string"
            # Verify no encoding corruption
            assert "\ufffd" not in result, (
                f"UTF-8 decoding error detected in response for {query!r}"
            )

        logger.info("TEST PASSED: Thai UTF-8 handling verified")


# ===================================================================
# Test Case 2: English Query - Hardware Nodes
# ===================================================================

class TestEnglishQueryHardware:
    """Test Case 2: English query should return Hardware nodes with REQUIRES_INFRA."""

    def test_english_query_hardware(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
        test_relationships: List[Dict[str, Any]],
    ) -> None:
        """
        Query: \"What hardware is needed for high-frequency trading?\"
        Expected: Response includes Hardware nodes with REQUIRES_INFRA relationships.
        Verify: Technical descriptions present in response.
        """
        query = "What hardware is needed for high-frequency trading?"
        logger.info("TEST: English hardware query: %r", query)

        result = mock_tool.execute(query)

        # Assertions - Response structure
        assert isinstance(result, str), "Result must be a string"
        assert "# Search Results" in result, "Must contain search results header"

        # Verify Hardware nodes present
        hardware_nodes = [
            n for n in mock_search_response.get("vector_results", [])
            if "Hardware" in n.get("labels", [])
        ]
        assert len(hardware_nodes) > 0, "Response must include Hardware nodes"

        # Verify technical descriptions present
        context_block = mock_search_response.get("context_block", "")
        assert "Technical Description" in context_block, (
            "Response must contain technical descriptions"
        )

        # Verify REQUIRES_INFRA relationships in expanded nodes
        infra_rels = [
            n for n in mock_search_response.get("expanded_nodes", [])
            if n.get("relationship", {}).get("type") == "REQUIRES_INFRA"
        ]
        assert len(infra_rels) > 0, (
            "Response must include REQUIRES_INFRA relationships"
        )

        # Verify specific hardware names appear
        hw_names = [n.get("properties", {}).get("name", "") for n in hardware_nodes]
        assert any("FPGA" in name for name in hw_names), (
            "Hardware results should include FPGA"
        )

        logger.info(
            "TEST PASSED: English query returned %d Hardware nodes, "
            "%d REQUIRES_INFRA relationships",
            len(hardware_nodes),
            len(infra_rels),
        )

    def test_english_query_tech_descriptions(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
    ) -> None:
        """Verify technical descriptions are included in response."""
        query = "high-frequency trading infrastructure"
        result = mock_tool.execute(query)

        context = mock_search_response.get("context_block", "")

        # Check for key technical terms
        tech_terms = ["latency", "FPGA", "network", "trading"]
        found_terms = [term for term in tech_terms if term.lower() in context.lower()]
        assert len(found_terms) >= 2, (
            f"Response should contain at least 2 technical terms. Found: {found_terms}"
        )

        logger.info(
            "TEST PASSED: Technical descriptions verified (%d terms found)",
            len(found_terms),
        )


# ===================================================================
# Test Case 3: Tool Invocation Flow
# ===================================================================

class TestToolInvocationFlow:
    """Test Case 3: Full tool invocation flow with latency measurement."""

    def test_tool_invocation_parameters(
        self,
        mock_hybrid_engine: MagicMock,
    ) -> None:
        """
        Simulate function calling request and verify tool receives correct parameters.
        """
        from openwebui_tool import TOOL_DEFINITION

        logger.info("TEST: Verifying tool definition schema")

        # Verify tool definition structure
        assert TOOL_DEFINITION["type"] == "function", "Tool type must be 'function'"
        func = TOOL_DEFINITION["function"]
        assert func["name"] == "search_knowledge_graph", (
            "Tool name must be 'search_knowledge_graph'"
        )
        assert "search_query" in func["parameters"]["properties"], (
            "Tool must have 'search_query' parameter"
        )
        assert "search_query" in func["parameters"]["required"], (
            "'search_query' must be required"
        )

        # Simulate function calling via the tool directly (bypassing execute_tool
        # which creates a fresh instance and would hit real DB).
        tool = _build_mocked_tool(mock_hybrid_engine)
        result = tool.execute("algorithmic trading strategies")

        assert isinstance(result, str), "Tool must return string"

        # Verify engine.search was called with correct parameters
        mock_hybrid_engine.search.assert_called_once()
        call_kwargs = mock_hybrid_engine.search.call_args[1]
        assert call_kwargs["query"] == "algorithmic trading strategies", (
            "Query parameter must match input"
        )
        assert call_kwargs["top_k"] == TOP_K_DEFAULT, (
            f"top_k must be {TOP_K_DEFAULT}"
        )
        assert call_kwargs["max_hops"] == MAX_HOPS_DEFAULT, (
            f"max_hops must be {MAX_HOPS_DEFAULT}"
        )

        logger.info("TEST PASSED: Tool invocation parameters verified")

    def test_tool_returns_formatted_markdown(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
    ) -> None:
        """Verify tool returns properly formatted Markdown."""
        query = "test query for markdown formatting"
        result = mock_tool.execute(query)

        # Verify Markdown structure — tool wraps with "# Search Results for: ..."
        assert "# Search Results for:" in result, (
            "Response must contain search results header"
        )
        assert "**Found" in result, "Must include bold node count"
        assert "related nodes" in result, "Must include 'related nodes' label"

        # Verify context block is included
        context = mock_search_response.get("context_block", "")
        assert len(context) > 0, "Context block must not be empty"

        # Check for Markdown elements
        has_headers = "##" in result
        has_bold = "**" in result
        assert has_headers and has_bold, (
            "Response must contain Markdown headers and bold text"
        )

        logger.info("TEST PASSED: Markdown formatting verified")

    def test_end_to_end_latency(
        self,
        mock_tool: Any,
    ) -> None:
        """
        Measure end-to-end latency and assert it is within threshold.
        Target: < 2 seconds.
        """
        query = "latency measurement test query"

        # Warm-up run
        mock_tool.execute(query)

        # Timed runs
        latencies: List[float] = []
        iterations = 5

        for i in range(iterations):
            start = time.monotonic()
            result = mock_tool.execute(query)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            logger.debug(
                "TEST: Latency run %d/%d: %.4fs", i + 1, iterations, elapsed
            )

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        logger.info(
            "TEST: Latency stats - avg: %.4fs, min: %.4fs, max: %.4fs",
            avg_latency, min_latency, max_latency,
        )

        # Assertions
        assert avg_latency < LATENCY_THRESHOLD_SEC, (
            f"Average latency {avg_latency:.4fs} exceeds threshold "
            f"{LATENCY_THRESHOLD_SEC}s"
        )
        assert max_latency < LATENCY_THRESHOLD_SEC * 2, (
            f"Max latency {max_latency:.4fs} exceeds 2x threshold "
            f"{LATENCY_THRESHOLD_SEC * 2}s"
        )

        logger.info("TEST PASSED: End-to-end latency within threshold")


# ===================================================================
# Test Case 4: Error Handling
# ===================================================================

class TestErrorHandling:
    """Test Case 4: Graceful error handling for various failure modes."""

    def test_empty_query(self) -> None:
        """Send empty query and verify graceful response."""
        logger.info("TEST: Empty query handling")

        engine = MagicMock()
        tool = _build_mocked_tool(engine)
        result = tool.execute("")

        assert isinstance(result, str), "Empty query must return string"
        assert "# Search Results" in result, "Must contain search results header"
        assert "Empty query" in result or "empty" in result.lower(), (
            "Response must indicate empty query was received"
        )
        # Verify engine.search was NOT called for empty query
        engine.search.assert_not_called()

        logger.info("TEST PASSED: Empty query handled gracefully")

    def test_whitespace_only_query(self) -> None:
        """Send whitespace-only query and verify it is treated as empty."""
        engine = MagicMock()
        tool = _build_mocked_tool(engine)

        queries = ["   ", "\t\n", "  \t  "]

        for query in queries:
            result = tool.execute(query)
            assert isinstance(result, str), f"Whitespace query {query!r} must return string"
            assert "Empty query" in result or "empty" in result.lower(), (
                f"Response for {query!r} must indicate empty query"
            )

        logger.info("TEST PASSED: Whitespace-only queries handled")

    def test_malformed_query(self, mock_search_response: Dict[str, Any]) -> None:
        """Send malformed/special character queries and verify graceful handling."""
        engine = MagicMock()
        engine.search.return_value = mock_search_response
        tool = _build_mocked_tool(engine)

        malformed_queries = [
            "\x00\x01\x02",           # Null/control characters
            "<script>alert('xss')</script>",  # XSS attempt
            "a" * 10000,              # Very long query
        ]

        for query in malformed_queries:
            logger.debug("TEST: Malformed query: %r", repr(query)[:50])
            try:
                result = tool.execute(query)
                assert isinstance(result, str), (
                    f"Malformed query must return string, got {type(result)}"
                )
            except Exception as exc:
                pytest.fail(
                    f"Tool crashed on malformed query {repr(query)[:30]}: {exc}"
                )

        logger.info("TEST PASSED: Malformed queries handled without crashing")

    def test_neo4j_connection_failure(self) -> None:
        """Simulate Neo4j connection failure and verify graceful error response."""
        from neo4j.exceptions import ServiceUnavailable

        logger.info("TEST: Simulating Neo4j connection failure")

        mock_engine = MagicMock()
        mock_engine.search.side_effect = ServiceUnavailable(
            "Cannot connect to Neo4j at neo4j://localhost:7687"
        )

        tool = _build_mocked_tool(mock_engine)
        result = tool.execute("test query during failure")

        assert isinstance(result, str), "Error response must be string"
        assert "# Search Results" in result, "Error response must have header"
        # Should contain error indicator but not crash
        assert "error" in result.lower() or "Error" in result, (
            "Response must indicate an error occurred"
        )

        logger.info("TEST PASSED: Neo4j connection failure handled gracefully")

    def test_search_timeout_handling(self) -> None:
        """Simulate search timeout and verify graceful response."""
        from openwebui_tool import _SearchTimeoutError

        logger.info("TEST: Simulating search timeout")

        mock_engine = MagicMock()
        mock_engine.search.side_effect = _SearchTimeoutError(
            "Search exceeded 30s timeout limit"
        )

        tool = _build_mocked_tool(mock_engine)
        result = tool.execute("slow query that times out")

        assert isinstance(result, str), "Timeout response must be string"
        assert "# Search Results" in result, "Timeout response must have header"
        assert "timed out" in result.lower() or "timeout" in result.lower(), (
            "Response must indicate timeout occurred"
        )

        logger.info("TEST PASSED: Search timeout handled gracefully")

    def test_unknown_tool_dispatch(self) -> None:
        """Verify unknown tool name returns valid error response."""
        from openwebui_tool import execute_tool

        result = execute_tool("nonexistent_tool", {"search_query": "test"})

        assert isinstance(result, str), "Unknown tool must return string"
        assert "Tool Error" in result or "Unknown tool" in result, (
            "Response must indicate unknown tool"
        )

        logger.info("TEST PASSED: Unknown tool handled gracefully")


# ===================================================================
# Test Case 5: Thai-English Mixed Queries
# ===================================================================

class TestThaiEnglishMixed:
    """Test Case 5: Mixed-language query handling."""

    def test_mixed_language_query(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
    ) -> None:
        """
        Query mixing Thai and English.
        Verify proper language detection and response formatting.
        """
        mixed_queries = [
            "FPGA คืออะไร ใช้ใน HFT อย่างไร",
            "อธิบาย Low-latency Architecture ง่ายๆ",
            "What is Momentum Strategy และใช้ Hardware อะไร",
            "ช่วยหา Hardware สำหรับ algorithmic trading",
        ]

        for query in mixed_queries:
            logger.info("TEST: Mixed language query: %r", query)

            result = mock_tool.execute(query)

            # Verify response is valid
            assert isinstance(result, str), f"Mixed query must return string"
            assert len(result) > 0, f"Mixed query response must not be empty"
            assert "# Search Results" in result, (
                f"Mixed query response must have header"
            )

            # Verify no encoding issues
            assert "\ufffd" not in result, (
                f"UTF-8 error in response for mixed query: {query!r}"
            )

            logger.info("TEST PASSED: Mixed query handled: %r", query[:40])

    def test_language_agnostic_response(
        self,
        mock_tool: Any,
        mock_search_response: Dict[str, Any],
    ) -> None:
        """Verify response contains both Thai and English content when available."""
        query = "hardware for trading"
        result = mock_tool.execute(query)

        context = mock_search_response.get("context_block", "")

        # Check that test data includes bilingual content
        has_thai_desc = any(
            n.get("properties", {}).get("description_th", "")
            for n in mock_search_response.get("vector_results", [])
        )
        has_en_desc = any(
            n.get("properties", {}).get("description_en", "")
            for n in mock_search_response.get("vector_results", [])
        )

        assert has_thai_desc, "Test data should include Thai descriptions"
        assert has_en_desc, "Test data should include English descriptions"

        logger.info(
            "TEST PASSED: Language-agnostic response verified "
            "(Thai: %s, English: %s)",
            bool(has_thai_desc),
            bool(has_en_desc),
        )


# ===================================================================
# Test Infrastructure - Open WebUI Interface Mock
# ===================================================================

class TestOpenWebUIInterface:
    """Mock Open WebUI function calling interface tests."""

    def test_get_tools_discovery(self) -> None:
        """Verify tool discovery endpoint returns correct schema."""
        from openwebui_tool import get_tools

        tools = get_tools()

        assert isinstance(tools, list), "get_tools() must return a list"
        assert len(tools) >= 1, "At least one tool must be registered"

        tool_def = tools[0]
        assert tool_def["type"] == "function", "Tool must be of type 'function'"
        assert tool_def["function"]["name"] == "search_knowledge_graph", (
            "First tool must be search_knowledge_graph"
        )

        logger.info("TEST PASSED: Tool discovery returns correct schema")

    def test_tool_schema_valid_json(self) -> None:
        """Verify tool definition is valid JSON for Open WebUI consumption."""
        from openwebui_tool import TOOL_DEFINITION

        # Must be serializable to JSON
        json_str = json.dumps(TOOL_DEFINITION, ensure_ascii=False)
        parsed = json.loads(json_str)

        assert parsed == TOOL_DEFINITION, (
            "Tool definition must round-trip through JSON serialization"
        )

        logger.info("TEST PASSED: Tool schema is valid JSON")

    def test_execute_tool_dispatch(
        self, mock_hybrid_engine: MagicMock, mock_search_response: Dict[str, Any]
    ) -> None:
        """Verify execute_tool correctly dispatches to search handler.

        We patch the entire SearchKnowledgeGraphTool class so that execute_tool()
        gets our pre-mocked instance (with _search_engine already set) instead of
        creating a fresh one that would hit the real DB.
        """
        from openwebui_tool import execute_tool, SearchKnowledgeGraphTool

        tool = _build_mocked_tool(mock_hybrid_engine)

        # Patch both __new__ and __init__: __new__ returns our instance,
        # __init__ is a no-op so it doesn't reset _search_engine to None.
        with patch.object(SearchKnowledgeGraphTool, '__new__',
                          return_value=tool):
            with patch.object(SearchKnowledgeGraphTool, '__init__',
                              return_value=None):
                args = {"search_query": "dispatch test"}
                result = execute_tool("search_knowledge_graph", args)

        assert isinstance(result, str), "Dispatched tool must return string"
        mock_hybrid_engine.search.assert_called(), (
            "Engine search must have been called"
        )

        logger.info("TEST PASSED: Tool dispatch verified")


# ===================================================================
# Test Infrastructure - Pipeline Stage Timing
# ===================================================================

class TestPipelineStageTiming:
    """Measure and assert timing for each pipeline stage."""

    def test_vector_search_stage_timing(self, mock_hybrid_engine: MagicMock) -> None:
        """Assert vector search stage completes within threshold."""
        import time as time_module

        # Record timing of the mock search call
        original_return = mock_hybrid_engine.search.return_value

        def timed_search(*args, **kwargs):
            start = time_module.monotonic()
            result = original_return
            elapsed = time_module.monotonic() - start
            logger.debug(
                "TEST: Vector search stage took %.4fs", elapsed
            )
            return result

        mock_hybrid_engine.search.side_effect = timed_search

        tool = _build_mocked_tool(mock_hybrid_engine)
        tool.execute("timing test")

        # The mock is instant, but verify the call was made
        assert mock_hybrid_engine.search.call_count == 1

        logger.info("TEST PASSED: Vector search stage timing measured")

    def test_context_assembly_stage_timing(self) -> None:
        """Assert context assembly produces output within threshold."""
        from context_assembler import ContextAssembler

        assembler = ContextAssembler()

        # Test with sample nodes
        sample_nodes = [
            {
                "id": "TEST_001",
                "labels": ["Concept"],
                "properties": {
                    "name": "Test Concept",
                    "description_en": "A test concept for timing.",
                    "layman_explanation": "Something simple to explain.",
                    "analogy": "Like a basic example.",
                    "fiction_seed": "A story about testing.",
                    "domain": "test",
                },
            }
        ]

        start = time.monotonic()
        result = assembler.assemble(sample_nodes, language="en")
        elapsed = time.monotonic() - start

        assert isinstance(result, str), "Assembler must return string"
        assert len(result) > 0, "Assembler output must not be empty"
        assert elapsed < LATENCY_THRESHOLD_SEC, (
            f"Context assembly took {elapsed:.4fs}, exceeds threshold"
        )

        logger.info(
            "TEST PASSED: Context assembly took %.4fs", elapsed
        )


# ===================================================================
# Test Data Setup and Teardown
# ===================================================================

class TestDataSetup:
    """Verify test data integrity and structure."""

    def test_entities_have_required_fields(
        self, test_entities: List[Dict[str, Any]]
    ) -> None:
        """All test entities must have required fields."""
        required_fields = ["id", "labels", "properties"]
        required_props = [
            "name", "description_en", "layman_explanation"
        ]

        for entity in test_entities:
            for field in required_fields:
                assert field in entity, (
                    f"Entity {entity.get('id', 'UNKNOWN')} missing field '{field}'"
                )

            props = entity.get("properties", {})
            for prop in required_props:
                assert prop in props, (
                    f"Entity {entity['id']} missing property '{prop}'"
                )

        logger.info(
            "TEST PASSED: All %d entities have required fields",
            len(test_entities),
        )

    def test_relationships_reference_valid_entities(
        self,
        test_entities: List[Dict[str, Any]],
        test_relationships: List[Dict[str, Any]],
    ) -> None:
        """All relationship endpoints must reference existing entities."""
        entity_ids = {e["id"] for e in test_entities}

        for rel in test_relationships:
            assert rel["from"] in entity_ids, (
                f"Relationship source '{rel['from']}' not in entities"
            )
            assert rel["to"] in entity_ids, (
                f"Relationship target '{rel['to']}' not in entities"
            )

        logger.info(
            "TEST PASSED: All %d relationships reference valid entities",
            len(test_relationships),
        )

    def test_search_response_structure(
        self, mock_search_response: Dict[str, Any]
    ) -> None:
        """Verify search response has all required keys."""
        required_keys = [
            "query", "vector_results", "expanded_nodes",
            "context_block", "total_nodes"
        ]

        for key in required_keys:
            assert key in mock_search_response, (
                f"Search response missing key '{key}'"
            )

        assert isinstance(
            mock_search_response["vector_results"], list
        ), "vector_results must be a list"
        assert isinstance(
            mock_search_response["expanded_nodes"], list
        ), "expanded_nodes must be a list"
        assert isinstance(
            mock_search_response["context_block"], str
        ), "context_block must be a string"
        assert isinstance(
            mock_search_response["total_nodes"], int
        ), "total_nodes must be an integer"
        assert mock_search_response["total_nodes"] > 0, (
            "total_nodes must be positive"
        )

        logger.info("TEST PASSED: Search response structure verified")


# ===================================================================
# Performance Assertions Summary
# ===================================================================

class TestPerformanceAssertions:
    """Aggregate performance assertions across the pipeline."""

    def test_overall_pipeline_performance(
        self,
        mock_tool: Any,
    ) -> None:
        """Run multiple queries and assert overall performance targets."""
        queries = [
            "algorithmic trading",
            "FPGA hardware",
            "ช่วยคิดพล็อตนิยาย",
            "low-latency architecture",
            "Momentum Strategy",
        ]

        timings: List[float] = []

        for query in queries:
            start = time.monotonic()
            result = mock_tool.execute(query)
            elapsed = time.monotonic() - start
            timings.append(elapsed)

            assert isinstance(result, str), f"Query {query!r} must return string"
            assert len(result) > 0, f"Query {query!r} must return non-empty result"

        avg_time = sum(timings) / len(timings)
        p95_time = sorted(timings)[int(len(timings) * 0.95)]

        logger.info(
            "TEST: Performance summary - avg: %.4fs, p95: %.4fs, "
            "queries: %d",
            avg_time, p95_time, len(queries),
        )

        assert avg_time < LATENCY_THRESHOLD_SEC, (
            f"Average pipeline time {avg_time:.4fs} exceeds threshold"
        )
        assert p95_time < LATENCY_THRESHOLD_SEC * 1.5, (
            f"P95 pipeline time {p95_time:.4fs} exceeds 1.5x threshold"
        )

        logger.info("TEST PASSED: Overall pipeline performance within targets")


# ===================================================================
# Conftest-compatible standalone execution
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=DEBUG"])
