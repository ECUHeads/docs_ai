"""
Tests for context_assembler module.

Covers:
    - ContextAssembler.assemble() with various node types
    - Language switching (Thai / English)
    - Creative field toggling
    - Node deduplication
    - Score-based sorting
    - Missing property handling (N/A fallback)
    - Relationship formatting and grouping
"""

import pytest
from context_assembler import (
    ContextAssembler,
    format_concept_block,
    format_relationships,
    _deduplicate,
    _sort_by_score,
)

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

def make_concept(node_id: str = "c1", score: float = 0.9) -> dict:
    """Create a sample KnowledgeConcept node."""
    return {
        "id": node_id,
        "label": "KnowledgeConcept",
        "name_th": f"แนวคิดทดสอบ {node_id}",
        "name_en": f"Test Concept {node_id}",
        "domain": "AI/ML",
        "technical_desc_th": f"คำอธิบายทางเทคนิคของ {node_id}",
        "technical_desc_en": f"Technical description of {node_id}",
        "layman_explanation": f"คำอธิบายแบบง่ายของ {node_id}",
        "analogy": f"อุปมาสำหรับ {node_id}",
        "fiction_seed": f"แนวคิดสร้างสรรค์ {node_id}",
        "visual_concept": f"ภาพเชิงแนวคิด {node_id}",
        "score": score,
    }


def make_hardware(node_id: str = "h1", score: float = 0.7) -> dict:
    """Create a sample Hardware node."""
    return {
        "id": node_id,
        "label": "Hardware",
        "name_th": f"ฮาร์ดแวร์ {node_id}",
        "name_en": f"Hardware {node_id}",
        "domain": "Computing",
        "technical_desc_th": f"สเปคของ {node_id}",
        "technical_desc_en": f"Specs of {node_id}",
        "layman_explanation": f"อธิบายง่าย ๆ ของ {node_id}",
        "score": score,
    }


def make_strategy(node_id: str = "s1", score: float = 0.8) -> dict:
    """Create a sample Strategy node."""
    return {
        "id": node_id,
        "label": "Strategy",
        "name_th": f"กลยุทธ์ {node_id}",
        "name_en": f"Strategy {node_id}",
        "domain": "Trading",
        "technical_desc_th": f"รายละเอียดกลยุทธ์ {node_id}",
        "technical_desc_en": f"Strategy details for {node_id}",
        "layman_explanation": f"อธิบายกลยุทธ์ {node_id} แบบง่าย",
        "analogy": f"อุปมากลยุทธ์ {node_id}",
        "score": score,
    }


def make_narrative(node_id: str = "n1", score: float = 0.6) -> dict:
    """Create a sample Narrative node."""
    return {
        "id": node_id,
        "label": "Narrative",
        "name_th": f"เรื่องราว {node_id}",
        "name_en": f"Narrative {node_id}",
        "domain": "Storytelling",
        "technical_desc_th": f"เนื้อหาเรื่อง {node_id}",
        "technical_desc_en": f"Story content for {node_id}",
        "layman_explanation": f"สรุปเรื่อง {node_id}",
        "fiction_seed": f"เมล็ดพันธุ์นิยาย {node_id}",
        "score": score,
    }


def make_relationship(source: str = "c1", target: str = "h1",
                      rel_type: str = "relates_to") -> dict:
    """Create a sample relationship dict."""
    return {
        "type": rel_type,
        "source": source,
        "target": target,
    }


# --------------------------------------------------------------------------- #
#  Tests: _deduplicate
# --------------------------------------------------------------------------- #

class TestDeduplicate:
    def test_removes_duplicates_by_id(self):
        nodes = [make_concept("c1"), make_concept("c1"), make_concept("c2")]
        result = _deduplicate(nodes)
        assert len(result) == 2
        ids = [n["id"] for n in result]
        assert ids == ["c1", "c2"]

    def test_preserves_order(self):
        nodes = [make_concept("a"), make_hardware("b"), make_strategy("c")]
        result = _deduplicate(nodes)
        assert len(result) == 3
        assert [n["id"] for n in result] == ["a", "b", "c"]

    def test_empty_input(self):
        assert _deduplicate([]) == []

    def test_nodes_without_id_kept_once(self):
        nodes = [{"label": "X"}, {"label": "Y"}]
        result = _deduplicate(nodes)
        # Nodes without id are not deduplicated (no key to track)
        assert len(result) == 2


# --------------------------------------------------------------------------- #
#  Tests: _sort_by_score
# --------------------------------------------------------------------------- #

class TestSortByScore:
    def test_descending_order(self):
        nodes = [make_concept("low", 0.3), make_concept("high", 0.9)]
        result = _sort_by_score(nodes, reverse=True)
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.3

    def test_unscored_at_end(self):
        nodes = [
            make_concept("scored", 0.5),
            {"id": "no_score", "label": "X"},
        ]
        result = _sort_by_score(nodes, reverse=True)
        assert result[0]["id"] == "scored"
        assert result[1]["id"] == "no_score"

    def test_empty_input(self):
        assert _sort_by_score([]) == []


# --------------------------------------------------------------------------- #
#  Tests: format_concept_block
# --------------------------------------------------------------------------- #

class TestFormatConceptBlock:
    def test_thai_output(self):
        concept = make_concept()
        block = format_concept_block(concept, language="th", include_creative=True)
        assert "## แนวคิดทดสอบ c1" in block
        assert "**โดเมน:** AI/ML" in block
        assert "**คำอธิบายทางเทคนิค:**" in block
        assert "**อุปมา:**" in block
        assert "**แนวคิดสร้างสรรค์:**" in block

    def test_english_output(self):
        concept = make_concept()
        block = format_concept_block(concept, language="en", include_creative=True)
        assert "## Test Concept c1" in block
        assert "**Domain:** AI/ML" in block
        assert "**Technical Description:**" in block

    def test_creative_excluded(self):
        concept = make_concept()
        block = format_concept_block(concept, language="th", include_creative=False)
        assert "**อุปมา:**" not in block
        assert "**แนวคิดสร้างสรรค์:**" not in block
        assert "**ภาพเชิงแนวคิด:**" not in block

    def test_missing_properties_use_na(self):
        bare = {"id": "bare1", "label": "KnowledgeConcept"}
        block = format_concept_block(bare, language="en", include_creative=True)
        assert "N/A" in block


# --------------------------------------------------------------------------- #
#  Tests: format_relationships
# --------------------------------------------------------------------------- #

class TestFormatRelationships:
    def test_groups_by_type(self):
        rels = [
            make_relationship("A", "B", "relates_to"),
            make_relationship("C", "D", "relates_to"),
            make_relationship("A", "E", "depends_on"),
        ]
        block = format_relationships(rels, language="en")
        assert "relates to" in block.lower()
        assert "depends on" in block.lower()

    def test_thai_labels(self):
        rels = [make_relationship("X", "Y", "causes")]
        block = format_relationships(rels, language="th")
        assert "ทำให้เกิด" in block

    def test_empty_input_returns_empty(self):
        assert format_relationships([], language="en") == ""

    def test_unknown_rel_type_falls_back_to_raw(self):
        rels = [{"type": "custom_link", "source": "S", "target": "T"}]
        block = format_relationships(rels, language="en")
        assert "custom_link" in block


# --------------------------------------------------------------------------- #
#  Tests: ContextAssembler.assemble()
# --------------------------------------------------------------------------- #

class TestContextAssembler:
    def test_basic_assembly_thai(self):
        assembler = ContextAssembler()
        nodes = [make_concept("c1"), make_hardware("h1")]
        result = assembler.assemble(nodes, language="th")
        assert "## แนวคิดหลัก" in result
        assert "## ฮาร์ดแวร์" in result

    def test_basic_assembly_english(self):
        assembler = ContextAssembler()
        nodes = [make_concept("c1"), make_strategy("s1")]
        result = assembler.assemble(nodes, language="en")
        assert "## Key Concepts" in result
        assert "## Strategies" in result

    def test_includes_relationships(self):
        assembler = ContextAssembler()
        nodes = [make_concept("c1")]
        rels = [make_relationship("c1", "h1")]
        result = assembler.assemble(nodes, rels, language="en")
        assert "## Relationships" in result

    def test_excludes_creative_when_disabled(self):
        assembler = ContextAssembler()
        nodes = [make_concept("c1")]
        result = assembler.assemble(nodes, include_creative=False)
        assert "**Analogy:**" not in result
        assert "**Creative Seed:**" not in result

    def test_deduplicates_input_nodes(self):
        assembler = ContextAssembler()
        c = make_concept("dup")
        nodes = [c, c, c]
        result = assembler.assemble(nodes, language="en")
        # Count how many times the concept name appears (should be once)
        count = result.count("Test Concept dup")
        assert count == 1

    def test_sorts_by_score(self):
        assembler = ContextAssembler()
        nodes = [
            make_concept("low", score=0.2),
            make_concept("high", score=0.95),
        ]
        result = assembler.assemble(nodes, language="en")
        # High-score concept should appear before low-score
        high_pos = result.index("Test Concept high")
        low_pos = result.index("Test Concept low")
        assert high_pos < low_pos

    def test_empty_inputs_returns_no_results(self):
        assembler = ContextAssembler(default_language="th")
        result = assembler.assemble([], [])
        assert "ไม่พบผลลัพธ์" in result

    def test_mixed_node_types_grouped_correctly(self):
        assembler = ContextAssembler()
        nodes = [
            make_narrative("n1"),
            make_concept("c1"),
            make_hardware("h1"),
            make_strategy("s1"),
        ]
        result = assembler.assemble(nodes, language="en")
        assert "## Key Concepts" in result
        assert "## Hardware" in result
        assert "## Strategies" in result
        assert "## Narratives" in result

    def test_unknown_label_goes_to_other(self):
        assembler = ContextAssembler()
        nodes = [{"id": "x1", "label": "ExoticType", "name_en": "Weird"}]
        result = assembler.assemble(nodes, language="en")
        assert "## Additional Information" in result

    def test_default_language_fallback(self):
        assembler = ContextAssembler(default_language="en")
        nodes = [make_concept("c1")]
        result = assembler.assemble(nodes)  # No language arg
        assert "## Key Concepts" in result
