"""
Context assembler for hybrid search results.

Takes raw node and relationship dictionaries from hybrid search and assembles
them into structured Markdown context blocks suitable for LLM consumption.

Features:
    - Language-aware formatting (Thai / English)
    - Creative section toggling (analogy, fiction_seed, visual_concept)
    - Node deduplication by ID
    - Relevance-score-based sorting
    - Graceful handling of missing properties via "N/A" placeholders
    - Relationship grouping and readable descriptions

Usage:
    assembler = ContextAssembler()
    markdown = assembler.assemble(nodes, relationships, language="th")
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Translations for relationship types and UI labels
# --------------------------------------------------------------------------- #

_REL_LABELS: Dict[str, Dict[str, str]] = {
    "relates_to": {"th": "เกี่ยวข้องกับ", "en": "relates to"},
    "depends_on": {"th": "ขึ้นอยู่กับ", "en": "depends on"},
    "part_of": {"th": "เป็นส่วนหนึ่งของ", "en": "is part of"},
    "contrasts_with": {"th": "ตรงข้ามกับ", "en": "contrasts with"},
    "causes": {"th": "ทำให้เกิด", "en": "causes"},
    "supports": {"th": "สนับสนุน", "en": "supports"},
    "conflicts_with": {"th": "ขัดแย้งกับ", "en": "conflicts with"},
    "similar_to": {"th": "คล้ายกับ", "en": "similar to"},
}

_UI_LABELS: Dict[str, Dict[str, str]] = {
    "concept_header": {"th": "## แนวคิดหลัก", "en": "## Key Concepts"},
    "hardware_header": {"th": "## ฮาร์ดแวร์", "en": "## Hardware"},
    "strategy_header": {"th": "## กลยุทธ์", "en": "## Strategies"},
    "narrative_header": {"th": "## เรื่องราวเชิงบรรยาย", "en": "## Narratives"},
    "other_header": {"th": "## ข้อมูลเพิ่มเติม", "en": "## Additional Information"},
    "relationships_header": {"th": "## ความสัมพันธ์", "en": "## Relationships"},
    "domain": {"th": "**โดเมน:**", "en": "**Domain:**"},
    "description": {"th": "**คำอธิบายทางเทคนิค:**", "en": "**Technical Description:**"},
    "layman": {"th": "**คำอธิบายแบบง่าย:**", "en": "**Simple Explanation:**"},
    "analogy": {"th": "**อุปมา:**", "en": "**Analogy:**"},
    "fiction_seed": {"th": "**แนวคิดสร้างสรรค์:**", "en": "**Creative Seed:**"},
    "visual_concept": {"th": "**ภาพเชิงแนวคิด:**", "en": "**Visual Concept:**"},
    "no_results": {
        "th": "> **ไม่พบผลลัพธ์** — ไม่มีข้อมูลที่จะแสดง",
        "en": "> **No results** — nothing to display",
    },
    "na": "N/A",
}


def _t(key: str, language: str) -> str:
    """Translate a UI label key for the given language.

    Args:
        key: Label key matching ``_UI_LABELS``.
        language: Language code ("th" or "en").

    Returns:
        Translated string or the key itself when no translation exists.
        Handles both dict-style entries (per-language) and plain strings.
    """
    entry = _UI_LABELS.get(key, key)
    if isinstance(entry, dict):
        return entry.get(language, key)
    return entry  # Plain string value (e.g. "N/A")


def _deduplicate(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate nodes by ``id``, keeping the first occurrence.

    Args:
        nodes: Raw node list potentially containing duplicates.

    Returns:
        Deduplicated list preserving original insertion order.
    """
    seen: Dict[str, bool] = {}
    deduped: List[Dict[str, Any]] = []
    for node in nodes:
        nid = node.get("id", "")
        if nid and nid not in seen:
            seen[nid] = True
            deduped.append(node)
    logger.debug("Deduplicated %d → %d nodes", len(nodes), len(deduped))
    return deduped


def _sort_by_score(nodes: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort nodes by relevance score (descending by default).

    Nodes without a ``score`` field are placed at the end.

    Args:
        nodes: Node list to sort.
        reverse: Sort descending when ``True`` (default).

    Returns:
        New sorted list (does not mutate input).
    """
    scored = [n for n in nodes if n.get("score") is not None and n.get("score", 0) > 0]
    unscored = [n for n in nodes if not scored.count(n)]

    scored.sort(key=lambda n: n.get("score", 0.0), reverse=reverse)
    return scored + unscored


# --------------------------------------------------------------------------- #
#  Formatting helpers
# --------------------------------------------------------------------------- #

def format_concept_block(concept: Dict[str, Any], language: str = "th",
                         include_creative: bool = True) -> str:
    """Format a single concept node as a Markdown block.

    Args:
        concept: Node dictionary with properties like ``name_th``,
                 ``technical_desc_th``, ``analogy``, etc.
        language: Output language ("th" or "en").
        include_creative: Whether to include analogy, fiction_seed,
                          and visual_concept fields.

    Returns:
        Markdown-formatted string for this concept.
    """
    name_key = "name_th" if language == "th" else "name_en"
    desc_key = "technical_desc_th" if language == "th" else "technical_desc_en"

    display_name = concept.get(name_key) or concept.get(
        ("name_en" if name_key == "name_th" else "name_th"), ""
    ) or concept.get("id", _t("na", language))

    domain = concept.get("domain", _t("na", language))
    description = concept.get(desc_key, _t("na", language))
    layman = concept.get("layman_explanation", _t("na", language))

    lines: List[str] = []
    lines.append(f"## {display_name}")
    lines.append(f"{_t('domain', language)} {domain}")
    lines.append(f"{_t('description', language)} {description}")
    lines.append(f"{_t('layman', language)} {layman}")

    if include_creative:
        analogy = concept.get("analogy", _t("na", language))
        fiction = concept.get("fiction_seed", _t("na", language))
        visual = concept.get("visual_concept", _t("na", language))
        lines.append(f"{_t('analogy', language)} {analogy}")
        lines.append(f"{_t('fiction_seed', language)} {fiction}")
        lines.append(f"{_t('visual_concept', language)} {visual}")

    return "\n".join(lines)


def format_relationships(relationships: List[Dict[str, Any]],
                         language: str = "th") -> str:
    """Format a list of relationship dictionaries as a Markdown section.

    Relationships are grouped by type and rendered as readable descriptions
    showing source → target pairs.

    Args:
        relationships: List of relationship dicts, each expected to contain
                       ``type``, ``source`` (or ``source_id``), and
                       ``target`` (or ``target_id``).
        language: Output language ("th" or "en").

    Returns:
        Markdown-formatted string for all relationships.
    """
    if not relationships:
        return ""

    # Group by relationship type
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        grouped.setdefault(rel_type, []).append(rel)

    lines: List[str] = []
    lines.append(f"### {_t('relationships_header', language)}")
    lines.append("")

    for rel_type, rels in grouped.items():
        # Resolve human-readable label for relationship type
        label_map = _REL_LABELS.get(rel_type, {})
        rel_label = label_map.get(language, label_map.get("en", rel_type))

        lines.append(f"**{rel_label}:**")
        lines.append("")

        for rel in rels:
            source = rel.get("source") or rel.get("source_id", "…")
            target = rel.get("target") or rel.get("target_id", "…")
            # Try to enrich with display names if available
            if isinstance(source, dict):
                source = source.get("name_th") or source.get("name_en", source.get("id", "…"))
            if isinstance(target, dict):
                target = target.get("name_th") or target.get("name_en", target.get("id", "…"))

            arrow = " → " if language == "en" else " → "
            lines.append(f"- {source}{arrow}{target}")

        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Main assembler class
# --------------------------------------------------------------------------- #

class ContextAssembler:
    """Assemble hybrid-search results into structured Markdown context.

    Groups nodes by label type (KnowledgeConcept, Hardware, Strategy,
    Narrative), formats each group with language-aware labels, and appends
    a relationship section when relationships are provided.

    Attributes:
        default_language: Fallback language when none is specified ("th").
        default_creative: Whether creative fields are included by default.
    """

    # Label-to-header-key mapping
    _LABEL_MAP: Dict[str, str] = {
        "KnowledgeConcept": "concept_header",
        "Hardware": "hardware_header",
        "Strategy": "strategy_header",
        "Narrative": "narrative_header",
    }

    def __init__(self,
                 default_language: str = "th",
                 default_creative: bool = True) -> None:
        self.default_language = default_language
        self.default_creative = default_creative

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def assemble(self,
                 nodes: List[Dict[str, Any]],
                 relationships: Optional[List[Dict[str, Any]]] = None,
                 language: Optional[str] = None,
                 include_creative: Optional[bool] = None) -> str:
        """Assemble search results into a structured Markdown context block.

        Args:
            nodes: List of node dicts from hybrid search. Each dict should
                   contain at minimum an ``id`` and ``label`` key, plus
                   language-specific name/description fields.
            relationships: Optional list of relationship dicts describing
                           connections between nodes.
            language: Output language ("th" or "en"). Falls back to the
                      instance's ``default_language`` when ``None``.
            include_creative: Whether to include analogy, fiction_seed,
                              and visual_concept fields. Falls back to the
                              instance's ``default_creative`` when ``None``.

        Returns:
            Formatted Markdown string ready for LLM context injection.
            Returns a "no results" block when both inputs are empty.
        """
        lang = language or self.default_language
        creative = include_creative if include_creative is not None else self.default_creative

        if not nodes and not relationships:
            return _t("no_results", lang)

        # ---- Deduplicate & sort ----
        unique_nodes = _deduplicate(nodes) if nodes else []
        sorted_nodes = _sort_by_score(unique_nodes)

        sections: List[str] = []

        # ---- Group by label type ----
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for node in sorted_nodes:
            label = node.get("label", "Other")
            grouped.setdefault(label, []).append(node)

        # Render known label groups first (in canonical order), then others
        canonical_order = ["KnowledgeConcept", "Hardware", "Strategy", "Narrative"]
        rendered_labels: set = set()

        for label in canonical_order:
            if label in grouped:
                header_key = self._LABEL_MAP.get(label, "other_header")
                section = self._build_section(grouped[label], header_key, lang, creative)
                sections.append(section)
                rendered_labels.add(label)

        # Render remaining (non-canonical) labels
        for label in sorted(grouped.keys()):
            if label not in rendered_labels:
                section = self._build_section(grouped[label], "other_header", lang, creative)
                sections.append(section)
                rendered_labels.add(label)

        # ---- Relationship section ----
        if relationships:
            rel_block = format_relationships(relationships, lang)
            if rel_block.strip():
                sections.append(rel_block)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _build_section(self,
                       nodes: List[Dict[str, Any]],
                       header_key: str,
                       language: str,
                       include_creative: bool) -> str:
        """Build a Markdown section for a group of same-label nodes.

        Args:
            nodes: Nodes sharing the same label.
            header_key: Key into ``_UI_LABELS`` for the section header.
            language: Output language.
            include_creative: Whether to render creative fields.

        Returns:
            Markdown string for this section.
        """
        lines: List[str] = []
        lines.append(_t(header_key, language))
        lines.append("")

        for node in nodes:
            block = format_concept_block(node, language, include_creative)
            lines.append(block)
            lines.append("")  # Blank line separator between nodes

        return "\n".join(lines)
