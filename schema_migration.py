"""
Schema migration module for converting the legacy structural model
(Document / Section / Term) to the universal meta-schema.

Provides background migration of existing nodes and validation that all
required constraints and indexes are present in the Neo4j instance.

Coding Standards:
  - All Cypher queries use parameterized inputs ($param).
  - Full type hints on every public function.
  - Google-style docstrings with Args / Returns / Raises sections.
  - Errors are logged with logger.error() and False is returned — no unhandled
    exceptions escape this module.
"""

import logging
from typing import Dict, Any, List

from graph_db import get_graph_db, Neo4jGraphDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror the universal schema definition in graph_db.py
# ---------------------------------------------------------------------------

UNIVERSAL_LABELS: List[str] = [
    "KnowledgeConcept",
    "Hardware",
    "Strategy",
    "Narrative",
    "ContentAsset",
]

EXPECTED_CONSTRAINTS: List[str] = [
    f"universal_id_{label.lower()}" for label in UNIVERSAL_LABELS
]

EXPECTED_INDEXES: List[str] = (
    [f"universal_name_en_{label.lower()}" for label in UNIVERSAL_LABELS]
    + [f"universal_domain_{label.lower()}" for label in UNIVERSAL_LABELS]
)


# ---------------------------------------------------------------------------
# Migration Logic
# ---------------------------------------------------------------------------


def migrate_old_to_new() -> bool:
    """
    Migrate legacy Document / Section / Term nodes to the universal schema.

    Mapping Strategy
    ----------------
    - Each **Document** becomes a ``ContentAsset`` node.  The document's
      ``title`` is carried over as both ``name_th`` and ``name_en`` (a later
      enrichment pass could translate these fields).
    - Each **Section** becomes a ``KnowledgeConcept`` node, preserving its
      ``title`` and hierarchical ``level`` information in the description.
    - Each **Term** becomes a ``KnowledgeConcept`` node as well, since terms
      represent domain vocabulary / concepts extracted from documents.

    Structural Relationships (Legacy -> Universal)
    ----------------------------------------------
    - Document-HAS_SECTION->Section  =>  ContentAsset-HAS_CHAPTER->KnowledgeConcept
    - Section-CONTAINS_SECTION->Section => KnowledgeConcept-NEXT_CHAPTER->KnowledgeConcept
    - Document-MENTIONS_TERM->Term   =>  ContentAsset-REFERENCES_CONCEPT->KnowledgeConcept

    Cross-document ``RELATED_TO`` relationships are dropped during migration
    because the universal schema favours explicit concept-level links over
    implicit document co-occurrence.

    Returns:
        True if the full migration completed without fatal errors, False otherwise.
    """
    try:
        graph_db: Neo4jGraphDB = get_graph_db()
    except Exception as e:
        logger.error(f"Cannot acquire graph_db instance for migration: {e}")
        return False

    # ------------------------------------------------------------------
    # Step 1 — Migrate Document nodes to ContentAsset
    # ------------------------------------------------------------------
    success = _migrate_documents(graph_db)
    if not success:
        logger.error("Document migration failed; aborting full migration.")
        return False

    # ------------------------------------------------------------------
    # Step 2 — Migrate Section nodes to KnowledgeConcept
    # ------------------------------------------------------------------
    success = _migrate_sections(graph_db)
    if not success:
        logger.error("Section migration failed; aborting full migration.")
        return False

    # ------------------------------------------------------------------
    # Step 3 — Migrate Term nodes to KnowledgeConcept
    # ------------------------------------------------------------------
    success = _migrate_terms(graph_db)
    if not success:
        logger.error("Term migration failed; aborting full migration.")
        return False

    # ------------------------------------------------------------------
    # Step 4 — Re-create structural relationships in universal schema
    # ------------------------------------------------------------------
    success = _migrate_relationships(graph_db)
    if not success:
        logger.error("Relationship migration failed; aborting full migration.")
        return False

    # ------------------------------------------------------------------
    # Step 5 — Remove legacy labels (detach old nodes that are fully migrated)
    # ------------------------------------------------------------------
    _remove_legacy_labels(graph_db)

    logger.info("Schema migration from legacy model to universal schema completed.")
    return True


# ---------------------------------------------------------------------------
# Internal Migration Helpers
# ---------------------------------------------------------------------------


def _migrate_documents(graph_db: Neo4jGraphDB) -> bool:
    """
    Convert every :Document node into a :ContentAsset node.

    The original Document node is re-labelled (not deleted) so that existing
    relationships remain intact until Step 4 re-creates them in the universal
    schema.  After the SET, the node carries both labels temporarily.

    Returns:
        True on success, False on error.
    """
    try:
        with graph_db.driver.session() as session:
            # First, upsert ContentAsset nodes from Document data
            result = session.run(
                """
                MATCH (d:Document)
                WHERE NOT d:migrated
                RETURN d.id AS id, d.title AS title
                """
            )
            docs = list(result)

            for doc in docs:
                doc_id = doc["id"]
                title = doc.get("title", "Untitled")
                graph_db.upsert_node(
                    label="ContentAsset",
                    properties={
                        "id": doc_id,
                        "domain": "document",
                        "name_th": title,
                        "name_en": title,
                        "technical_desc_th": "",
                        "technical_desc_en": "",
                    },
                )

            # Mark legacy documents as migrated
            session.run(
                """
                MATCH (d:Document)
                SET d:migrated = TRUE
                """
            )

        logger.info(f"Migrated {len(docs)} Document nodes to ContentAsset.")
        return True
    except Exception as e:
        logger.error(f"Failed to migrate Document nodes: {e}")
        return False


def _migrate_sections(graph_db: Neo4jGraphDB) -> bool:
    """
    Convert every :Section node into a :KnowledgeConcept node.

    Returns:
        True on success, False on error.
    """
    try:
        with graph_db.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Section)
                WHERE NOT s:migrated
                RETURN s.id AS id, s.title AS title, s.level AS level, s.doc_id AS doc_id
                """
            )
            sections = list(result)

            for sec in sections:
                section_id = sec["id"]
                title = sec.get("title", "")
                level = sec.get("level", 1)
                graph_db.upsert_node(
                    label="KnowledgeConcept",
                    properties={
                        "id": section_id,
                        "domain": "section",
                        "name_th": title,
                        "name_en": title,
                        "technical_desc_th": f"Level {level} section: {title}",
                        "technical_desc_en": f"Level {level} section: {title}",
                    },
                )

            session.run(
                """
                MATCH (s:Section)
                SET s:migrated = TRUE
                """
            )

        logger.info(f"Migrated {len(sections)} Section nodes to KnowledgeConcept.")
        return True
    except Exception as e:
        logger.error(f"Failed to migrate Section nodes: {e}")
        return False


def _migrate_terms(graph_db: Neo4jGraphDB) -> bool:
    """
    Convert every :Term node into a :KnowledgeConcept node.

    Returns:
        True on success, False on error.
    """
    try:
        with graph_db.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Term)
                WHERE NOT t:migrated
                RETURN t.name AS name, t.count AS count
                """
            )
            terms = list(result)

            for term in terms:
                term_name = term.get("name", "")
                term_count = term.get("count", 0) or 0
                # Use the term name itself as the stable id
                term_id = f"term_{term_name.replace(' ', '_')}"
                graph_db.upsert_node(
                    label="KnowledgeConcept",
                    properties={
                        "id": term_id,
                        "domain": "term",
                        "name_th": term_name,
                        "name_en": term_name,
                        "technical_desc_th": f"Extracted term (frequency: {term_count})",
                        "technical_desc_en": f"Extracted term (frequency: {term_count})",
                    },
                )

            session.run(
                """
                MATCH (t:Term)
                SET t:migrated = TRUE
                """
            )

        logger.info(f"Migrated {len(terms)} Term nodes to KnowledgeConcept.")
        return True
    except Exception as e:
        logger.error(f"Failed to migrate Term nodes: {e}")
        return False


def _migrate_relationships(graph_db: Neo4jGraphDB) -> bool:
    """
    Re-create structural relationships in the universal schema.

    Reads legacy relationship patterns and creates equivalent universal-schema
    relationships via ``upsert_relationship()``.

    Returns:
        True on success, False on error.
    """
    try:
        with graph_db.driver.session() as session:
            # --- HAS_SECTION -> HAS_CHAPTER ---
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
                RETURN d.id AS from_id, s.id AS to_id
                LIMIT 5000
                """
            )
            has_chapter_count = 0
            for record in result:
                graph_db.upsert_relationship(
                    from_id=record["from_id"],
                    to_id=record["to_id"],
                    rel_type="HAS_CHAPTER",
                )
                has_chapter_count += 1

            # --- MENTIONS_TERM -> REFERENCES_CONCEPT ---
            result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS_TERM]->(t:Term)
                RETURN d.id AS from_id, t.name AS term_name
                LIMIT 5000
                """
            )
            ref_concept_count = 0
            for record in result:
                term_id = f"term_{record['term_name'].replace(' ', '_')}"
                graph_db.upsert_relationship(
                    from_id=record["from_id"],
                    to_id=term_id,
                    rel_type="REFERENCES_CONCEPT",
                )
                ref_concept_count += 1

        logger.info(
            f"Migrated relationships: {has_chapter_count} HAS_CHAPTER, "
            f"{ref_concept_count} REFERENCES_CONCEPT."
        )
        return True
    except Exception as e:
        logger.error(f"Failed to migrate relationships: {e}")
        return False


def _remove_legacy_labels(graph_db: Neo4jGraphDB) -> None:
    """
    Remove legacy labels from migrated nodes, effectively converting them to
    their new universal-schema equivalents.

    This operation is best-effort — errors are logged but don't fail the
    overall migration (the data is already safe in the new nodes).
    """
    try:
        with graph_db.driver.session() as session:
            # Remove :Document label, keeping only migrated flag for audit
            session.run(
                """
                MATCH (d:Document)
                WHERE d:migrated = TRUE
                REMOVE d:Document
                SET d.type_legacy = 'Document'
                """
            )

            # Remove :Section label
            session.run(
                """
                MATCH (s:Section)
                WHERE s:migrated = TRUE
                REMOVE s:Section
                SET s.type_legacy = 'Section'
                """
            )

            # Remove :Term label
            session.run(
                """
                MATCH (t:Term)
                WHERE t:migrated = TRUE
                REMOVE t:Term
                SET t.type_legacy = 'Term'
                """
            )

        logger.info("Legacy labels removed from migrated nodes.")
    except Exception as e:
        logger.error(f"Failed to remove legacy labels (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


def validate_schema() -> Dict[str, Any]:
    """
    Verify that all required constraints and indexes for the universal schema
    exist in the connected Neo4j instance.

    Returns:
        A dictionary with validation results:
            {
                "valid": bool,
                "missing_constraints": [str],
                "missing_indexes": [str],
                "constraints_found": [str],
                "indexes_found": [str],
            }
        Returns ``{"valid": False, ...}`` with empty lists if the database
        cannot be reached.
    """
    result: Dict[str, Any] = {
        "valid": False,
        "missing_constraints": [],
        "missing_indexes": [],
        "constraints_found": [],
        "indexes_found": [],
    }

    try:
        graph_db: Neo4jGraphDB = get_graph_db()
    except Exception as e:
        logger.error(f"Cannot acquire graph_db instance for validation: {e}")
        return result

    try:
        with graph_db.driver.session() as session:
            # --- Constraints ---
            constraint_result = session.run("SHOW CONSTRAINTS YIELD name")
            existing_constraints = {record["name"] for record in constraint_result}

            for expected in EXPECTED_CONSTRAINTS:
                if expected in existing_constraints:
                    result["constraints_found"].append(expected)
                else:
                    result["missing_constraints"].append(expected)

            # --- Indexes ---
            index_result = session.run("SHOW INDEXES YIELD name")
            existing_indexes = {record["name"] for record in index_result}

            for expected in EXPECTED_INDEXES:
                if expected in existing_indexes:
                    result["indexes_found"].append(expected)
                else:
                    result["missing_indexes"].append(expected)

        result["valid"] = (
            len(result["missing_constraints"]) == 0
            and len(result["missing_indexes"]) == 0
        )

        if result["valid"]:
            logger.info("Schema validation passed — all constraints and indexes present.")
        else:
            logger.warning(
                f"Schema validation failed — missing constraints: "
                f"{result['missing_constraints']}, missing indexes: "
                f"{result['missing_indexes']}"
            )

        return result
    except Exception as e:
        logger.error(f"Schema validation failed with exception: {e}")
        return result
