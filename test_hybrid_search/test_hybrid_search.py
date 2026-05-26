"""
Test script for vector-first hybrid search functionality.
Tests the three-stage pipeline: Vector Search → Graph Traversal → Context Assembly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the configuration
try:
    from config import get_config, is_config_ready
    print("✓ Configuration module loaded successfully")

    if is_config_ready():
        cfg = get_config().get_config()
        print("✓ Configuration loaded:")
        print(f"  GraphDB URI: {cfg['graph_db']['neo4j_uri']}")
        print(f"  Vector Backend: {os.environ.get('VECTOR_BACKEND', 'qdrant')}")
    else:
        print("⚠ Configuration not ready (missing env vars), skipping config test")
except Exception as e:
    print(f"✗ Configuration error: {e}")

# Test Neo4j vector store (lazy-init - won't crash if DB is down)
try:
    from neo4j_vector_store import get_vector_backend
    print("✓ Neo4j Vector Store module loaded successfully (lazy-init)")
except Exception as e:
    print(f"✗ Neo4j Vector Store error: {e}")

# Test hybrid search (lazy-init - won't crash if DBs are down)
try:
    from hybrid_search import get_hybrid_search
    print("✓ Hybrid search module loaded successfully (lazy-init)")

    engine = get_hybrid_search()
    print("✓ HybridSearchEngine instance created")

    # Verify new pipeline methods exist
    assert hasattr(engine, "search"), "Missing search() method"
    assert hasattr(engine, "_graph_traversal"), "Missing _graph_traversal() method"
    assert hasattr(engine, "_assemble_context"), "Missing _assemble_context() method"
    assert hasattr(engine, "_combine_results"), "Missing _combine_results() method"
    print("✓ All pipeline methods verified")

    # Test _combine_results with sample data
    sample_results = [
        {"id": "node-1", "score": 0.95, "name_en": "Alpha"},
        {"id": "node-2", "score": 0.87, "name_en": "Beta"},
        {"id": "node-1", "score": 0.70, "name_en": "Alpha-dup"},  # duplicate
    ]
    combined = engine._combine_results(sample_results)
    assert len(combined) == 2, f"Expected 2 unique nodes, got {len(combined)}"
    assert combined[0]["id"] == "node-1", "Highest-score node should be first"
    assert combined[0]["score"] == 0.95, "Should keep highest score for duplicates"
    print("✓ _combine_results deduplication and ranking verified")

    # Test _assemble_context with sample data
    vector_res = [
        {
            "id": "concept-1",
            "label": "KnowledgeConcept",
            "name_th": "แนวคิดเอلفา",
            "name_en": "Alpha Concept",
            "score": 0.92,
            "layman_explanation": "A simple idea about patterns.",
            "analogy": "Like finding shapes in clouds.",
            "fiction_seed": "In a world where thoughts are visible...",
        }
    ]
    expanded = [
        {
            "id": "concept-2",
            "label": "Strategy",
            "name_th": "กลยุทธ์เบต้า",
            "name_en": "Beta Strategy",
            "score": 0.0,
            "hops": 1,
            "layman_explanation": "A method to achieve goals.",
            "analogy": "Like a chess opening.",
            "fiction_seed": "The strategist smiled knowingly...",
        }
    ]
    context = engine._assemble_context(vector_res, expanded)
    assert "Search Results" in context, "Missing header"
    assert "Alpha Concept" in context, "Missing vector result node"
    assert "Beta Strategy" in context, "Missing expanded node"
    assert "---" in context, "Missing separator"
    print("✓ _assemble_context Markdown formatting verified")

    # Test empty-result handling
    empty_context = engine._assemble_context([], [])
    assert empty_context == "", "Empty results should return empty string"
    print("✓ Empty result handling verified")

    print("\n✓ Hybrid search engine ready with vector-first pipeline")
except Exception as e:
    print(f"✗ Hybrid search error: {e}")
    import traceback
    traceback.print_exc()

print("\nHybrid search modules loaded successfully!")
