# Technical Analysis Report: Universal GraphRAG & Cognitive Content Engine

**Date:** 2026-05-23  
**Author:** Senior Software Engineer / Technical Lead  
**Scope:** Full-gap analysis between `BusinessRequirement.md` and existing codebase  

---

## 1. Executive Summary

The current codebase implements a **PDF-to-Markdown pipeline with Hybrid Search (GraphDB + VectorDB)**. The business requirement calls for a significantly expanded system: **Universal GraphRAG & Cognitive Content Engine** covering multi-domain knowledge extraction, LLM-driven entity extraction, creative content generation, Open WebUI integration, and VRAM-aware sequential model management.

The gap between current state and target state is substantial but structured. This report breaks down the required work into phased sprints with clear dependencies, risk assessments, and design patterns.

---

## 2. Feature Breakdown by Priority

### P0 — Core Foundation (Must Have for MVP)

| # | Feature | Description | Current State | Gap |
|---|---------|-------------|---------------|-----|
| F1 | Universal Meta-Schema in Neo4j | Node labels: `KnowledgeConcept`, `Hardware`, `Strategy`, `Narrative`, `ContentAsset` with multilingual properties | Existing schema uses `Document`, `Section`, `Term`, `CodeBlock`, `URL` — structural only | **Complete redesign** of Cypher MERGE queries and index definitions |
| F2 | LLM-Based Entity Extraction Pipeline | Qwen 3.6 via `llama-server` extracts structured JSON with `layman_explanation`, `analogy`, `fiction_seed`, etc. | No LLM integration exists — only heuristic regex-based term extraction | **New module** required: `llm_extractor.py` |
| F3 | Dual-Storage Routing (Graph + Vector in Neo4j) | Store both graph relationships AND vector embeddings in Neo4j (using Neo4j Vector Index) | Currently split: Graph in Neo4j, Vector in Qdrant | **Architecture shift**: migrate vector storage from Qdrant to Neo4j native vector index |

### P1 — Search & Retrieval Enhancement

| # | Feature | Description | Current State | Gap |
|---|---------|-------------|---------------|-----|
| F4 | Hybrid Search with Graph Traversal | Vector search → Top-3 nodes → 1-2 hop graph expansion → Context assembly | `HybridSearchEngine.search()` exists but uses keyword-based graph search, not vector-first | **Enhance**: add vector-first routing + multi-hop traversal + context assembly |
| F5 | Embedding Model Migration to bge-m3 | Use `bge-m3` (multilingual) instead of `all-MiniLM-L6-v2` | Uses SentenceTransformer with configurable model | **Config change** + model download + vector dimension update (768 → 1024) |

### P2 — Integration & Interface

| # | Feature | Description | Current State | Gap |
|---|---------|-------------|---------------|-----|
| F6 | Open WebUI Custom Tool | `search_knowledge_graph` tool for function calling in Open WebUI | No Open WebUI integration | **New module**: `openwebui_tool.py` |
| F7 | YAML Frontmatter Injection | Add `domain` and `language` metadata to parsed Markdown files | Markdown output is raw from Marker | **Enhance** `main.py` post-processing step |

### P3 — Content Engine & VRAM Management

| # | Feature | Description | Current State | Gap |
|---|---------|-------------|---------------|-----|
| F8 | Sequential VRAM Manager | Load/unload Qwen → Omnivoice → ComfyUI sequentially on single RTX 3090 | No model lifecycle management | **New module**: `vram_manager.py` |
| F9 | Automated Content Pipeline | Script generation → Audio → Image with sequential model loading | Not implemented | **New module**: `content_pipeline.py` |

---

## 3. Impact Analysis on Existing Codebase

### 3.1 Files Requiring Major Refactoring

#### [`graph_db.py`](graph_db.py) — **HIGH IMPACT**

| Current Function | Required Change | Risk Level |
|-----------------|-----------------|------------|
| `create_nodes_and_relationships()` (line 322) | Replace structural schema (`Document`/`Section`/`Term`) with universal meta-schema (`KnowledgeConcept`/`Hardware`/`Strategy`/`Narrative`/`ContentAsset`). Must support backward-compatible coexistence during migration. | **HIGH** — Breaking change to existing graph data |
| `graph_rag_search()` (line 488) | Replace keyword-based `CONTAINS` search with vector-first approach: query → embedding → Neo4j vector index → top-3 nodes → multi-hop Cypher traversal | **MEDIUM** — New query patterns |
| `_extract_key_terms()` / `_extract_key_terms_th()` (line 215, 242) | Deprecate heuristic extraction; replace with LLM-driven structured JSON extraction via Qwen 3.6 | **LOW** — Can coexist during transition |
| `create_indexes()` (line 142) | Add indexes for new labels and vector index for embedding property | **MEDIUM** |

**New Functions Required:**
- `create_universal_schema()` — Cypher queries for new node labels + relationships
- `store_embedding_in_neo4j(node_id, embedding_vector)` — Update node with embedding list
- `vector_graph_hybrid_search(query, max_hops=2)` — Combined vector + graph traversal
- `assemble_context_block(nodes, relationships)` — Format `layman_explanation`, `analogy`, `fiction_seed` into Markdown

#### [`config.py`](config.py) — **MEDIUM IMPACT**

| Current Section | Required Change |
|-----------------|-----------------|
| `graph_db_config` (line 153) | Add `vector_index_name`, `embedding_dimension` (1024 for bge-m3), `llm_server_url`, `llm_temperature`, `llm_enable_thinking` |
| `vector_db_config` (line 171) | **Deprecate** Qdrant config in Phase 2; keep for backward compatibility during migration |
| New section needed | `llm_config`: server URL, model path, temperature (0.0), enable_thinking (False), max_tokens, timeout |
| New section needed | `vram_config`: sequential mode flag, model load/unload timeouts, VRAM threshold |

#### [`vector_db.py`](vector_db.py) — **HIGH IMPACT**

| Current Function | Required Change |
|-----------------|-----------------|
| Entire `QdrantVectorDB` class | **Phase 1**: Keep as-is for backward compatibility. **Phase 2**: Deprecate in favor of Neo4j native vector index. New class `Neo4jVectorStore` handles embedding storage/retrieval within Neo4j. |
| `create_embeddings()` (line 144) | Update model from `all-MiniLM-L6-v2` (768-dim) to `bge-m3` (1024-dim). Requires vector index recreation in Neo4j. |

#### [`hybrid_search.py`](hybrid_search.py) — **MEDIUM IMPACT**

| Current Function | Required Change |
|-----------------|-----------------|
| `search()` (line 47) | Restructure to: (1) Vector search via Neo4j vector index → top-3, (2) Graph traversal from those nodes → 1-2 hops, (3) Context assembly with `layman_explanation`, `analogy`, `fiction_seed` |
| `_combine_results()` (line 117) | Simplify — no longer need to merge Qdrant + Neo4j results. Single-source vector + graph from Neo4j. |

#### [`main.py`](main.py) — **LOW IMPACT**

| Current Function | Required Change |
|-----------------|-----------------|
| `_process_single_file()` (line 162) | Add YAML frontmatter injection step after Marker conversion: prepend `---\ndomain: auto_detected\nlanguage: th/en\n---` |
| `run_step_1()` (line 238) | After Markdown generation, trigger new LLM extraction pipeline for each chunk |

#### [`chunking.py`](chunking.py) — **LOW IMPACT**

- No structural changes needed. Current implementation supports character/semantic/thai_aware modes which align with the requirement for `MarkdownHeaderTextSplitter`-style chunking.
- Consider adding a `header_based` SplitMode variant that splits strictly at Markdown heading boundaries (as specified in requirement section 2).

### 3.2 New Files to Create

| File | Purpose | Dependencies |
|------|---------|-------------|
| `llm_extractor.py` | Qwen 3.6 integration for structured entity extraction via `llama-server` API | `config.py` (LLM config) |
| `neo4j_vector_store.py` | Neo4j native vector index operations (store/query embeddings) | `graph_db.py`, `config.py` |
| `openwebui_tool.py` | Open WebUI custom tool definition for `search_knowledge_graph` | `hybrid_search.py` |
| `vram_manager.py` | Sequential model load/unload manager for RTX 3090 (24GB) | `config.py` |
| `content_pipeline.py` | Automated content generation: script → audio → image | `vram_manager.py`, `llm_extractor.py` |
| `schema_migration.py` | Migration scripts for transitioning from old schema to universal meta-schema | `graph_db.py` |

### 3.3 Files to Keep Unchanged (Backward Compatibility)

| File | Reason |
|------|--------|
| [`microservice_v2.py`](microservice_v2.py) | PDF conversion microservice — still needed for Phase 1 ingestion |
| [`pdf_client.py`](pdf_client.py) | Test client for microservice |
| [`rate_limiter.py`](rate_limiter.py) | Rate limiting — still applicable |
| [`metrics.py`](metrics.py) | Prometheus metrics — still applicable |
| [`utils.py`](utils.py) | Shared utilities — no changes needed |
| [`parallel_processor.py`](parallel_processor.py) | Parallel processing — still useful for batch ingestion |

---

## 4. Risk Assessment & Technical Debt

### 4.1 Critical Risks

| Risk | Impact | Likelihood | Mitigation Strategy |
|------|--------|------------|---------------------|
| **VRAM Exhaustion** — Loading Qwen 3.6 (27B) + bge-m3 simultaneously on RTX 3090 (24GB) | System crash, OOM kills | HIGH | Implement strict sequential loading via `vram_manager.py` with CUDA memory tracking (`torch.cuda.memory_allocated()`) |
| **Schema Migration Data Loss** — Transitioning from structural to universal schema may lose existing graph relationships | Data integrity loss | MEDIUM | Use dual-write strategy during migration period: write to both old and new schemas, validate parity, then switch reads |
| **Qdrant → Neo4j Vector Migration** — Moving vector storage from Qdrant to Neo4j requires re-embedding all documents | Downtime, data inconsistency | MEDIUM | Parallel run: keep Qdrant as fallback during migration. Use feature flag `VECTOR_BACKEND=neo4j|qdrant` |
| **LLM JSON Extraction Failures** — Qwen 3.6 may produce malformed JSON despite `temperature=0.0` | Pipeline stalls, partial data | MEDIUM | Implement JSON validation with retry logic (max 3 attempts). Fallback to heuristic extraction on persistent failure |
| **bge-m3 Model Availability** — Multilingual embedding model may not be pre-downloaded | Startup failure | LOW | Add model download check at startup with progress indicator. Cache in `~/.cache/huggingface` |

### 4.2 Technical Debt Inventory

| Debt Item | Location | Severity | Resolution Plan |
|-----------|----------|----------|-----------------|
| Heuristic term extraction (`_extract_key_terms`) | [`graph_db.py:215`](graph_db.py:215) | MEDIUM | Deprecate after LLM extraction pipeline is stable. Keep as fallback. |
| Qdrant dependency | [`vector_db.py`](vector_db.py), [`config.py:171`](config.py:171) | LOW (Phase 2) | Phase 1: keep. Phase 2: add feature flag. Phase 3: remove after Neo4j vector migration complete. |
| No domain/language detection in Markdown output | [`main.py:220`](main.py:220) | MEDIUM | Add post-processing step with language detection (`langdetect` or heuristic Unicode range check) |
| Hardcoded embedding dimension (768) | [`config.py:175`](config.py:175) | HIGH | Make dynamic based on selected embedding model. Map: `all-MiniLM-L6-v2` → 768, `bge-m3` → 1024 |
| No transaction rollback in graph operations | [`graph_db.py:334`](graph_db.py:334) | MEDIUM | Wrap multi-step Cypher operations in Neo4j explicit transactions with rollback on failure |

---

## 5. Phased Implementation Plan

### Phase 1: Foundation & LLM Integration (Sprint 1-2, ~2 weeks)

**Goal:** Establish universal meta-schema and LLM extraction pipeline.

| Task | File(s) | Estimated Effort | Dependencies |
|------|---------|-----------------|--------------|
| 1.1 Define universal Cypher schema | `graph_db.py`, new `schema_migration.py` | 2 days | None |
| 1.2 Implement LLM client for llama-server | New `llm_extractor.py` | 3 days | `config.py` update |
| 1.3 Add LLM config section | [`config.py`](config.py) | 0.5 day | None |
| 1.4 Build extraction prompt templates | `llm_extractor.py` | 2 days | Task 1.2 |
| 1.5 Implement JSON validation + retry logic | `llm_extractor.py` | 1 day | Task 1.4 |
| 1.6 Add YAML frontmatter injection | [`main.py:220`](main.py:220) | 1 day | None |
| 1.7 Domain/language auto-detection | New utility in `utils.py` or `llm_extractor.py` | 1 day | Task 1.6 |

**Quality Gates:**
- [ ] All new Cypher queries use parameterized inputs (no string concatenation)
- [ ] LLM extraction produces valid JSON for 95%+ of test chunks
- [ ] Unit tests cover all 5 node label types with sample data
- [ ] Integration test: full pipeline from PDF → Markdown → LLM extraction → Neo4j storage

**Success Criteria:**
- Universal schema deployed in Neo4j with all required properties (`id`, `domain`, `name_th`/`name_en`, `technical_desc_th`/`technical_desc_en`, `layman_explanation`, `analogy`, `visual_concept`, `fiction_seed`, `embedding`)
- LLM extraction pipeline processes 10 test documents end-to-end with <5% JSON failure rate

---

### Phase 2: Vector Migration & Hybrid Search Enhancement (Sprint 3-4, ~2 weeks)

**Goal:** Migrate vector storage to Neo4j and implement vector-first hybrid search.

| Task | File(s) | Estimated Effort | Dependencies |
|------|---------|-----------------|--------------|
| 2.1 Implement `Neo4jVectorStore` class | New `neo4j_vector_store.py` | 3 days | Phase 1 complete |
| 2.2 Create Neo4j vector index (bge-m3, 1024-dim) | `neo4j_vector_store.py`, `config.py` | 1 day | Task 2.1 |
| 2.3 Migrate bge-m3 embedding model | [`vector_db.py`](vector_db.py), `config.py` | 1 day | Task 2.2 |
| 2.4 Implement dual-write (Qdrant + Neo4j) | `hybrid_search.py` | 2 days | Task 2.1, 2.3 |
| 2.5 Restructure hybrid search: vector-first → graph traversal | [`hybrid_search.py`](hybrid_search.py) | 3 days | Task 2.4 |
| 2.6 Implement context assembly (layman + analogy + fiction_seed) | `hybrid_search.py` or new module | 2 days | Phase 1 complete |
| 2.7 Feature flag for vector backend selection | `config.py` | 0.5 day | Task 2.4 |

**Quality Gates:**
- [ ] Vector search returns top-3 results within 500ms for test queries
- [ ] Graph traversal from vector results produces valid context blocks
- [ ] Feature flag `VECTOR_BACKEND=neo4j` works independently of Qdrant
- [ ] Load test: 100 concurrent search requests with <2s p95 latency

**Success Criteria:**
- Hybrid search returns enriched context including `layman_explanation`, `analogy`, and `fiction_seed` properties
- Neo4j vector index operational with bge-m3 embeddings (1024-dim)
- Qdrant remains functional as fallback via feature flag

---

### Phase 3: Open WebUI Integration (Sprint 5, ~1 week)

**Goal:** Enable GraphRAG search through Open WebUI chat interface.

| Task | File(s) | Estimated Effort | Dependencies |
|------|---------|-----------------|--------------|
| 3.1 Implement `search_knowledge_graph` tool | New `openwebui_tool.py` | 2 days | Phase 2 complete |
| 3.2 Package tool for Open WebUI Workspace | `openwebui_tool.py` + config | 1 day | Task 3.1 |
| 3.3 Test function calling flow: user query → Qwen → tool → Neo4j → response | Integration test | 2 days | Task 3.2 |

**Quality Gates:**
- [ ] Tool correctly parses search queries and returns formatted context
- [ ] Open WebUI successfully invokes tool via function calling
- [ ] End-to-end test: Thai language query returns relevant creative content

**Success Criteria:**
- User can ask "ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย" in Open WebUI and receive a coherent fiction plot based on graph data

---

### Phase 4: VRAM Management & Content Pipeline (Sprint 6-7, ~2 weeks)

**Goal:** Sequential model management for automated content generation.

| Task | File(s) | Estimated Effort | Dependencies |
|------|---------|-----------------|--------------|
| 4.1 Implement `VRAMManager` class | New `vram_manager.py` | 3 days | None (independent) |
| 4.2 CUDA memory monitoring + auto-unload | `vram_manager.py` | 2 days | Task 4.1 |
| 4.3 Build content pipeline: script → audio → image | New `content_pipeline.py` | 3 days | Phase 1, Task 4.1 |
| 4.4 Omnivoice integration (audio generation) | `content_pipeline.py` | 2 days | Task 4.3 |
| 4.5 ComfyUI integration (image generation) | `content_pipeline.py` | 2 days | Task 4.3 |

**Quality Gates:**
- [ ] VRAM usage never exceeds 22GB (safety margin of 2GB on 24GB card)
- [ ] Sequential model loading: Qwen → unload → Omnivoice → unload → ComfyUI completes without OOM
- [ ] Pipeline processes one full content cycle (script + audio + image) within 10 minutes

**Success Criteria:**
- Automated content pipeline runs end-to-end on single RTX 3090 without manual intervention
- VRAM manager prevents concurrent model loading via lock mechanism

---

## 6. Dependency Graph

```
Phase 1 (Foundation)
├── Task 1.1: Universal Schema ──────────────────────────────┐
├── Task 1.2: LLM Client ────────────────────────────────────┤
├── Task 1.3: Config Update ─────────────────────────────────┤
├── Task 1.4: Prompt Templates ──────────────────────────────┤
├── Task 1.5: JSON Validation ───────────────────────────────┤
├── Task 1.6: YAML Frontmatter ──────────────────────────────┤
└── Task 1.7: Domain/Language Detection ─────────────────────┘
         │
         ▼
Phase 2 (Vector Migration)
├── Task 2.1: Neo4jVectorStore ◄──────── Phase 1 complete
├── Task 2.2: Vector Index (bge-m3)
├── Task 2.3: Embedding Model Migration
├── Task 2.4: Dual-Write Strategy
├── Task 2.5: Hybrid Search Restructure ◄─ Phase 1 + Task 2.4
├── Task 2.6: Context Assembly ◄───────── Phase 1 complete
└── Task 2.7: Feature Flag
         │
         ▼
Phase 3 (Open WebUI)
├── Task 3.1: search_knowledge_graph Tool ◄─ Phase 2 complete
├── Task 3.2: Open WebUI Packaging
└── Task 3.3: Integration Testing
         │
         ▼
Phase 4 (VRAM + Content Pipeline)
├── Task 4.1: VRAMManager ◄────────────── Independent
├── Task 4.2: CUDA Monitoring
├── Task 4.3: Content Pipeline ◄───────── Phase 1 + Task 4.1
├── Task 4.4: Omnivoice Integration
└── Task 4.5: ComfyUI Integration
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Module | Test File | Key Test Cases |
|--------|-----------|----------------|
| `llm_extractor.py` | `test_llm_extractor.py` (new) | JSON schema validation, retry logic, temperature=0.0 enforcement, enable_thinking=False enforcement |
| `neo4j_vector_store.py` | `test_neo4j_vector_store.py` (new) | Embedding storage/retrieval, vector index creation, similarity search accuracy |
| `graph_db.py` | Extend `test_graph_db.py` (new) | Universal schema node creation, relationship types (PREDICTS, APPLIES_TO, INSPIRES_PLOT, etc.), multilingual properties |
| `config.py` | Extend existing tests | LLM config validation, VRAM config defaults, embedding dimension mapping |
| `vram_manager.py` | `test_vram_manager.py` (new) | Sequential lock mechanism, CUDA memory tracking, model unload verification |

### 7.2 Integration Tests

| Test Scenario | Files Involved | Expected Outcome |
|--------------|----------------|------------------|
| PDF → Markdown → LLM Extraction → Neo4j | `main.py`, `llm_extractor.py`, `graph_db.py` | Document ingested with universal schema nodes and relationships |
| Hybrid Search: Vector → Graph Traversal → Context | `hybrid_search.py`, `neo4j_vector_store.py`, `graph_db.py` | Query returns enriched context with layman_explanation, analogy, fiction_seed |
| Open WebUI Tool Invocation | `openwebui_tool.py`, `hybrid_search.py` | Tool returns formatted Markdown context block |
| VRAM Sequential Loading | `vram_manager.py`, `content_pipeline.py` | Models load/unload sequentially without exceeding 22GB VRAM |

### 7.3 End-to-End Tests

| Scenario | Flow | Success Criteria |
|----------|------|------------------|
| Full Ingestion Pipeline | Upload PDF → Marker parse → Chunk → LLM extract → Neo4j store (graph + vector) | All nodes created with required properties; vector index populated |
| GraphRAG Query via Open WebUI | User types Thai query → Qwen calls tool → Hybrid search → Response with creative content | Response includes relevant concepts with layman explanations and fiction seeds |
| Automated Content Generation | Trigger pipeline → Qwen generates script → Omnivoice generates audio → ComfyUI generates image | Complete content package produced within time budget; VRAM never exceeds limit |

### 7.4 Performance Benchmarks

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Vector search latency (p95) | < 500ms | Load test with 100 concurrent queries |
| Graph traversal (2-hop) latency | < 300ms | Timed Cypher execution |
| LLM extraction throughput | > 5 chunks/minute | Measured over 100-chunk batch |
| Full hybrid search round-trip | < 2s | End-to-end timing from query to context assembly |
| VRAM peak usage during pipeline | < 22GB | `torch.cuda.max_memory_allocated()` |

---

## 8. Proposed Design Patterns

### 8.1 Strategy Pattern for Vector Backend

```python
# Allows switching between Qdrant and Neo4j vector storage at runtime
class VectorBackend(Protocol):
    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool: ...
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]: ...

class QdrantBackend(VectorBackend): ...   # Existing implementation
class Neo4jVectorBackend(VectorBackend): ...  # New implementation

# Factory based on config flag
def get_vector_backend() -> VectorBackend:
    backend = os.getenv("VECTOR_BACKEND", "qdrant")
    return Neo4jVectorBackend() if backend == "neo4j" else QdrantBackend()
```

**Benefit:** Zero-downtime migration from Qdrant to Neo4j vector storage. Both backends coexist during transition period.

### 8.2 Pipeline Pattern for Ingestion

```python
# Each stage is independent, testable, and composable
class IngestionPipeline:
    def __init__(self):
        self.stages = [
            ParseStage(),          # Marker PDF → Markdown + YAML frontmatter
            ChunkStage(),          # MarkdownHeaderTextSplitter chunking
            DomainDetectStage(),   # Auto-detect domain and language
            LLMExtractStage(),     # Qwen 3.6 structured extraction
            GraphStoreStage(),     # Neo4j MERGE for nodes + relationships
            VectorStoreStage(),    # Embedding storage (Neo4j or Qdrant)
        ]

    def process(self, file_path: str) -> PipelineResult:
        context = PipelineContext(input_file=file_path)
        for stage in self.stages:
            context = stage.execute(context)
            if not context.success:
                self._handle_failure(stage, context)
        return context.result
```

**Benefit:** Each stage can be tested independently. Failures at any stage are isolated with retry/skip logic. New stages (e.g., quality scoring) can be inserted without modifying existing code.

### 8.3 State Machine for VRAM Management

```python
# Ensures only one model is loaded at a time
class VRAMState(Enum):
    IDLE = "idle"
    QWEN_LOADED = "qwen_loaded"
    OMNIVOICE_LOADED = "omnivoice_loaded"
    COMFYUI_LOADED = "comfyui_loaded"

class VRAMManager:
    _state: VRAMState = VRAMState.IDLE
    _lock: asyncio.Lock = asyncio.Lock()

    async def load_model(self, model_type: ModelType) -> ModelHandle:
        async with self._lock:
            if self._state != VRAMState.IDLE:
                raise VRAMConflictError(f"Cannot load {model_type}: {self._state} active")
            model = await self._load(model_type)
            self._state = f"{model_type}_LOADED"
            return model

    async def unload_model(self) -> None:
        async with self._lock:
            self._cleanup_cuda_cache()
            self._state = VRAMState.IDLE
```

**Benefit:** Prevents concurrent model loading through asyncio lock. State transitions are explicit and testable. CUDA cache cleanup ensures clean state between models.

### 8.4 Repository Pattern for Graph Operations

```python
# Abstracts Neo4j operations behind interface for testability
class KnowledgeGraphRepository(Protocol):
    def upsert_concept(self, concept: KnowledgeConcept) -> str: ...
    def search_by_vector(self, embedding: List[float], top_k: int) -> List[str]: ...
    def traverse_from(self, node_ids: List[str], max_hops: int) -> GraphTraversalResult: ...
    def assemble_context(self, node_ids: List[str]) -> str: ...

class Neo4jKnowledgeGraphRepository(KnowledgeGraphRepository): ...
```

**Benefit:** Enables mocking in unit tests. Allows swapping Neo4j for alternative graph databases without changing business logic.

---

## 9. Backward Compatibility Strategy

### 9.1 Schema Coexistence

During migration, both old and new schemas must coexist in Neo4j:

```cypher
// Old schema (preserved during migration)
(:Document)-[:HAS_SECTION]->(:Section)
(:Document)-[:MENTIONS_TERM]->(:Term)

// New schema (gradually populated)
(:KnowledgeConcept {domain: "finance"})-[:PREDICTS]->(:Strategy)
(:Hardware)-[:REQUIRES_INFRA]->(:KnowledgeConcept)
```

**Migration Approach:**
1. **Phase 1:** Deploy new schema alongside old. New documents use new schema; existing documents remain in old schema.
2. **Phase 2:** Run background migration job to convert old `Document`/`Section`/`Term` nodes to universal meta-schema equivalents.
3. **Phase 3:** Validate parity between old and new representations. Drop old schema indexes and labels after confirmation.

### 9.2 API Compatibility

- Keep existing `graph_db.graph_rag_search()` signature but route internally to new hybrid search when `VECTOR_BACKEND=neo4j`.
- Maintain Qdrant client initialization as optional (graceful degradation when `QDRANT_URI` is unset).
- Preserve all existing environment variable names. Add new ones with clear defaults.

---

## 10. Coding Standards & Conventions

All new code must follow these standards consistent with the existing codebase:

| Standard | Rule | Enforcement |
|----------|------|-------------|
| **Type Hints** | All function signatures must include full type annotations | `mypy --strict` in CI |
| **Docstrings** | Google-style docstrings with Args/Returns/Raises sections | Existing pattern in [`graph_db.py:168`](graph_db.py:168) |
| **Cypher Safety** | All Cypher queries must use parameterized inputs (`$param`) — never string concatenation | Code review + existing pattern in [`graph_db.py:374`](graph_db.py:374) |
| **Error Handling** | Log errors with `logger.error()`, return False/empty on failure — never raise unhandled exceptions | Existing pattern throughout codebase |
| **Lazy Initialization** | Database clients must use lazy init (singleton pattern with global variable) | Existing pattern in [`graph_db.py:583`](graph_db.py:583) |
| **Environment Config** | All configurable values via `os.getenv()` with safe defaults | Existing pattern in [`config.py:39`](config.py:39) |
| **UTF-8 Safety** | All file I/O must specify `encoding="utf-8"` | Existing pattern in [`main.py:220`](main.py:220) |

---

## 11. Environment Variables Summary

### New Variables Required

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_SERVER_URL` | `http://localhost:8080` | llama-server endpoint for Qwen 3.6 |
| `LLM_MODEL` | `Qwen3.6-27B-MTP-GGUF` | Model identifier for extraction |
| `LLM_TEMPERATURE` | `0.0` | Temperature for deterministic JSON output |
| `LLM_ENABLE_THINKING` | `false` | Disable chain-of-thought for structured output |
| `LLM_MAX_TOKENS` | `4096` | Maximum tokens per extraction request |
| `LLM_TIMEOUT` | `120` | Request timeout in seconds |
| `VECTOR_BACKEND` | `qdrant` | Switch between `qdrant` and `neo4j` vector storage |
| `EMBEDDING_DIMENSION` | `1024` | Vector dimension (auto-detected from model if unset) |
| `VRAM_SEQUENTIAL_MODE` | `true` | Enable strict sequential model loading |
| `VRAM_SAFE_LIMIT_GB` | `22` | VRAM usage safety threshold (GB) |
| `DOMAIN_DETECTION_MODE` | `auto` | Domain detection: `auto`, `manual`, or `llm` |

---

## 12. Sprint Planning Recommendation

### Sprint 1 (Week 1-2): Foundation
- **Capacity:** ~15 developer days
- **Deliverables:** Universal schema, LLM extraction pipeline, YAML frontmatter, domain detection
- **Definition of Done:** 10 test PDFs processed end-to-end with universal schema nodes in Neo4j

### Sprint 2 (Week 3-4): Vector Migration
- **Capacity:** ~15 developer days
- **Deliverables:** Neo4j vector store, bge-m3 migration, enhanced hybrid search, context assembly
- **Definition of Done:** Hybrid search returns enriched context; feature flag controls backend

### Sprint 3 (Week 5): Open WebUI Integration
- **Capacity:** ~5 developer days
- **Deliverables:** `search_knowledge_graph` tool, Open WebUI packaging, integration tests
- **Definition of Done:** Thai language query in Open WebUI returns creative content from graph

### Sprint 4 (Week 6-7): VRAM & Content Pipeline
- **Capacity:** ~10 developer days
- **Deliverables:** VRAM manager, sequential model loading, automated content pipeline
- **Definition of Done:** Full content cycle (script → audio → image) runs on single RTX 3090

---

## 13. Conclusion

The gap between the current codebase and the business requirement is significant but well-structured for phased delivery. The existing foundation — particularly the hybrid search architecture, chunking module, and microservice layer — provides a solid base that can be extended rather than replaced.

**Key architectural decisions:**
1. **Neo4j as single source of truth** for both graph and vector data eliminates cross-database synchronization complexity
2. **Strategy pattern** for vector backend enables zero-downtime migration from Qdrant
3. **Pipeline pattern** for ingestion ensures each stage is independently testable and composable
4. **State machine** for VRAM management prevents the most critical failure mode (OOM on RTX 3090)

**Estimated total effort:** ~45 developer days across 4 sprints (7 weeks).  
**Critical path:** LLM extraction pipeline (Phase 1) → Vector migration (Phase 2) → All downstream features depend on these two foundations.
