# Action Plan: Universal GraphRAG & Cognitive Content Engine

**Generated:** 2026-05-23  
**Source Documents:** [`brief.md`](brief.md), [`TECHNICAL_ANALYSIS_REPORT.md`](TECHNICAL_ANALYSIS_REPORT.md), [`BusinessRequirement.md`](BusinessRequirement.md)  
**Total Actions:** 22 actions across 4 phases  
**Estimated Effort:** ~45 developer days (7 weeks)  

---

## Table of Contents

| Phase | Sprint | Focus | Actions |
|-------|--------|-------|---------|
| Phase 1 | Week 1-2 | Foundation & LLM Integration | A001 – A007 |
| Phase 2 | Week 3-4 | Vector Migration & Hybrid Search | A008 – A014 |
| Phase 3 | Week 5 | Open WebUI Integration | A015 – A017 |
| Phase 4 | Week 6-7 | VRAM Management & Content Pipeline | A018 – A022 |

---

## PHASE 1: Foundation & LLM Integration (Sprint 1-2)

**Goal:** Establish universal meta-schema in Neo4j and build the LLM extraction pipeline using Qwen 3.6 via llama-server.

---

### ACTION A001 — Define Universal Cypher Schema in Neo4j

| Field | Detail |
|-------|--------|
| **Action Name** | `define_universal_schema` |
| **File(s)** | [`graph_db.py`](graph_db.py), new file `schema_migration.py` |
| **Effort** | 2 days |
| **Dependencies** | None |

**Description:**  
Redesign the Neo4j schema from the current structural model (`Document` / `Section` / `Term`) to a universal meta-schema supporting multi-domain knowledge. Implement Cypher queries for node creation, relationship definition, and index configuration.

**Expected Result:**  
- New node labels: `KnowledgeConcept`, `Hardware`, `Strategy`, `Narrative`, `ContentAsset`
- Each node carries properties: `id`, `domain`, `name_th`, `name_en`, `technical_desc_th`, `technical_desc_en`, `layman_explanation`, `analogy`, `visual_concept`, `fiction_seed`, `embedding`
- Relationships defined: `PREDICTS`, `APPLIES_TO`, `CORRELATES_WITH`, `REQUIRES_INFRA`, `INSPIRES_PLOT`, `IS_EXAMPLE_OF`, `CONTRADICTS`, `HAS_CHAPTER`, `NEXT_CHAPTER`, `REFERENCES_CONCEPT`
- Indexes created for all new labels and properties used in search queries

**Prompt:**
```
You are a Neo4j database engineer. Redesign the graph schema in graph_db.py from the current structural model (Document/Section/Term) to a universal meta-schema.

CREATE the following node labels with these properties:
- KnowledgeConcept {id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed, embedding}
- Hardware {id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed, embedding}
- Strategy {id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed, embedding}
- Narrative {id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed, embedding}
- ContentAsset {id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed, embedding}

CREATE the following relationship types:
- Logical: PREDICTS, APPLIES_TO, CORRELATES_WITH, REQUIRES_INFRA
- Creative: INSPIRES_PLOT, IS_EXAMPLE_OF, CONTRADICTS
- Structural: HAS_CHAPTER, NEXT_CHAPTER, REFERENCES_CONCEPT

Implement these functions in graph_db.py:
1. create_universal_schema() — Create constraints and indexes for all new labels
2. upsert_node(label, properties) — MERGE node by id, SET properties
3. upsert_relationship(from_id, to_id, rel_type, properties) — MERGE relationship

Also create schema_migration.py with:
1. migrate_old_to_new() — Background migration converting Document/Section/Term nodes to universal schema equivalents
2. validate_schema() — Verify all constraints and indexes exist

Coding Standards:
- All Cypher queries MUST use parameterized inputs ($param) — never string concatenation
- Full type hints on all function signatures
- Google-style docstrings with Args/Returns/Raises sections
- Log errors with logger.error(), return False on failure — never raise unhandled exceptions
- Neo4j connection: host=10.10.1.210, port=7474/7687, NEO4J_AUTH=neo4j/214356ppKK#
```

---

### ACTION A002 — Implement LLM Client for llama-server (Qwen 3.6)

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_llm_client` |
| **File(s)** | New file `llm_extractor.py`, [`config.py`](config.py) |
| **Effort** | 3 days |
| **Dependencies** | Task A003 (Config Update) must be completed first |

**Description:**  
Build an LLM client module that communicates with llama-server running Qwen 3.6 27B via the OpenAI-compatible API protocol. The client handles structured JSON extraction requests with strict parameter enforcement.

**Expected Result:**  
- `LLMClient` class with methods: `extract_entities(chunk_text)`, `detect_domain(text)`, `generate_analogy(concept)`
- Enforces `temperature=0.0` and `enable_thinking=False` on every request
- Supports retry logic with exponential backoff (max 3 attempts)
- Returns structured JSON matching the universal meta-schema properties

**Prompt:**
```
You are a Python backend engineer. Create a new file llm_extractor.py implementing an LLM client for Qwen 3.6 running via llama-server.

Requirements:
1. Class LLMClient with lazy initialization (singleton pattern)
2. Connect to llama-server at URL from config (LLM_SERVER_URL, default http://localhost:8080)
3. Use OpenAI-compatible API protocol (POST /v1/chat/completions)
4. Enforce these parameters on EVERY request:
   - temperature=0.0 (deterministic output)
   - enable_thinking=False (no chain-of-thought)
   - max_tokens from config (default 4096)
   - timeout from config (default 120 seconds)

Implement these methods:
1. extract_entities(chunk_text: str) -> Dict[str, Any]
   - Sends chunk to Qwen with system prompt requesting structured JSON extraction
   - Returns dict with keys: concepts[], hardware[], strategies[], narratives[], relationships[]
   - Each entity must include: id, domain, name_th, name_en, technical_desc_th, technical_desc_en, layman_explanation, analogy, visual_concept, fiction_seed

2. detect_domain(text: str) -> str
   - Determines domain category (finance, hardware, engineering, fiction, etc.)
   - Returns domain string

3. generate_analogy(concept: str, context: str) -> str
   - Generates a relatable analogy for a technical concept

4. _validate_json(response_text: str) -> Optional[Dict]
   - Validates returned JSON structure
   - Returns parsed dict or None if invalid

5. _retry_with_backoff(func, *args, max_retries=3, **kwargs)
   - Retry decorator with exponential backoff

Error Handling:
- Log all errors with logger.error()
- Return empty structures on persistent failure (never raise unhandled exceptions)
- Implement JSON validation — if Qwen returns malformed JSON, retry up to 3 times
- Fallback to heuristic extraction from graph_db.py._extract_key_terms() on persistent failure

Config values used:
- LLM_SERVER_URL (default: http://localhost:8080)
- LLM_MODEL (default: Qwen3.6-27B-MTP-GGUF)
- LLM_TEMPERATURE (default: 0.0)
- LLM_ENABLE_THINKING (default: false)
- LLM_MAX_TOKENS (default: 4096)
- LLM_TIMEOUT (default: 120)
```

---

### ACTION A003 — Add LLM and VRAM Configuration Sections

| Field | Detail |
|-------|--------|
| **Action Name** | `add_llm_vram_config` |
| **File(s)** | [`config.py`](config.py) |
| **Effort** | 0.5 day |
| **Dependencies** | None |

**Description:**  
Extend the existing configuration module with new sections for LLM server settings, VRAM management parameters, and embedding dimension mapping. All values must be environment-variable driven with safe defaults.

**Expected Result:**  
- New `llm_config` dictionary with keys: `server_url`, `model`, `temperature`, `enable_thinking`, `max_tokens`, `timeout`
- New `vram_config` dictionary with keys: `sequential_mode`, `safe_limit_gb`, `model_load_timeout`, `model_unload_timeout`
- Updated `embedding_dimension` mapping: `all-MiniLM-L6-v2` → 768, `bge-m3` → 1024
- New environment variable: `VECTOR_BACKEND` (default: `qdrant`)
- Backward compatible — all existing config sections remain unchanged

**Prompt:**
```
You are a Python configuration engineer. Extend config.py with new configuration sections.

Add these new config dictionaries (read from environment variables with defaults):

1. llm_config = {
    "server_url": os.getenv("LLM_SERVER_URL", "http://localhost:8080"),
    "model": os.getenv("LLM_MODEL", "Qwen3.6-27B-MTP-GGUF"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    "enable_thinking": os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
    "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
}

2. vram_config = {
    "sequential_mode": os.getenv("VRAM_SEQUENTIAL_MODE", "true").lower() == "true",
    "safe_limit_gb": int(os.getenv("VRAM_SAFE_LIMIT_GB", "22")),
    "model_load_timeout": int(os.getenv("VRAM_LOAD_TIMEOUT", "300")),
    "model_unload_timeout": int(os.getenv("VRAM_UNLOAD_TIMEOUT", "60")),
}

3. Update embedding dimension mapping to be dynamic:
   EMBEDDING_DIMENSIONS = {
       "all-MiniLM-L6-v2": 768,
       "bge-m3": 1024,
       "nomic-embed-text-v1.5": 768,
   }
   embedding_dimension = EMBEDDING_DIMENSIONS.get(embedding_model, 1024)

4. Add VECTOR_BACKEND:
   vector_backend = os.getenv("VECTOR_BACKEND", "qdrant")  # "qdrant" or "neo4j"

5. Add DOMAIN_DETECTION_MODE:
   domain_detection_mode = os.getenv("DOMAIN_DETECTION_MODE", "auto")  # "auto", "manual", "llm"

Rules:
- Do NOT modify existing config sections — only add new ones
- All values must use os.getenv() with safe defaults
- Maintain existing code style and structure
- Add comments explaining each new variable
```

---

### ACTION A004 — Build LLM Extraction Prompt Templates

| Field | Detail |
|-------|--------|
| **Action Name** | `build_prompt_templates` |
| **File(s)** | [`llm_extractor.py`](llm_extractor.py) |
| **Effort** | 2 days |
| **Dependencies** | Task A002 (LLM Client) must be completed first |

**Description:**  
Design and implement system prompt templates for the LLM extraction pipeline. These prompts instruct Qwen 3.6 to extract structured knowledge entities from text chunks, producing JSON output conforming to the universal meta-schema.

**Expected Result:**  
- System prompt template for entity extraction with explicit JSON schema definition
- System prompt template for domain detection
- System prompt template for analogy generation
- All prompts support multilingual output (Thai + English)
- Prompts enforce `layman_explanation`, `analogy`, `visual_concept`, and `fiction_seed` generation

**Prompt:**
```
You are a prompt engineering specialist. Add prompt templates to llm_extractor.py for structured knowledge extraction.

Create these prompt templates as module-level constants:

1. EXTRACTION_SYSTEM_PROMPT — Main entity extraction prompt:
   - Instruct Qwen to analyze input text and extract knowledge entities
   - Output MUST be valid JSON matching this schema:
     {
       "concepts": [{"id":"", "domain":"", "name_th":"", "name_en":"", "technical_desc_th":"", "technical_desc_en":"", "layman_explanation":"", "analogy":"", "visual_concept":"", "fiction_seed":""}],
       "hardware": [...same structure...],
       "strategies": [...same structure...],
       "narratives": [...same structure...],
       "relationships": [{"from_id":"", "to_id":"", "type":"PREDICTS|APPLIES_TO|CORRELATES_WITH|REQUIRES_INFRA|INSPIRES_PLOT|IS_EXAMPLE_OF|CONTRADICTS|HAS_CHAPTER|NEXT_CHAPTER|REFERENCES_CONCEPT"}]
     }
   - Emphasize: layman_explanation must be simple language for general audience
   - analogy must compare technical concept to everyday objects/situations
   - fiction_seed must be a creative plot idea inspired by the concept
   - visual_concept must describe an image that could illustrate the concept

2. DOMAIN_DETECTION_PROMPT — Domain classification prompt:
   - Instruct Qwen to classify text into domain: finance, hardware, engineering, fiction, general
   - Return single JSON: {"domain": "finance"}

3. ANALOGY_GENERATION_PROMPT — Standalone analogy generation:
   - Input: technical concept + context
   - Output: relatable analogy in Thai and English

4. User prompt template for extraction:
   EXTRACT_USER_TEMPLATE = "Analyze this text and extract all knowledge entities:\n\n{text}\n\nReturn ONLY valid JSON."

Requirements:
- Prompts must be in English (for Qwen system prompt) but request bilingual output (Thai + English)
- Include explicit JSON schema in the system prompt to reduce malformed output
- Add Thai-language examples in few-shot format if possible
- Temperature is enforced at 0.0 externally — do not mention temperature in prompts
```

---

### ACTION A005 — Implement JSON Validation and Retry Logic

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_json_validation` |
| **File(s)** | [`llm_extractor.py`](llm_extractor.py) |
| **Effort** | 1 day |
| **Dependencies** | Task A004 (Prompt Templates) must be completed first |

**Description:**  
Implement robust JSON validation for LLM responses with automatic retry logic. Handle edge cases where Qwen 3.6 produces partial JSON, markdown-wrapped JSON, or completely malformed output.

**Expected Result:**  
- `_validate_json()` method that parses and validates LLM response against expected schema
- Automatic cleaning of markdown code blocks (```json ... ```)
- Retry mechanism with exponential backoff (max 3 attempts)
- Fallback to heuristic extraction from [`graph_db.py:215`](graph_db.py:215) on persistent failure
- Metrics tracking for JSON success/failure rate

**Prompt:**
```
You are a Python engineer implementing robust JSON validation. Add these functions to llm_extractor.py:

1. _validate_json(response_text: str) -> Optional[Dict[str, Any]]
   - Strip markdown code blocks (```json ... ```) if present
   - Attempt json.loads() on cleaned text
   - Validate required keys exist: "concepts", "relationships" at minimum
   - Validate each entity has required fields: id, domain, name_th, name_en
   - If any field missing, attempt to fill with defaults (empty string for text, "unknown" for domain)
   - Return validated dict or None if completely unparseable

2. _clean_llm_response(raw_text: str) -> str
   - Remove markdown formatting
   - Handle partial JSON by finding last complete JSON object
   - Fix common issues: trailing commas, unescaped quotes
   - Return cleaned text ready for json.loads()

3. _retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0, **kwargs) -> Any
   - Call func with kwargs
   - If returns None or raises exception, retry with exponential backoff
   - Delay sequence: 1s, 2s, 4s (max 3 retries)
   - Log each retry attempt with logger.warning()
   - After all retries fail, log error and call fallback_heuristic_extraction()

4. fallback_heuristic_extraction(chunk_text: str) -> Dict[str, Any]
   - Import and call graph_db._extract_key_terms() as fallback
   - Convert heuristic results to universal schema format
   - Fill layman_explanation, analogy, fiction_seed with placeholder text
   - Log warning that fallback was used

5. Add metrics counter:
   - json_success_count: incremented on successful JSON parse
   - json_failure_count: incremented on failed JSON parse
   - fallback_used_count: incremented when heuristic fallback is triggered

Error Handling:
- Never raise unhandled exceptions
- Log all failures with logger.error() including the raw response (truncated to 500 chars)
- Return empty structures {} on total failure
```

---

### ACTION A006 — Add YAML Frontmatter Injection

| Field | Detail |
|-------|--------|
| **Action Name** | `add_yaml_frontmatter` |
| **File(s)** | [`main.py`](main.py) |
| **Effort** | 1 day |
| **Dependencies** | None (independent task) |

**Description:**  
Enhance the Markdown output pipeline to inject YAML frontmatter at the top of each generated Markdown file. The frontmatter includes `domain` and `language` metadata auto-detected from content.

**Expected Result:**  
- Every Markdown file produced by Marker parsing has YAML frontmatter prepended
- Format: `---\ndomain: <auto_detected>\nlanguage: <th|en>\n---\n`
- Domain detection uses heuristic analysis (keyword matching) or LLM-based detection
- Language detection uses Unicode range analysis (Thai character detection)

**Prompt:**
````
You are a Python engineer. Modify main.py to add YAML frontmatter injection after Marker PDF-to-Markdown conversion.

Location: Modify _process_single_file() function around line 162 in main.py.

Implementation:
1. After Marker converts PDF to Markdown, before saving the file:
   - Detect language by analyzing character ranges:
     * If Thai Unicode range (U+0E00-U+0E7F) present → language = "th"
     * Otherwise → language = "en"
   - Detect domain using keyword matching:
     * Finance keywords: trading, strategy, alpha, quant, portfolio, risk
     * Hardware keywords: server, IC, latency, FPGA, network
     * Engineering keywords: architecture, system, design, pattern
     * Default: "general"

2. Prepend YAML frontmatter to Markdown content:
   
   ---
   ```cypher
   domain: <detected_domain>
   language: <detected_language>
   source_pdf: <original_filename>
   processed_date: <ISO 8601 timestamp>
   ```
   ---
   <original markdown content>
   

3. Create helper function in utils.py:
   def inject_frontmatter(content: str, domain: str, language: str, source: str) -> str
   def detect_language(text: str) -> str  # Returns "th" or "en"
   def detect_domain(text: str) -> str    # Returns domain category string

Requirements:
- Check if frontmatter already exists before adding (avoid double-injection)
- Use encoding="utf-8" for all file I/O
- Log the detected domain and language for each file
- If DOMAIN_DETECTION_MODE config = "llm", use LLMClient.detect_domain() instead of heuristic
````

---

### ACTION A007 — Implement Domain/Language Auto-Detection Utility

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_domain_detection` |
| **File(s)** | [`utils.py`](utils.py) or [`llm_extractor.py`](llm_extractor.py) |
| **Effort** | 1 day |
| **Dependencies** | Task A006 (YAML Frontmatter) should be started first |

**Description:**  
Build a robust domain and language detection utility supporting three modes: heuristic keyword matching, Unicode range analysis, and LLM-based classification. This utility serves both the frontmatter injection and the LLM extraction pipeline.

**Expected Result:**  
- `detect_language(text)` — Returns "th" or "en" based on character analysis
- `detect_domain(text, mode="auto")` — Returns domain string using configured detection mode
- Supports three modes: `auto` (heuristic + confidence check), `manual` (user-specified), `llm` (Qwen-based)
- Confidence scoring for heuristic detection with LLM fallback when confidence is low

**Prompt:**
```
You are a Python engineer. Add domain and language detection utilities to utils.py.

Implement these functions:

1. detect_language(text: str) -> Tuple[str, float]
   - Analyze Unicode character ranges:
     * Thai: U+0E00-U+0E7F
     * Latin: U+0041-U+007A
     * Mixed: both present
   - Return (language_code, confidence_score)
   - language_code: "th", "en", or "mixed"
   - confidence_score: 0.0 to 1.0

2. detect_domain(text: str, mode: str = "auto") -> Tuple[str, float]
   - Mode "auto": Use keyword matching with confidence scoring
   - Mode "manual": Return domain from optional parameter
   - Mode "llm": Call LLMClient.detect_domain() 
   
   Domain keywords dictionary:
   DOMAIN_KEYWORDS = {
       "finance": ["trading", "strategy", "alpha", "quant", "portfolio", "risk", "market", "stock", "bond", "derivative"],
       "hardware": ["server", "IC", "latency", "FPGA", "network", "CPU", "GPU", "memory", "cache"],
       "engineering": ["architecture", "system", "design", "pattern", "algorithm", "optimization"],
       "fiction": ["character", "plot", "story", "narrative", "chapter", "scene"],
       "general": [],  # Default fallback
   }
   
   - Count keyword matches per domain
   - Return domain with highest score and normalized confidence
   - If confidence < 0.3 and mode != "manual", optionally escalate to LLM

3. detect_domain_and_language(text: str, config: Dict) -> Dict[str, Any]
   - Combined detection function
   - Returns: {"domain": "...", "language": "...", "domain_confidence": 0.XX, "language_confidence": 0.XX}
   - Reads DOMAIN_DETECTION_MODE from config

Requirements:
- Full type hints
- Google-style docstrings
- Handle empty input gracefully (return defaults)
- Log detection results at DEBUG level
```

---

## PHASE 2: Vector Migration & Hybrid Search Enhancement (Sprint 3-4)

**Goal:** Migrate vector storage from Qdrant to Neo4j native vector index and implement vector-first hybrid search with graph traversal.

---

### ACTION A008 — Implement Neo4jVectorStore Class

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_neo4j_vector_store` |
| **File(s)** | New file `neo4j_vector_store.py`, [`config.py`](config.py) |
| **Effort** | 3 days |
| **Dependencies** | Phase 1 complete (A001-A007) |

**Description:**  
Create a new vector storage backend that uses Neo4j's native vector index capabilities instead of Qdrant. This class stores embedding vectors as node properties and performs similarity search via Neo4j's `node.similarity()` function.

**Expected Result:**  
- `Neo4jVectorStore` class implementing the `VectorBackend` protocol
- Methods: `create_vector_index()`, `store_embedding()`, `search()`, `delete_embedding()`
- Uses Strategy Pattern for runtime switching between Qdrant and Neo4j backends
- Supports bge-m3 embeddings (1024-dim) with cosine similarity

**Prompt:**
```
You are a Python engineer implementing a Neo4j-based vector store. Create neo4j_vector_store.py.

Implement the VectorBackend protocol:

from typing import Protocol, List, Dict, Any

class VectorBackend(Protocol):
    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool: ...
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]: ...
    def create_index(self) -> bool: ...
    def delete_embedding(self, node_id: str) -> bool: ...

Class Neo4jVectorStore(VectorBackend):
    
    def __init__(self, neo4j_uri: str, username: str, password: str, 
                 index_name: str = "knowledge_vector", dimension: int = 1024):
        # Lazy initialization with singleton pattern
        # Connect to Neo4j at 10.10.1.210:7687 (Bolt protocol)
        
    def create_index(self) -> bool:
        # Create Neo4j vector index:
        # CREATE VECTOR INDEX knowledge_vector IF NOT EXISTS
        # FOR (n:KnowledgeConcept) ON n.embedding
        # OPTIONS {indexConfig: {
        #   `vector.dimensions`: 1024,
        #   `vector.similarity_function`: 'cosine'
        # }}
        # Also create for Hardware, Strategy, Narrative, ContentAsset labels
        
    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool:
        # MATCH (n) WHERE n.id = $node_id
        # SET n.embedding = $embedding, n.embedded_text = $text
        # Return True on success
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Generate embedding for query using configured embedding model
        # 2. CALL db.index.vector.queryNodes("knowledge_vector", $top_k, $query_embedding)
        #    YIELD node, score
        # 3. Return list of dicts with: id, label, name_th, name_en, score, properties
        
    def delete_embedding(self, node_id: str) -> bool:
        # MATCH (n) WHERE n.id = $node_id
        # REMOVE n.embedding, n.embedded_text

Factory function:
def get_vector_backend() -> VectorBackend:
    backend = os.getenv("VECTOR_BACKEND", "qdrant")
    if backend == "neo4j":
        return Neo4jVectorStore(...)
    else:
        return QdrantBackend(...)  # Existing implementation from vector_db.py

Requirements:
- Use Neo4j Bolt protocol (port 7687) for binary communication
- Lazy initialization — connect only when first method called
- Full type hints and Google-style docstrings
- Log all operations at INFO level, errors at ERROR level
- Support both 768-dim (all-MiniLM) and 1024-dim (bge-m3) embeddings
```

---

### ACTION A009 — Create Neo4j Vector Index for bge-m3 (1024-dim)

| Field | Detail |
|-------|--------|
| **Action Name** | `create_neo4j_vector_index` |
| **File(s)** | [`neo4j_vector_store.py`](neo4j_vector_store.py), [`config.py`](config.py) |
| **Effort** | 1 day |
| **Dependencies** | Task A008 (Neo4jVectorStore) must be completed first |

**Description:**  
Configure and create the Neo4j vector index optimized for bge-m3 embeddings (1024 dimensions). Set up cosine similarity as the default similarity function and ensure the index covers all universal schema node labels.

**Expected Result:**  
- Vector index `knowledge_vector` created in Neo4j
- Covers labels: `KnowledgeConcept`, `Hardware`, `Strategy`, `Narrative`, `ContentAsset`
- Uses cosine similarity function
- Configured for 1024-dimensional vectors (bge-m3)
- Index creation is idempotent (safe to run multiple times)

**Prompt:**
````
You are a Neo4j database engineer. Extend the create_index() method in Neo4jVectorStore class.

Implement vector index creation:

1. Create composite vector index covering all universal schema labels:
   ```cypher
   CREATE VECTOR INDEX knowledge_vector IF NOT EXISTS
   FOR (n:KnowledgeConcept|Hardware|Strategy|Narrative|ContentAsset)
   ON n.embedding
   OPTIONS {indexConfig: {
     `vector.dimensions`: $dimensions,
     `vector.similarity_function`: 'cosine'
   }}
   ```

2. Parameters come from config:
   - dimensions: from EMBEDDING_DIMENSIONS[embedding_model] (1024 for bge-m3)
   - index_name: from config vector_index_name (default "knowledge_vector")

3. Implementation details:
   - Use parameterized Cypher ($dimensions)
   - Check if index exists before creating (SHOW VECTOR INDEXES)
   - Handle Neo4j version compatibility (vector indexes require Neo4j 5+)
   - Log index creation status

4. Add method show_indexes() -> List[Dict]:
   - RUN: SHOW INDEXES YIELD name, type, labelsOrTypes, properties
   - Return list of current indexes for verification

5. Add method drop_index(index_name: str) -> bool:
   - DROP INDEX $index_name IF EXISTS
   - Use only for maintenance/migration

Requirements:
- Idempotent operations — safe to run multiple times
- Full error handling with logging
- Verify index creation by querying SHOW INDEXES after creation
````

---

### ACTION A010 — Migrate Embedding Model from all-MiniLM to bge-m3

| Field | Detail |
|-------|--------|
| **Action Name** | `migrate_embedding_model` |
| **File(s)** | [`vector_db.py`](vector_db.py), [`config.py`](config.py) |
| **Effort** | 1 day |
| **Dependencies** | Task A009 (Vector Index) must be completed first |

**Description:**  
Replace the current SentenceTransformer model (`all-MiniLM-L6-v2`, 768-dim) with `bge-m3` (1024-dim, multilingual). Update embedding generation code and ensure backward compatibility during transition.

**Expected Result:**  
- Embedding model configurable via environment variable
- Supports both `all-MiniLM-L6-v2` (768-dim) and `bge-m3` (1024-dim)
- Model auto-downloaded from HuggingFace if not cached
- Startup validation checks model availability

**Prompt:**
```
You are a Python ML engineer. Update vector_db.py to support bge-m3 embedding model.

Changes to create_embeddings() method (around line 144):

1. Make embedding model configurable:
   embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
   
2. Support multiple models:
   SUPPORTED_MODELS = {
       "all-MiniLM-L6-v2": {"dim": 768, "source": "sentence_transformers"},
       "BAAI/bge-m3": {"dim": 1024, "source": "sentence_transformers"},
       "nomic-embed-text-v1.5": {"dim": 768, "source": "external_api"},
   }

3. Model loading with validation:
   def load_embedding_model(model_name: str) -> Any:
       # Check if model exists in cache (~/.cache/huggingface)
       # If not, download with progress indicator
       # Load model and verify dimension matches expected
       # Return model instance
       
4. Add startup check:
   def validate_embedding_model() -> bool:
       # Verify model is available and loadable
       # Log model name, dimension, and cache status
       # Return True if valid, False otherwise

5. External embedding API support for nomic-embed-text-v1.5:
   - If EMBEDDING_MODEL = "nomic-embed-text-v1.5"
   - Use external API at http://192.168.1.212:7700/v1 (OpenAI-compatible)
   - POST /v1/embeddings with input text
   - Return embedding vector

Requirements:
- Backward compatible — default can still be all-MiniLM-L6-v2
- Log model loading time and dimension
- Handle download failures gracefully
- Cache model in ~/.cache/huggingface
```

---

### ACTION A011 — Implement Dual-Write Strategy (Qdrant + Neo4j)

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_dual_write` |
| **File(s)** | [`hybrid_search.py`](hybrid_search.py), [`neo4j_vector_store.py`](neo4j_vector_store.py) |
| **Effort** | 2 days |
| **Dependencies** | Tasks A008, A010 must be completed first |

**Description:**  
Implement a dual-write strategy where new embeddings are written to both Qdrant and Neo4j simultaneously. This ensures zero data loss during migration and allows seamless switching between backends via feature flag.

**Expected Result:**  
- All new embeddings stored in both Qdrant and Neo4j
- Feature flag `VECTOR_BACKEND` controls read path (which backend serves queries)
- Write path always targets both backends during migration period
- Parity validation tool to compare results between backends

**Prompt:**
```
You are a Python engineer implementing dual-write vector storage.

Create a DualWriteVectorStore wrapper class:

class DualWriteVectorStore:
    """Writes to both Qdrant and Neo4j, reads from configured backend."""
    
    def __init__(self):
        self.qdrant_backend = QdrantBackend(...)  # Existing
        self.neo4j_backend = Neo4jVectorStore(...)  # New
        self.read_backend = os.getenv("VECTOR_BACKEND", "qdrant")
        
    def store_embedding(self, node_id: str, text: str, embedding: List[float]) -> bool:
        # Write to BOTH backends
        qdrant_success = self.qdrant_backend.store_embedding(node_id, text, embedding)
        neo4j_success = self.neo4j_backend.store_embedding(node_id, text, embedding)
        
        if qdrant_success and neo4j_success:
            logger.info(f"Dual-write successful for {node_id}")
            return True
        elif qdrant_success or neo4j_success:
            logger.warning(f"Partial write for {node_id}: qdrant={qdrant_success}, neo4j={neo4j_success}")
            return True  # Partial success acceptable
        else:
            logger.error(f"Dual-write failed for {node_id}")
            return False
            
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Read from configured backend only
        if self.read_backend == "neo4j":
            return self.neo4j_backend.search(query, top_k)
        else:
            return self.qdrant_backend.search(query, top_k)

Add parity validation function:
def validate_backend_parity(test_queries: List[str]) -> Dict[str, Any]:
    """Compare search results between Qdrant and Neo4j for same queries."""
    results = {}
    for query in test_queries:
        qdrant_results = self.qdrant_backend.search(query)
        neo4j_results = self.neo4j_backend.search(query)
        # Compare top-k overlap, score differences
        results[query] = {
            "qdrant_top_ids": [r["id"] for r in qdrant_results],
            "neo4j_top_ids": [r["id"] for r in neo4j_results],
            "overlap_count": len(set intersection),
        }
    return results

Requirements:
- Never block writes because one backend is down
- Log all dual-write operations
- Parity validation should be run as periodic job or manual test
```

---

### ACTION A012 — Restructure Hybrid Search: Vector-First with Graph Traversal

| Field | Detail |
|-------|--------|
| **Action Name** | `restructure_hybrid_search` |
| **File(s)** | [`hybrid_search.py`](hybrid_search.py), [`graph_db.py`](graph_db.py) |
| **Effort** | 3 days |
| **Dependencies** | Task A011 (Dual-Write) must be completed first |

**Description:**  
Restructure the hybrid search pipeline to follow a vector-first approach: (1) Vector similarity search returns top-3 nodes, (2) Graph traversal expands from those nodes 1-2 hops, (3) Context assembly formats results with `layman_explanation`, `analogy`, and `fiction_seed`.

**Expected Result:**  
- New search flow: Vector Search → Top-3 Nodes → Multi-hop Graph Traversal → Context Assembly
- Replaces existing keyword-based graph search in [`hybrid_search.py:47`](hybrid_search.py:47)
- Simplifies `_combine_results()` since both vector and graph data come from Neo4j

**Prompt:**
```
You are a search engine engineer. Restructure hybrid_search.py to implement vector-first hybrid search.

Replace the current search() method with this new pipeline:

def search(self, query: str, top_k: int = 3, max_hops: int = 2) -> Dict[str, Any]:
    """
    Vector-first hybrid search pipeline.
    
    Pipeline:
    1. Vector Search — Embed query → Neo4j vector index → Top-K nodes
    2. Graph Traversal — Expand from top-K nodes via relationships (max_hops)
    3. Context Assembly — Format results with layman_explanation, analogy, fiction_seed
    """
    
    # Step 1: Vector Search
    vector_results = self.vector_backend.search(query, top_k=top_k)
    seed_node_ids = [r["id"] for r in vector_results]
    
    # Step 2: Graph Traversal (multi-hop expansion)
    expanded_nodes = self._graph_traversal(seed_node_ids, max_hops=max_hops)
    
    # Step 3: Context Assembly
    context = self._assemble_context(vector_results, expanded_nodes)
    
    return {
        "query": query,
        "vector_results": vector_results,
        "expanded_nodes": expanded_nodes,
        "context_block": context,
        "total_nodes": len(set(all node IDs)),
    }

def _graph_traversal(self, seed_node_ids: List[str], max_hops: int = 2) -> List[Dict]:
    """Expand from seed nodes via graph relationships."""
    # Cypher for multi-hop traversal:
    # MATCH path = (start)-[r*1..$max_hops]-(neighbor)
    # WHERE start.id IN $seed_ids
    # RETURN path, length(path) as hops
    # LIMIT 50
    
    # Return expanded nodes with relationship metadata

def _assemble_context(self, vector_results: List[Dict], expanded_nodes: List[Dict]) -> str:
    """Format search results into Markdown context block."""
    # Combine all unique nodes
    # For each node, extract: name_th/name_en, layman_explanation, analogy, fiction_seed
    # Format as structured Markdown:
    # ## Concept: {name}
    # **Simple Explanation:** {layman_explanation}
    # **Analogy:** {analogy}
    # **Creative Seed:** {fiction_seed}
    # ---
    
    return markdown_context

Update _combine_results():
- No longer need to merge Qdrant + Neo4j results
- Single source: Neo4j provides both vector and graph data
- Simplify to deduplication + ranking logic only

Requirements:
- Parameterized Cypher queries only ($param)
- Handle empty results gracefully at each pipeline stage
- Log timing for each pipeline stage (vector, traversal, assembly)
- Full type hints and docstrings
```

---

### ACTION A013 — Implement Context Assembly with Creative Properties

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_context_assembly` |
| **File(s)** | [`hybrid_search.py`](hybrid_search.py) or new module `context_assembler.py` |
| **Effort** | 2 days |
| **Dependencies** | Phase 1 complete (universal schema with creative properties) |

**Description:**  
Build a context assembly module that formats search results into rich Markdown blocks containing `layman_explanation`, `analogy`, `fiction_seed`, and other creative properties. This formatted context is consumed by the LLM for response generation.

**Expected Result:**  
- `ContextAssembler` class with method `assemble(nodes, relationships)`
- Outputs structured Markdown with sections for each concept
- Includes relationship descriptions between concepts
- Supports both Thai and English output based on query language

**Prompt:**
```
You are a Python engineer. Create context_assembler.py (or add to hybrid_search.py).

Class ContextAssembler:

def assemble(self, nodes: List[Dict], relationships: List[Dict], 
             language: str = "th", include_creative: bool = True) -> str:
    """
    Assemble search results into structured Markdown context block.
    
    Args:
        nodes: List of node dicts from hybrid search
        relationships: List of relationship dicts  
        language: Output language ("th" or "en")
        include_creative: Whether to include analogy, fiction_seed, visual_concept
        
    Returns:
        Formatted Markdown string
    """
    
    sections = []
    
    # Group nodes by label type
    concepts = [n for n in nodes if n["label"] == "KnowledgeConcept"]
    hardware = [n for n in nodes if n["label"] == "Hardware"]
    strategies = [n for n in nodes if n["label"] == "Strategy"]
    narratives = [n for n in nodes if n["label"] == "Narrative"]
    
    # Format each section
    for concept in concepts:
        section = format_concept_block(concept, language, include_creative)
        sections.append(section)
        
    # Add relationship descriptions
    if relationships:
        rel_section = format_relationships(relationships, language)
        sections.append(rel_section)
        
    return "\n\n".join(sections)

def format_concept_block(concept: Dict, language: str, include_creative: bool) -> str:
    """Format single concept as Markdown block."""
    name_key = "name_th" if language == "th" else "name_en"
    desc_key = "technical_desc_th" if language == "th" else "technical_desc_en"
    
    block = f"## {concept[name_key]}\n"
    block += f"**Domain:** {concept['domain']}\n"
    block += f"**Description:** {concept[desc_key]}\n"
    block += f"**Simple Explanation:** {concept.get('layman_explanation', 'N/A')}\n"
    
    if include_creative:
        block += f"**Analogy:** {concept.get('analogy', 'N/A')}\n"
        block += f"**Creative Seed:** {concept.get('fiction_seed', 'N/A')}\n"
        block += f"**Visual Concept:** {concept.get('visual_concept', 'N/A')}\n"
        
    return block

def format_relationships(relationships: List[Dict], language: str) -> str:
    """Format relationships as Markdown section."""
    # Group by relationship type
    # Create readable descriptions
    
    return markdown_block

Requirements:
- Handle missing properties gracefully (use "N/A" placeholder)
- Support mixed-language output
- Deduplicate nodes before formatting
- Sort concepts by relevance score (if available)
```

---

### ACTION A014 — Implement Feature Flag for Vector Backend Selection

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_vector_backend_flag` |
| **File(s)** | [`config.py`](config.py), [`hybrid_search.py`](hybrid_search.py) |
| **Effort** | 0.5 day |
| **Dependencies** | Task A011 (Dual-Write) must be completed first |

**Description:**  
Implement a runtime feature flag allowing operators to switch the vector search backend between Qdrant and Neo4j without code changes or service restart. The flag controls the read path while dual-write ensures data parity.

**Expected Result:**  
- `VECTOR_BACKEND` environment variable controls active read backend
- Values: `qdrant` (default), `neo4j`, `dual` (compare both)
- Runtime switching via API endpoint or config reload
- Graceful degradation if selected backend is unavailable

**Prompt:**
```
You are a Python engineer. Implement feature flag for vector backend selection.

In config.py, add:
vector_backend = os.getenv("VECTOR_BACKEND", "qdrant")  # "qdrant" | "neo4j" | "dual"

In hybrid_search.py, update initialization:

class HybridSearchEngine:
    def __init__(self):
        self.backend_mode = config.vector_backend
        
        # Always initialize both backends for flexibility
        self.qdrant_backend = QdrantBackend(...)
        self.neo4j_backend = Neo4jVectorStore(...)
        
    def _get_active_backend(self) -> VectorBackend:
        """Return the active read backend based on feature flag."""
        if self.backend_mode == "neo4j":
            return self.neo4j_backend
        elif self.backend_mode == "dual":
            return DualCompareBackend(self.qdrant_backend, self.neo4j_backend)
        else:  # default: qdrant
            return self.qdrant_backend
            
    def switch_backend(self, backend: str) -> bool:
        """Runtime backend switching."""
        if backend in ("qdrant", "neo4j", "dual"):
            self.backend_mode = backend
            logger.info(f"Switched vector backend to: {backend}")
            return True
        return False

Add health check endpoint logic:
def check_backend_health(backend_name: str) -> Dict[str, Any]:
    """Verify backend connectivity and index status."""
    # Test connection, query index existence, return health status

Requirements:
- Feature flag change does not require service restart
- Log all backend switches
- Health check before switching (verify target backend is available)
- Default fallback to qdrant if neo4j unavailable
```

---

## PHASE 3: Open WebUI Integration (Sprint 5)

**Goal:** Enable GraphRAG search through Open WebUI chat interface via custom tool definition.

---

### ACTION A015 — Implement search_knowledge_graph Tool for Open WebUI

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_openwebui_tool` |
| **File(s)** | New file `openwebui_tool.py` |
| **Effort** | 2 days |
| **Dependencies** | Phase 2 complete (A008-A014) |

**Description:**  
Create an Open WebUI custom tool that exposes the hybrid search capability as a function-calling tool. This allows Qwen 3.6 in Open WebUI to invoke knowledge graph searches during conversations.

**Expected Result:**  
- `search_knowledge_graph` tool definition compatible with Open WebUI Workspace Tools
- Tool accepts natural language query, returns formatted Markdown context
- Handles Thai and English queries
- Includes error handling and timeout protection

**Prompt:**
```
You are a Python engineer. Create openwebui_tool.py implementing an Open WebUI custom tool.

Tool Definition:

class SearchKnowledgeGraphTool:
    """
    Open WebUI Custom Tool for Universal Knowledge Graph Search.
    
    Install in Open WebUI: Workspace -> Tools -> Add Custom Tool
    """
    
    name = "search_knowledge_graph"
    description = """
    Use this tool to search the Universal Knowledge Graph for concepts, equations, 
    hardware specifications, trading strategies, and creative content seeds.
    
    The graph contains multi-domain knowledge with:
    - Technical descriptions (Thai and English)
    - Simple explanations for general audience
    - Analogies comparing technical concepts to everyday things
    - Creative fiction seeds inspired by technical knowledge
    
    Input: Natural language search query (Thai or English)
    Output: Structured context block with relevant knowledge
    """
    
    parameters = {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "The search query in natural language (Thai or English)"
            }
        },
        "required": ["search_query"]
    }
    
    def __init__(self):
        self.search_engine = HybridSearchEngine()
        
    def execute(self, search_query: str) -> str:
        """
        Execute hybrid search and return formatted context.
        
        Args:
            search_query: Natural language query
            
        Returns:
            Formatted Markdown context block
        """
        try:
            # Run hybrid search pipeline
            results = self.search_engine.search(search_query, top_k=3, max_hops=2)
            
            # Format response
            response = f"# Search Results for: \"{search_query}\"\n\n"
            response += f"**Found {results['total_nodes']} related nodes**\n\n"
            response += results["context_block"]
            
            return response
            
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return f"# Search Error\n\nUnable to complete search: {str(e)}"

Requirements:
- Follow Open WebUI tool specification format
- Include JSON Schema for function calling
- Timeout protection (max 30 seconds per search)
- Handle Thai language queries properly (UTF-8)
- Return empty but valid response on error (never crash)
```

---

### ACTION A016 — Package Tool for Open WebUI Workspace

| Field | Detail |
|-------|--------|
| **Action Name** | `package_openwebui_tool` |
| **File(s)** | [`openwebui_tool.py`](openwebui_tool.py), deployment config |
| **Effort** | 1 day |
| **Dependencies** | Task A015 (Tool Implementation) must be completed first |

**Description:**  
Package the search_knowledge_graph tool for deployment in Open WebUI Workspace. Create installation configuration and documentation for deploying the tool as a custom pipeline or external tool.

**Expected Result:**  
- Tool packaged as Open WebUI-compatible module
- Installation instructions documented
- Configuration file for tool registration
- Environment variable mapping for Open WebUI deployment

**Prompt:**
```
You are a DevOps engineer. Package the search_knowledge_graph tool for Open WebUI deployment.

Create these files:

1. openwebui_tool_config.json — Tool registration configuration:
{
  "name": "search_knowledge_graph",
  "version": "1.0.0",
  "description": "Universal Knowledge Graph Search Tool",
  "endpoint": "http://localhost:8080/api/search",
  "parameters": {
    "type": "object",
    "properties": {
      "search_query": {"type": "string"}
    },
    "required": ["search_query"]
  }
}

2. openwebui_server.py — HTTP endpoint wrapper:
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/api/search")
async def search(request: SearchRequest):
    tool = SearchKnowledgeGraphTool()
    result = tool.execute(request.search_query)
    return {"result": result}

# Run with: uvicorn openwebui_server:app --host 0.0.0.0 --port 8081

3. INSTALLATION.md — Deployment instructions:
   - How to register tool in Open WebUI Workspace -> Tools
   - Environment variables needed
   - Testing the tool connection
   - Example queries in Thai and English

4. Dockerfile (optional) for containerized deployment:
   - Base image: python:3.12-slim
   - Install dependencies from requirements.txt
   - Expose port 8081
   - Health check endpoint

Requirements:
- Tool must be accessible via HTTP API for Open WebUI
- Include CORS headers if running on separate port
- Document all environment variables
- Include example curl commands for testing
```

---

### ACTION A017 — Test Open WebUI Function Calling Flow (E2E Integration)

| Field | Detail |
|-------|--------|
| **Action Name** | `test_openwebui_integration` |
| **File(s)** | New file `test_openwebui_integration.py` |
| **Effort** | 2 days |
| **Dependencies** | Task A016 (Tool Packaging) must be completed first |

**Description:**  
Perform end-to-end integration testing of the complete function calling flow: user query in Open WebUI → Qwen 3.6 decides to call tool → tool executes hybrid search → Neo4j returns results → Qwen generates response with creative content.

**Expected Result:**  
- Integration test script covering full function calling flow
- Test cases for Thai and English queries
- Verification that creative properties (fiction_seed, analogy) appear in responses
- Performance benchmarks for end-to-end latency

**Prompt:**
```
You are a QA engineer. Create test_openwebui_integration.py with E2E integration tests.

Test Cases:

1. test_thai_query_fiction_seed():
   - Query: "ช่วยคิดพล็อตนิยายจากสถาปัตยกรรม Low-latency C++ หน่อย"
   - Expected: Response includes fiction_seed content from graph
   - Verify: Tool was invoked, search returned results, response contains creative content

2. test_english_query_hardware():
   - Query: "What hardware is needed for high-frequency trading?"
   - Expected: Response includes Hardware nodes with REQUIRES_INFRA relationships
   - Verify: Technical descriptions present in response

3. test_tool_invocation_flow():
   - Simulate function calling request
   - Verify tool receives correct parameters
   - Verify tool returns properly formatted Markdown
   - Measure end-to-end latency (< 2 seconds target)

4. test_error_handling():
   - Send malformed query
   - Send empty query
   - Simulate Neo4j connection failure
   - Verify graceful error responses

5. test_thai_english_mixed():
   - Query mixing Thai and English
   - Verify proper language detection and response formatting

Test Infrastructure:
- Mock Open WebUI function calling interface
- Use test Neo4j database (or Docker container)
- Pre-load test data with known entities
- Measure timing for each pipeline stage

Requirements:
- pytest framework with fixtures
- Test data setup and teardown
- Assertions on response structure and content
- Performance assertions (latency thresholds)
- Log all test activity for debugging
```

---

## PHASE 4: VRAM Management & Content Pipeline (Sprint 6-7)

**Goal:** Implement sequential model management for automated content generation on a single RTX 3090 (24GB VRAM).

---

### ACTION A018 — Implement VRAMManager Class with State Machine

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_vram_manager` |
| **File(s)** | New file `vram_manager.py`, [`config.py`](config.py) |
| **Effort** | 3 days |
| **Dependencies** | None (independent task, but Phase 1 recommended for LLM integration) |

**Description:**  
Build a VRAM management module using a state machine pattern to ensure only one AI model is loaded at a time on the RTX 3090. Implements async locking, CUDA memory tracking, and automatic model unload between sequential operations.

**Expected Result:**  
- `VRAMManager` class with state machine: IDLE → QWEN_LOADED → OMNIVOICE_LOADED → COMFYUI_LOADED
- Async lock preventing concurrent model loading
- CUDA memory monitoring via `torch.cuda.memory_allocated()`
- Safety threshold enforcement (max 22GB on 24GB card)

**Prompt:**
```
You are a Python systems engineer. Create vram_manager.py implementing sequential VRAM management.

Imports:
import asyncio
import torch
from enum import Enum
from typing import Callable, Any, Optional

class VRAMState(Enum):
    IDLE = "idle"
    QWEN_LOADED = "qwen_loaded"
    OMNIVOICE_LOADED = "omnivoice_loaded"
    COMFYUI_LOADED = "comfyui_loaded"

class ModelType(Enum):
    QWEN = "qwen"
    OMNIVOICE = "omnivoice"
    COMFYUI = "comfyui"

class VRAMConflictError(Exception):
    pass

class VRAMManager:
    _instance = None
    _state: VRAMState = VRAMState.IDLE
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.safe_limit_gb = int(os.getenv("VRAM_SAFE_LIMIT_GB", "22"))
        self.sequential_mode = os.getenv("VRAM_SEQUENTIAL_MODE", "true").lower() == "true"
        
    async def load_model(self, model_type: ModelType, load_func: Callable) -> Any:
        """
        Load a model with VRAM safety checks.
        
        Args:
            model_type: Which model to load
            load_func: Async callable that loads the model
            
        Returns:
            Loaded model instance
            
        Raises:
            VRAMConflictError: If another model is already loaded
        """
        async with self._lock:
            if self._state != VRAMState.IDLE and self.sequential_mode:
                raise VRAMConflictError(
                    f"Cannot load {model_type.value}: {self._state.value} is active"
                )
                
            # Check available VRAM
            available_gb = self._get_available_vram()
            if available_gb < self.safe_limit_gb:
                logger.warning(f"Low VRAM: {available_gb}GB available, {self.safe_limit_gb}GB required")
                
            # Load model
            model = await load_func()
            
            # Update state
            self._state = VRAMState(f"{model_type.value}_LOADED")
            
            # Log VRAM usage
            used_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Loaded {model_type.value}: {used_gb:.1f}GB VRAM used")
            
            return model
            
    async def unload_model(self) -> None:
        """Unload current model and clean up CUDA cache."""
        async with self._lock:
            if self._state == VRAMState.IDLE:
                logger.warning("No model to unload")
                return
                
            model_name = self._state.value.replace("_LOADED", "")
            logger.info(f"Unloading {model_name}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            self._state = VRAMState.IDLE
            logger.info("Model unloaded, VRAM state: IDLE")
            
    def _get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_mem
        allocated = torch.cuda.memory_allocated()
        available = (total - allocated) / (1024**3)
        return available
        
    def get_state(self) -> VRAMState:
        return self._state
        
    def get_vram_stats(self) -> Dict[str, float]:
        """Return current VRAM statistics."""
        if not torch.cuda.is_available():
            return {"available_gb": 0, "used_gb": 0, "total_gb": 0}
            
        total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        used = torch.cuda.memory_allocated() / (1024**3)
        return {
            "total_gb": total,
            "used_gb": used,
            "available_gb": total - used,
            "state": self._state.value
        }

Requirements:
- Singleton pattern for global VRAM state
- Async lock prevents race conditions
- CUDA cache cleanup after unload
- Comprehensive logging of all state transitions
- Graceful handling when CUDA not available (CPU fallback)
```

---

### ACTION A019 — Implement CUDA Memory Monitoring and Auto-Unload

| Field | Detail |
|-------|--------|
| **Action Name** | `implement_cuda_monitoring` |
| **File(s)** | [`vram_manager.py`](vram_manager.py) |
| **Effort** | 2 days |
| **Dependencies** | Task A018 (VRAMManager) must be completed first |

**Description:**  
Add real-time CUDA memory monitoring with automatic model unloading when VRAM usage exceeds the safety threshold. Implement background monitoring thread and alerting mechanism.

**Expected Result:**  
- Background VRAM monitor checking usage at configurable interval
- Automatic unload trigger when usage exceeds `VRAM_SAFE_LIMIT_GB`
- Alert logging and optional webhook notification
- VRAM usage history tracking for diagnostics

**Prompt:**
```
You are a Python systems engineer. Extend vram_manager.py with CUDA monitoring.

Add these methods to VRAMManager:

1. async start_monitor(interval_seconds: int = 10, callback: Callable = None):
    """Start background VRAM monitoring."""
    self._monitoring = True
    self._monitor_interval = interval_seconds
    
    async def monitor_loop():
        while self._monitoring:
            stats = self.get_vram_stats()
            logger.debug(f"VRAM Monitor: {stats}")
            
            # Check threshold
            if stats["used_gb"] > self.safe_limit_gb:
                logger.warning(
                    f"VRAM threshold exceeded: {stats['used_gb']:.1f}GB > {self.safe_limit_gb}GB"
                )
                
                # Trigger auto-unload if in sequential mode
                if self.sequential_mode and self._state != VRAMState.IDLE:
                    logger.warning("Auto-unloading model due to VRAM threshold")
                    await self.unload_model()
                    
                # Call alert callback
                if callback:
                    await callback(stats, "threshold_exceeded")
            
            await asyncio.sleep(interval_seconds)
    
    self._monitor_task = asyncio.create_task(monitor_loop())

2. async stop_monitor():
    """Stop background VRAM monitoring."""
    self._monitoring = False
    if hasattr(self, '_monitor_task'):
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass

3. get_vram_history() -> List[Dict]:
    """Return VRAM usage history for diagnostics."""
    return self._vram_history  # List of timestamped stats

4. _record_vram_usage():
    """Record current VRAM usage to history."""
    import time
    self._vram_history.append({
        "timestamp": time.time(),
        "stats": self.get_vram_stats()
    })
    # Keep last 1000 entries
    if len(self._vram_history) > 1000:
        self._vram_history = self._vram_history[-1000:]

Add VRAM context manager for safe model usage:

class VRAMModelContext:
    """Async context manager for safe model loading/unloading."""
    
    def __init__(self, vram_manager: VRAMManager, model_type: ModelType, load_func: Callable):
        self.manager = vram_manager
        self.model_type = model_type
        self.load_func = load_func
        self.model = None
        
    async def __aenter__(self):
        self.model = await self.manager.load_model(self.model_type, self.load_func)
        return self.model
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.unload_model()
        self.model = None

Usage example:
async def generate_content():
    vram = VRAMManager()
    
    # Step 1: Load Qwen for script generation
    async with VRAMModelContext(vram, ModelType.QWEN, load_qwen) as qwen:
        script = qwen.generate(prompt)
        
    # Step 2: Load Omnivoice for audio generation  
    async with VRAMModelContext(vram, ModelType.OMNIVOICE, load_omnivoice) as tts:
        audio = tts.synthesize(script)
        
    # Step 3: Load ComfyUI for image generation
    async with VRAMModelContext(vram, ModelType.COMFYUI, load_comfyui) as generator:
        image = generator.create(prompt)

Requirements:
- Monitor runs as background task (non-blocking)
- Configurable check interval
- History retention limit (prevent memory leak)
- Clean shutdown on exit
```

---

### ACTION A020 — Build Content Pipeline: Script → Audio → Image

| Field | Detail |
|-------|--------|
| **Action Name** | `build_content_pipeline` |
| **File(s)** | New file `content_pipeline.py`, [`vram_manager.py`](vram_manager.py) |
| **Effort** | 3 days |
| **Dependencies** | Phase 1 complete (LLM extraction), Task A018 (VRAMManager) |

**Description:**  
Implement the automated content generation pipeline that sequentially loads and unloads models to produce a complete content package: script (Qwen) → audio (Omnivoice) → image (ComfyUI). Uses VRAMManager for safe model lifecycle management.

**Expected Result:**  
- `ContentPipeline` class orchestrating sequential model operations
- Pipeline stages: Script Generation → Audio Synthesis → Image Generation
- Each stage loads its model, performs work, and unloads before next stage
- Produces complete content package with all assets

**Prompt:**
```
You are a Python engineer. Create content_pipeline.py implementing the automated content pipeline.

Class ContentPipeline:

def __init__(self):
    self.vram_manager = VRAMManager()
    self.stages = [
        PipelineStage("script_generation", self._generate_script),
        PipelineStage("audio_synthesis", self._synthesize_audio),
        PipelineStage("image_generation", self._generate_image),
    ]

async def run(self, topic: str, language: str = "th") -> ContentPackage:
    """
    Execute full content pipeline: script → audio → image.
    
    Args:
        topic: Content topic/subject
        language: Output language ("th" or "en")
        
    Returns:
        ContentPackage with all generated assets
    """
    package = ContentPackage(topic=topic, language=language)
    
    for stage in self.stages:
        logger.info(f"Starting pipeline stage: {stage.name}")
        
        try:
            result = await stage.execute(package)
            package = result  # Each stage returns updated package
            
            if not package.success:
                logger.error(f"Pipeline failed at stage: {stage.name}")
                break
                
            logger.info(f"Stage {stage.name} completed successfully")
            
        except Exception as e:
            logger.error(f"Stage {stage.name} failed: {e}")
            package.success = False
            package.error = str(e)
            break
            
    return package

async def _generate_script(self, package: ContentPackage) -> ContentPackage:
    """Stage 1: Generate script using Qwen 3.6."""
    async with VRAMModelContext(
        self.vram_manager, ModelType.QWEN, load_qwen_model
    ) as qwen:
        # Use knowledge graph for context
        search_results = hybrid_search.search(package.topic)
        
        # Generate script with creative content from graph
        script = qwen.generate_script(
            topic=package.topic,
            context=search_results["context_block"],
            language=package.language
        )
        
        package.script = script
    return package

async def _synthesize_audio(self, package: ContentPackage) -> ContentPackage:
    """Stage 2: Synthesize audio using Omnivoice."""
    async with VRAMModelContext(
        self.vram_manager, ModelType.OMNIVOICE, load_omnivoice_model
    ) as omnivoice:
        audio_path = omnivoice.synthesize(
            text=package.script,
            language=package.language,
            output_dir=f"output/audio/"
        )
        
        package.audio_path = audio_path
    return package

async def _generate_image(self, package: ContentPackage) -> ContentPackage:
    """Stage 3: Generate image using ComfyUI."""
    async with VRAMModelContext(
        self.vram_manager, ModelType.COMFYUI, load_comfyui_api
    ) as comfyui:
        # Use visual_concept from knowledge graph as prompt
        image_prompt = package.get_visual_prompt() or f"Illustration of {package.topic}"
        
        image_path = comfyui.generate(
            prompt=image_prompt,
            output_dir=f"output/images/"
        )
        
        package.image_path = image_path
    return package

Data Classes:

@dataclass
class ContentPackage:
    topic: str
    language: str
    script: str = ""
    audio_path: str = ""
    image_path: str = ""
    success: bool = True
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_visual_prompt(self) -> Optional[str]:
        """Extract visual prompt from script or topic."""
        # Parse script for visual_concept references
        pass

@dataclass  
class PipelineStage:
    name: str
    execute_func: Callable

Requirements:
- Each stage is independent and testable
- Pipeline can resume from failed stage
- All file paths use absolute paths
- UTF-8 encoding for all text files
- Progress logging at each stage
```

---

### ACTION A021 — Integrate Omnivoice for Audio Generation

| Field | Detail |
|-------|--------|
| **Action Name** | `integrate_omnivoice` |
| **File(s)** | [`content_pipeline.py`](content_pipeline.py) |
| **Effort** | 2 days |
| **Dependencies** | Task A020 (Content Pipeline) must be completed first |

**Description:**  
Integrate Omnivoice TTS engine into the content pipeline. Handle model loading, text-to-speech synthesis, and audio file output with proper VRAM management.

**Expected Result:**  
- Omnivoice model loading function compatible with VRAMManager
- Text-to-speech synthesis supporting Thai and English
- Audio output in WAV/MP3 format
- Proper cleanup after synthesis

**Prompt:**
```
You are a Python engineer. Add Omnivoice TTS integration to content_pipeline.py.

Implement:

1. async load_omnivoice_model() -> OmnivoiceTTS:
    """Load Omnivoice TTS model."""
    # Import and initialize Omnivoice
    # Load model weights
    # Return model instance
    pass

2. Class OmnivoiceTTS:
    def synthesize(self, text: str, language: str = "th", 
                   output_dir: str = "output/audio/",
                   sample_rate: int = 24000) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            language: "th" or "en"
            output_dir: Directory for output audio file
            sample_rate: Audio sample rate
            
        Returns:
            Path to generated audio file
        """
        # Generate unique filename
        import uuid
        filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Synthesize audio
        # Save to file
        
        return output_path
        
    def cleanup(self):
        """Release model resources."""
        pass

3. Voice configuration:
   - Support multiple voice options
   - Configurable speech rate and pitch
   - Language-specific voice selection

Requirements:
- Handle long text by splitting into chunks (avoid OOM)
- Support Thai language TTS properly
- Create output directory if not exists
- Log synthesis time and file size
- Return error on synthesis failure (do not crash pipeline)
```

---

### ACTION A022 — Integrate ComfyUI for Image Generation

| Field | Detail |
|-------|--------|
| **Action Name** | `integrate_comfyui` |
| **File(s)** | [`content_pipeline.py`](content_pipeline.py) |
| **Effort** 2 days |
| **Dependencies** | Task A020 (Content Pipeline) must be completed first |

**Description:**  
Integrate ComfyUI for automated image generation in the content pipeline. Use ComfyUI's API to generate images from text prompts derived from `visual_concept` properties in the knowledge graph.

**Expected Result:**  
- ComfyUI API client for programmatic image generation
- Prompt construction from knowledge graph `visual_concept` properties
- Image output in PNG/JPG format
- Proper VRAM cleanup after generation

**Prompt:**
```
You are a Python engineer. Add ComfyUI integration to content_pipeline.py.

Implement:

1. async load_comfyui_api() -> ComfyUIClient:
    """Initialize ComfyUI API connection."""
    # Connect to ComfyUI API (typically http://localhost:8188)
    # Verify connection
    return ComfyUIClient()

2. Class ComfyUIClient:
    def __init__(self, api_url: str = "http://localhost:8188"):
        self.api_url = api_url
        self.session = aiohttp.ClientSession()
        
    async def generate(self, prompt: str, output_dir: str = "output/images/",
                       width: int = 1024, height: int = 1024,
                       steps: int = 20, cfg: float = 7.0) -> str:
        """
        Generate image from text prompt via ComfyUI API.
        
        Args:
            prompt: Text prompt for image generation
            output_dir: Directory for output image
            width/height: Image dimensions
            steps: Sampling steps
            cfg: Classifier-free guidance scale
            
        Returns:
            Path to generated image file
        """
        import uuid
        filename = f"image_{uuid.uuid4().hex[:8]}.png"
        
        # Build ComfyUI workflow JSON
        workflow = self._build_workflow(prompt, width, height, steps, cfg)
        
        # Submit to ComfyUI
        response = await self.session.post(
            f"{self.api_url}/prompt",
            json={"prompt": workflow}
        )
        prompt_id = (await response.json())["prompt_id"]
        
        # Wait for completion
        image_data = await self._wait_for_result(prompt_id)
        
        # Save image
        output_path = os.path.join(output_dir, filename)
        await self._save_image(image_data, output_path)
        
        return output_path
        
    def _build_workflow(self, prompt: str, width: int, height: int, 
                        steps: int, cfg: float) -> Dict:
        """Build ComfyUI workflow JSON."""
        # Standard SDXL/SD1.5 workflow with text-to-image
        pass
        
    async def _wait_for_result(self, prompt_id: str) -> bytes:
        """Poll ComfyUI for generation result."""
        # WebSocket or HTTP polling
        pass
        
    async def _save_image(self, data: bytes, path: str):
        """Save image data to file."""
        pass
        
    async def cleanup(self):
        await self.session.close()

Requirements:
- Handle ComfyUI connection failures gracefully
- Timeout protection (max 5 minutes per generation)
- Support different image models (SDXL, SD1.5)
- Log generation parameters and timing
- Create output directory if needed
```

---

## Summary Matrix

| Action ID | Name | Phase | Effort | Key File(s) | Status |
|-----------|------|-------|--------|-------------|--------|
| A001 | define_universal_schema | 1 | 2d | graph_db.py, schema_migration.py | ⬜ Pending |
| A002 | implement_llm_client | 1 | 3d | llm_extractor.py | ⬜ Pending |
| A003 | add_llm_vram_config | 1 | 0.5d | config.py | ⬜ Pending |
| A004 | build_prompt_templates | 1 | 2d | llm_extractor.py | ⬜ Pending |
| A005 | implement_json_validation | 1 | 1d | llm_extractor.py | ⬜ Pending |
| A006 | add_yaml_frontmatter | 1 | 1d | main.py, utils.py | ⬜ Pending |
| A007 | implement_domain_detection | 1 | 1d | utils.py | ⬜ Pending |
| A008 | implement_neo4j_vector_store | 2 | 3d | neo4j_vector_store.py | ⬜ Pending |
| A009 | create_neo4j_vector_index | 2 | 1d | neo4j_vector_store.py | ⬜ Pending |
| A010 | migrate_embedding_model | 2 | 1d | vector_db.py, config.py | ⬜ Pending |
| A011 | implement_dual_write | 2 | 2d | hybrid_search.py | ⬜ Pending |
| A012 | restructure_hybrid_search | 2 | 3d | hybrid_search.py | ⬜ Pending |
| A013 | implement_context_assembly | 2 | 2d | context_assembler.py | ⬜ Pending |
| A014 | implement_vector_backend_flag | 2 | 0.5d | config.py, hybrid_search.py | ⬜ Pending |
| A015 | implement_openwebui_tool | 3 | 2d | openwebui_tool.py | ⬜ Pending |
| A016 | package_openwebui_tool | 3 | 1d | openwebui_tool.py, config | ⬜ Pending |
| A017 | test_openwebui_integration | 3 | 2d | test_openwebui_integration.py | ⬜ Pending |
| A018 | implement_vram_manager | 4 | 3d | vram_manager.py | ⬜ Pending |
| A019 | implement_cuda_monitoring | 4 | 2d | vram_manager.py | ⬜ Pending |
| A020 | build_content_pipeline | 4 | 3d | content_pipeline.py | ⬜ Pending |
| A021 | integrate_omnivoice | 4 | 2d | content_pipeline.py | ⬜ Pending |
| A022 | integrate_comfyui | 4 | 2d | content_pipeline.py | ⬜ Pending |

**Total Estimated Effort:** ~45 developer days across 7 weeks

---

## Execution Order and Critical Path

```
Critical Path:
A003 → A002 → A004 → A005 → A001 → A008 → A009 → A010 → A011 → A012 → A015 → A017

Parallel Tasks (can run simultaneously):
- A006 + A007 (independent of main path)
- A018 + A019 + A020 + A021 + A022 (Phase 4 is largely independent)
- A013 can start after Phase 1 complete (parallel with Phase 2)
```

## Shared Coding Standards (Apply to ALL Actions)

| Standard | Rule |
|----------|------|
| Type Hints | All function signatures must include full type annotations |
| Docstrings | Google-style with Args/Returns/Raises sections |
| Cypher Safety | All Cypher queries MUST use parameterized inputs (`$param`) — never string concatenation |
| Error Handling | Log errors with `logger.error()`, return False/empty on failure — never raise unhandled exceptions |
| Lazy Initialization | Database clients must use lazy init (singleton pattern) |
| Environment Config | All configurable values via `os.getenv()` with safe defaults |
| UTF-8 Safety | All file I/O must specify `encoding="utf-8"` |

## Infrastructure References

| Service | Connection Details |
|---------|-------------------|
| Neo4j (HTTP) | `http://10.10.1.210:7474` |
| Neo4j (Bolt) | `bolt://10.10.1.210:7687` |
| Neo4j Auth | `neo4j/214356ppKK#` |
| Embedding API (external) | `http://192.168.1.212:7700/v1` (OpenAI-compatible, nomic-embed-text-v1.5.f16) |
| GPU | NVIDIA RTX 3090 (24GB VRAM), safety limit: 22GB |
