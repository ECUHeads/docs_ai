# Doc-AI2 — Document Processing and Hybrid Search Platform

## 1. Project Overview

**Doc-AI2** is a document-processing pipeline that converts PDF files into structured Markdown, indexes the content in both a Graph Database (Neo4j) and a Vector Database (Qdrant), and provides hybrid search combining graph-based relationship traversal with semantic vector similarity. The system exposes its capabilities through an async HTTP microservice built on **aiohttp**, supporting streaming responses, rate limiting, and Prometheus-compatible metrics export.

Key capabilities:

- **PDF-to-Markdown conversion** using the Marker library (GPU-accelerated via PyTorch).
- **Advanced chunking** with character-based, semantic, and Thai-aware splitting strategies.
- **Hybrid search** combining Neo4j graph traversal (graphRAG) and Qdrant vector similarity.
- **Distributed transaction management** via the Saga Pattern for consistent writes across both databases.
- **Microservice API** with health checks, streaming responses, rate limiting, and metrics.

---

## 2. Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python | 3.10+ | Required for async/await and type hinting |
| PyTorch | 2.0+ | CUDA support recommended for GPU acceleration |
| Neo4j | 5.0+ | Graph database for relationship indexing |
| Qdrant | Latest stable | Vector database for semantic search |
| Redis | 6.0+ | Optional; required for distributed rate limiting |
| GNU/Linux / macOS | — | Primary development and deployment platforms |

### Hardware Recommendations

- **GPU**: NVIDIA GPU with ≥ 8 GB VRAM for Marker model inference (fp16).
- **RAM**: ≥ 16 GB for batch processing of large PDF collections.
- **Storage**: SSD recommended for I/O-heavy chunking and indexing operations.

---

## 3. Installation

### 3.1 Clone the Repository

```bash
git clone <repository-url>
cd doc-ai2
```

### 3.2 Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Configure Environment Variables

Create a `.env` file in the project root. Refer to the table below for available variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `QDRANT_URI` | `http://localhost:6333` | Qdrant connection URI |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `MICROSERVICE_HOST` | `localhost` | Microservice bind host |
| `MICROSERVICE_PORT` | `8080` | Microservice bind port |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum PDF upload size (MB) |
| `MICROSERVICE_TIMEOUT` | `300` | Request timeout in seconds |
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window duration |
| `REDIS_HOST` | `localhost` | Redis host for distributed rate limiting |
| `REDIS_PORT` | `6379` | Redis port |
| `CHUNK_OVERLAP_CHARS` | `75` | Overlap characters between chunks |
| `VECTOR_DB_CHUNK_SIZE` | `500` | Default chunk size for vector indexing |
| `DOC_AI_INSTRUCTION` | *(Thai prompt)* | Default instruction template |
| `DOC_AI_MAX_FILE_SIZE_MB` | `500` | Max file size for batch processing |
| `OPENAI_API_KEY` | `sk-placeholder` | API key for external LLM endpoint |

---

## 4. File Structure and Descriptions

```
doc-ai2/
├── main.py                          # Batch PDF-to-Markdown processor (CLI entry point)
├── microservice_v2.py               # Async HTTP microservice (aiohttp) — production version
├── pdf_to_markdown_microservice.py  # Legacy microservice (superseded by microservice_v2.py)
├── pdf_client.py                    # Async client for testing the microservice endpoints
├── config.py                        # Centralized configuration with .env support
├── chunking.py                      # Content chunking: character, semantic, Thai-aware splitting
├── graph_db.py                      # Neo4j integration for graphRAG and relationship extraction
├── vector_db.py                     # Qdrant integration for vector embedding and similarity search
├── hybrid_search.py                 # Hybrid search engine combining GraphDB + VectorDB results
├── saga_orchestrator.py             # Saga-pattern distributed transaction orchestrator
├── parallel_processor.py            # Parallel processing utilities (CPU-bound and I/O-bound)
├── rate_limiter.py                  # Redis-based sliding-window rate limiter with in-memory fallback
├── metrics.py                       # Prometheus-compatible metrics collector and exporter
├── marker_run.py                    # MarkerProcessor class for advanced PDF/PUB processing
├── utils.py                         # Shared utilities: PyTorch memory, device detection, Marker init
├── requirements.txt                 # Python dependency manifest
├── test_chunking.py                 # Unit tests for chunking module
├── test_hybrid_search.py            # Unit tests for hybrid search engine
├── test_microservice.py             # Tests for legacy microservice
├── test_microservice_endpoints.py   # Integration tests for microservice_v2 endpoints
├── test_service.py                  # General service-level tests
├── unsloth_dataset.json             # Training/inference dataset (JSONL-style)
├── output/                          # Directory for converted Markdown outputs
└── docs/                            # Documentation and review artifacts
```

### Module Details

#### [`main.py`](main.py) — Batch PDF-to-Markdown Processor

**Purpose**: Command-line entry point for batch-converting multiple PDF files to Markdown. Uses `ProcessPoolExecutor` for parallel CPU-bound conversion with automatic worker-count detection.

**Usage**:

```bash
# Convert all PDFs in a directory
python main.py --input-dir ./pdfs --output-dir ./output

# Single file conversion
python main.py --input-dir ./pdfs --output-dir ./output --file my_document.pdf
```

---

#### [`microservice_v2.py`](microservice_v2.py) — Production Microservice

**Purpose**: Async HTTP server (aiohttp) exposing PDF conversion, hybrid search, and health-check endpoints. Includes rate limiting, file validation, streaming responses, and Prometheus metrics middleware.

**Endpoints**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check and readiness probe |
| `POST` | `/convert` | Upload PDF, receive Markdown (streaming or full response) |
| `GET` | `/metrics` | Prometheus-compatible metrics export |

**Startup**:

```bash
python microservice_v2.py
# Or with custom host/port:
MICROSERVICE_HOST=0.0.0.0 MICROSERVICE_PORT=9090 python microservice_v2.py
```

---

#### [`pdf_to_markdown_microservice.py`](pdf_to_markdown_microservice.py) — Legacy Microservice

**Purpose**: Earlier version of the microservice. Superseded by `microservice_v2.py` which adds rate limiting, metrics, and improved error handling. Retained for reference and migration purposes.

---

#### [`pdf_client.py`](pdf_client.py) — Microservice Test Client

**Purpose**: Async HTTP client for testing microservice endpoints. Supports health-check-only mode and full PDF conversion tests.

**Usage**:

```bash
# Full test with default file
python pdf_client.py

# Test with a specific PDF
python pdf_client.py --pdf-path ./my_document.pdf

# Health check only
python pdf_client.py --health-only
```

---

#### [`config.py`](config.py) — Configuration Manager

**Purpose**: Centralized configuration module loading from environment variables and `.env` files. Provides validated access to Neo4j, Qdrant, embedding model, rate-limiting, and metrics settings via singleton pattern.

**Usage**:

```python
from config import get_config
cfg = get_config().get_config()
neo4j_uri = cfg['graph_db']['neo4j_uri']
```

---

#### [`chunking.py`](chunking.py) — Content Chunking Engine

**Purpose**: Splits Markdown content into chunks for vector indexing. Supports three strategies:

- **Character-based**: UTF-8 byte-boundary-safe splitting with configurable overlap.
- **Semantic**: Embedding-similarity-based cut-point detection.
- **Thai-aware**: Sentence segmentation via `pythainlp`.

**Usage**:

```python
from chunking import split_content, SplitMode

chunks = split_content(
    text="Your long document content here...",
    mode=SplitMode.CHARACTER,
    max_chunk_size=500,
    overlap_ratio=0.15,
)
```

---

#### [`graph_db.py`](graph_db.py) — Neo4j Graph Database Integration

**Purpose**: Manages Neo4j connections and graphRAG operations. Extracts sections, headings, entities, and relationships from Markdown to build a knowledge graph. Uses batched `UNWIND` transactions for performance.

**Usage**:

```python
from graph_db import get_graph_db

gdb = get_graph_db()
gdb.store_document(doc_id="doc_001", content="# Section\nContent here...")
results = gdb.graph_rag_search(query="financial trading")
```

---

#### [`vector_db.py`](vector_db.py) — Qdrant Vector Database Integration

**Purpose**: Manages Qdrant connections and vector embedding operations. Creates embeddings using `sentence-transformers`, stores chunked content, and performs similarity search with configurable filters.

**Usage**:

```python
from vector_db import get_vector_db

vdb = get_vector_db()
vdb.store_chunks(doc_id="doc_001", chunks=["chunk 1...", "chunk 2..."])
results = vdb.search(query="algorithmic trading strategies", limit=5)
```

---

#### [`hybrid_search.py`](hybrid_search.py) — Hybrid Search Engine

**Purpose**: Combines graph-based (Neo4j) and vector-based (Qdrant) search results using weighted min-max normalized scoring. Includes deduplication when the same document appears in both result sets.

**Usage**:

```python
from hybrid_search import get_hybrid_search

engine = get_hybrid_search()
results = engine.search(query="machine learning finance", limit=10)
for r in results:
    print(f"[{r['score']:.3f}] {r['doc_id']} — {r['content'][:80]}")
```

---

#### [`saga_orchestrator.py`](saga_orchestrator.py) — Distributed Transaction Orchestrator

**Purpose**: Implements the Saga Pattern for consistent writes across Neo4j and Qdrant. Each forward step has a corresponding compensating (rollback) transaction executed in reverse order on failure.

**Usage**:

```python
from saga_orchestrator import SagaOrchestrator

orchestrator = SagaOrchestrator()
result = orchestrator.execute(
    doc_id="doc_001",
    content="# Document Title\nFull content here...",
    metadata={"source": "upload"},
)
```

---

#### [`parallel_processor.py`](parallel_processor.py) — Parallel Processing Utilities

**Purpose**: Provides `ProcessPoolExecutor` for CPU-bound tasks (e.g., PDF conversion) and `asyncio.gather` with semaphores for I/O-bound operations. Includes automatic workload detection, per-task error collection, and configurable retry logic.

---

#### [`rate_limiter.py`](rate_limiter.py) — Rate Limiting Module

**Purpose**: Redis-backed Sliding Window Counter rate limiter supporting multi-instance deployment. Falls back to in-memory limiting when Redis is unavailable.

---

#### [`metrics.py`](metrics.py) — Metrics Collection and Export

**Purpose**: Thread-safe Prometheus-compatible metrics collector. Tracks request counts, conversion times, error rates, and service health. Exports via `/metrics` endpoint in Prometheus text exposition format.

---

#### [`marker_run.py`](marker_run.py) — Advanced Marker Processor

**Purpose**: High-level wrapper around the Marker library supporting internal models, external LLM endpoints (OpenAI-compatible protocol), and configurable processing parameters (batch size, OCR, precision, memory limits).

---

#### [`utils.py`](utils.py) — Shared Utilities

**Purpose**: Common functions used across modules: PyTorch memory management, device detection (`cuda`/`cpu`), Marker library initialization with thread-safe singleton pattern, and PDF-to-Markdown conversion.

---

## 5. Usage Examples

### 5.1 Batch PDF Conversion

```bash
# Convert all PDFs in ./pdfs to Markdown in ./output
python main.py --input-dir ./pdfs --output-dir ./output
```

### 5.2 Start the Microservice

```bash
# Default: localhost:8080
python microservice_v2.py

# Custom configuration
MICROSERVICE_HOST=0.0.0.0 MICROSERVICE_PORT=9090 python microservice_v2.py
```

### 5.3 Test the Microservice

```bash
python pdf_client.py --pdf-path ./sample.pdf
```

### 5.3 Programmatic Hybrid Search

```python
from hybrid_search import get_hybrid_search

engine = get_hybrid_search()
results = engine.search("risk management in trading", limit=5)
for result in results:
    print(f"Score: {result['score']:.4f} | Doc: {result['doc_id']}")
```

### 5.4 Run Unit Tests

```bash
pytest test_chunking.py -v
pytest test_hybrid_search.py -v
pytest test_microservice_endpoints.py -v
```

---

## 6. Configuration Reference

All configuration is managed through environment variables or a `.env` file loaded by [`config.py`](config.py). The full list of supported variables is provided in **Section 3.4**. Key configuration categories:

- **Database**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `QDRANT_URI`
- **Model**: `EMBEDDING_MODEL`, `OPENAI_API_KEY`
- **Microservice**: `MICROSERVICE_HOST`, `MICROSERVICE_PORT`, `MAX_UPLOAD_SIZE_MB`, `MICROSERVICE_TIMEOUT`
- **Rate Limiting**: `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `REDIS_HOST`, `REDIS_PORT`
- **Chunking**: `CHUNK_OVERLAP_CHARS`, `VECTOR_DB_CHUNK_SIZE`

---

## 7. Architecture Diagram

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF Input   │────▶│  microservice_v2  │────▶│   chunking.py    │
│  (upload)     │     │   (aiohttp API)   │     │  (split chunks)  │
└──────────────┘     └────────┬─────────┘     └────────┬────────┘
                              │                        │
                    ┌─────────▼─────────┐       ┌──────▼──────────┐
                    │   main.py         │       │  vector_db.py   │
                    │  (batch CLI)      │       │  (Qdrant index) │
                    └───────────────────┘       └──────┬──────────┘
                                                       │
                    ┌──────────────────┐               │
                    │   graph_db.py    │               │
                    │  (Neo4j graph)   │               │
                    └────────┬─────────┘               │
                             │                        │
                    ┌────────▼────────────────────────▼────────┐
                    │        hybrid_search.py                   │
                    │   (weighted graph + vector results)       │
                    └──────────────────────────────────────────┘
```

---

## 8. License

This project is proprietary software. All rights reserved. Unauthorized copying, distribution, or modification is strictly prohibited.

---

## 9. Support and Contact

For issues, feature requests, or contributions, please refer to the project repository's issue tracker or contact the development team directly.
