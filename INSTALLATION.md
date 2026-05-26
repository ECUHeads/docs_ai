# Installation & Deployment Guide — search_knowledge_graph Tool for Open WebUI

This guide covers deploying the `search_knowledge_graph` tool as an HTTP service accessible by Open WebUI's function-calling agent workflow.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Variables](#environment-variables)
4. [Local Installation](#local-installation)
5. [Docker Deployment](#docker-deployment)
6. [Registering the Tool in Open WebUI](#registering-the-tool-in-open-webui)
7. [Testing the Connection](#testing-the-connection)
8. [Example Queries](#example-queries)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────┐         HTTP POST           ┌─────────────────────┐
│  Open WebUI   │  ──────────────────────►    │  openwebui_server   │
│  (AI Agent)   │  ←─────────────────────     │  :8081/api/search   │
└──────────────┘      Markdown Response       └────────┬────────────┘
                                                        │
                                                        │ Hybrid Search
                                                        ▼
                                               ┌────────────────┐
                                               │  Neo4j Graph   │
                                               │  + Vector DB   │
                                               └────────────────┘
```

The tool wraps the existing [`SearchKnowledgeGraphTool`](openwebui_tool.py:137) class behind a FastAPI HTTP endpoint, allowing Open WebUI to invoke it via REST API calls during agent function-calling workflows.

---

## Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.12 | Runtime for FastAPI + search engine |
| Neo4j | ≥ 5.0 | Graph database storing knowledge nodes |
| Docker (optional) | ≥ 24.0 | Containerized deployment |
| Open WebUI | Latest | AI interface consuming the tool |

---

## Environment Variables

All configuration is driven by environment variables. Create a `.env` file or export them in your shell:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEO4J_URI` | string | `bolt://localhost:7687` | Neo4j database connection URI |
| `NEO4J_USERNAME` | string | `neo4j` | Neo4j authentication username |
| `NEO4J_PASSWORD` | string | `password` | Neo4j authentication password |
| `SEARCH_TOP_K` | integer | `3` | Number of top results to return per search |
| `SEARCH_MAX_HOPS` | integer | `2` | Maximum graph traversal hops for context expansion |
| `SEARCH_TIMEOUT` | integer | `30` | Search timeout in seconds |
| `SERVER_PORT` | integer | `8081` | HTTP server listening port |
| `CORS_ORIGINS` | string | `*` | Comma-separated list of allowed CORS origins |

### Example `.env` File

```env
# Neo4j Connection
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password

# Search Configuration
SEARCH_TOP_K=5
SEARCH_MAX_HOPS=3
SEARCH_TIMEOUT=45

# Server Configuration
SERVER_PORT=8081
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

---

## Local Installation

### Step 1: Clone and Install Dependencies

```bash
cd /src-code/doc-ai2
pip install -r requirements.txt
```

Key dependencies installed:
- `fastapi` + `uvicorn` — HTTP server framework
- `neo4j` — Graph database driver
- `sentence-transformers` — Embedding generation for vector search
- `pythainlp` — Thai language processing support

### Step 2: Set Environment Variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export SERVER_PORT=8081
```

Or use `python-dotenv`:

```bash
pip install python-dotenv
# Place .env file in project root
set -a && source .env && set +a
```

### Step 3: Start the Server

```bash
uvicorn openwebui_server:app --host 0.0.0.0 --port 8081
```

Or use the built-in entry point:

```bash
python openwebui_server.py
```

Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8081 (Press CTRL+C to quit)
```

### Step 4: Verify Health Check

```bash
curl http://localhost:8081/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "search_knowledge_graph",
  "version": "1.0.0"
}
```

---

## Docker Deployment

### Option A: Build and Run Locally

```bash
# Build the image
docker build -t search-knowledge-graph:latest .

# Run with environment variables
docker run -d \
  --name kg-search \
  -p 8081:8081 \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your-password \
  -e SEARCH_TOP_K=5 \
  -e SEARCH_MAX_HOPS=3 \
  search-knowledge-graph:latest
```

### Option B: Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  kg-search:
    build: .
    container_name: kg-search-service
    ports:
      - "8081:8081"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}
      - SEARCH_TOP_K=5
      - SEARCH_MAX_HOPS=3
      - SEARCH_TIMEOUT=30
      - SERVER_PORT=8081
      - CORS_ORIGINS=*
    depends_on:
      neo4j:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    restart: unless-stopped

  neo4j:
    image: neo4j:5.20
    container_name: neo4j-db
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-password}
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
    volumes:
      - neo4j-data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD:-password}", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 60s

volumes:
  neo4j-data:
```

Run with:

```bash
docker compose up -d
```

### Option C: Production with Gunicorn

For production deployments, use Gunicorn with multiple Uvicorn workers:

```bash
gunicorn openwebui_server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8081 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile -
```

---

## Registering the Tool in Open WebUI

### Method 1: Manual Registration via UI

1. **Open Open WebUI** → Navigate to **Workspace** → **Tools**
2. Click **Add Custom Tool** (or **+ New Tool**)
3. Fill in the tool configuration:

   | Field | Value |
   |-------|-------|
   | **Name** | `search_knowledge_graph` |
   | **Description** | `Search the Universal Knowledge Graph for concepts, equations, hardware specifications, trading strategies, and creative content seeds.` |
   | **Endpoint URL** | `http://localhost:8081/api/search` (or your deployed URL) |
   | **HTTP Method** | `POST` |
   | **Content Type** | `application/json` |

4. **Paste the JSON Schema** into the tool definition field:

```json
{
  "type": "function",
  "function": {
    "name": "search_knowledge_graph",
    "description": "Search the Universal Knowledge Graph for concepts, equations, hardware specifications, trading strategies, and creative content seeds. The graph contains multi-domain knowledge with technical descriptions (Thai and English), simple explanations for general audience, analogies comparing technical concepts to everyday things, and creative fiction seeds inspired by technical knowledge.",
    "parameters": {
      "type": "object",
      "properties": {
        "search_query": {
          "type": "string",
          "description": "Natural language search query in Thai or English. Example: 'algorithmic trading strategies' or 'กลยุทธ์การเทรดอัตโนมัติ'"
        }
      },
      "required": ["search_query"]
    }
  }
}
```

5. Click **Save** to register the tool.
6. The tool is now available for AI agents to call during conversations.

### Method 2: Import Configuration File

If your Open WebUI supports tool import:

```bash
# Upload openwebui_tool_config.json via the UI import feature
# Or reference it directly:
curl -X POST http://localhost:8080/api/tools/import \
  -H "Content-Type: application/json" \
  -d @openwebui_tool_config.json
```

### Method 3: Dynamic Schema Discovery

The server exposes its own schema at `/api/schema`:

```bash
curl http://localhost:8081/api/schema
```

Use this response to auto-populate the tool definition in Open WebUI.

---

## Testing the Connection

### Test 1: Health Check

```bash
curl http://localhost:8081/health
```

Expected:
```json
{"status": "healthy", "service": "search_knowledge_graph", "version": "1.0.0"}
```

### Test 2: English Query

```bash
curl -X POST http://localhost:8081/api/search \
  -H "Content-Type: application/json" \
  -d '{"search_query": "algorithmic trading strategies"}'
```

Expected:
```json
{
  "result": "# Search Results for: \"algorithmic trading strategies\"\n\n**Found 5 related nodes**\n\n...",
  "status": "success",
  "query": "algorithmic trading strategies"
}
```

### Test 3: Thai Query

```bash
curl -X POST http://localhost:8081/api/search \
  -H "Content-Type: application/json" \
  -d '{"search_query": "กลยุทธ์การเทรดอัตโนมัติ"}'
```

### Test 4: With Custom Parameters

```bash
curl -X POST http://localhost:8081/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "search_query": "machine learning models",
    "top_k": 10,
    "max_hops": 3
  }'
```

### Test 5: Verify CORS Headers

```bash
curl -I -X POST http://localhost:8081/api/search \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  -d '{"search_query": "test"}'
```

Check response headers include:
```
access-control-allow-origin: *
```

---

## Example Queries

### English Queries

| Query | Use Case |
|-------|----------|
| `"algorithmic trading strategies"` | Find trading strategy knowledge |
| `"mean reversion model"` | Specific financial model lookup |
| `"Python machine learning libraries"` | Technical tool comparison |
| `"neural network architecture"` | Deep learning concepts |
| `"risk management framework"` | Financial risk knowledge |

### Thai Queries (ภาษาไทย)

| Query | Use Case |
|-------|----------|
| `"กลยุทธ์การเทรดอัตโนมัติ"` | Algorithmic trading strategies |
| `"โมเดลการเรียนรู้ของเครื่อง"` | Machine learning models |
| `"เครือข่ายประสาทเทียม"` | Neural networks |
| `"การจัดการความเสี่ยงทางการเงิน"` | Financial risk management |
| `"ระบบแนะนำสินค้า"` | Recommendation systems |

### Mixed Language Queries

The tool supports code-switching between Thai and English:

```bash
curl -X POST http://localhost:8081/api/search \
  -H "Content-Type: application/json" \
  -d '{"search_query": "algorithmic trading กลยุทธ์การเทรด"}'
```

---

## Troubleshooting

### Connection Refused on Port 8081

```bash
# Check if the server is running
curl http://localhost:8081/health

# If not, start it:
uvicorn openwebui_server:app --host 0.0.0.0 --port 8081
```

### Neo4j Connection Error

```bash
# Verify Neo4j is accessible:
neo4j console  # Check Neo4j logs

# Test connection manually:
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('$NEO4J_URI'); print(driver.verify_connectivity()); driver.close()"
```

### CORS Errors in Browser

Ensure `CORS_ORIGINS` includes your Open WebUI origin:

```bash
export CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
```

### Search Timeout

Increase the timeout if queries are complex:

```bash
export SEARCH_TIMEOUT=60
```

### Empty Results

1. Verify Neo4j database has data loaded
2. Check `SEARCH_TOP_K` is not too low
3. Increase `SEARCH_MAX_HOPS` for broader graph traversal
4. Try rephrasing the query with different keywords

### Docker Network Issues

When running Neo4j in a separate container, use Docker DNS:

```bash
# Inside the kg-search container:
NEO4J_URI=bolt://neo4j:7687  # Use service name, not localhost
```

For host-networked containers accessing host Neo4j:

```bash
NEO4J_URI=bolt://host.docker.internal:7687
```

---

## File Reference

| File | Purpose |
|------|---------|
| [`openwebui_tool_config.json`](openwebui_tool_config.json) | Tool registration metadata and schema |
| [`openwebui_server.py`](openwebui_server.py) | FastAPI HTTP server wrapper |
| [`openwebui_tool.py`](openwebui_tool.py) | Core tool implementation (existing) |
| [`Dockerfile`](Dockerfile) | Container build definition |
| [`INSTALLATION.md`](INSTALLATION.md) | This deployment guide |
| [`requirements.txt`](requirements.txt) | Python dependencies |

---

## License

This deployment package follows the same license as the parent project.
