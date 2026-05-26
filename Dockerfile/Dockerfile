# ===================================================================
# Multi-stage Dockerfile for search_knowledge_graph Open WebUI Tool
# ===================================================================
# Usage:
#   docker build -t search-knowledge-graph:latest .
#   docker run -d --name kg-search -p 8081:8081 \
#     -e NEO4J_URI=bolt://host.docker.internal:7687 \
#     -e NEO4J_USERNAME=neo4j \
#     -e NEO4J_PASSWORD=your-password \
#     search-knowledge-graph:latest
# ===================================================================

# ---------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Metadata labels
LABEL maintainer="devops@doc-ai"
LABEL description="Universal Knowledge Graph Search Tool for Open WebUI"
LABEL version="1.0.0"

# Run as non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY openwebui_tool.py .
COPY openwebui_server.py .
COPY hybrid_search.py .
COPY graph_db.py .
COPY config.py .
COPY chunking.py .
COPY vector_db.py .
COPY neo4j_vector_store.py .
COPY utils.py .
COPY metrics.py .
COPY rate_limiter.py .

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_PORT=8081 \
    SEARCH_TOP_K=3 \
    SEARCH_MAX_HOPS=2 \
    SEARCH_TIMEOUT=30 \
    CORS_ORIGINS="*"

# Expose the HTTP port
EXPOSE 8081

# Health check — verifies the service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Switch to non-root user
USER appuser

# Start the server with Uvicorn
CMD ["uvicorn", "openwebui_server:app", "--host", "0.0.0.0", "--port", "8081", "--log-level", "info"]
