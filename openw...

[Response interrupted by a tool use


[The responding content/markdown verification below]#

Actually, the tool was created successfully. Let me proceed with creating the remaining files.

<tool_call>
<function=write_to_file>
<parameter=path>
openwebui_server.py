"""
Open WebUI HTTP Server Wrapper for search_knowledge_graph Tool

Exposes the SearchKnowledgeGraphTool as a REST API endpoint for Open WebUI
integration. Built with FastAPI + Uvicorn for async performance.

Usage:
    uvicorn openwebui_server:app --host 0.0.0.0 --port 8081

Environment Variables:
    NEO4J_URI         — Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USERNAME    — Neo4j username (default: neo4j)
    NEO4J_PASSWORD    — Neo4j password (default: password)
    SEARCH_TOP_K      — Top-K results per search (default: 3)
    SEARCH_MAX_HOPS   — Max graph traversal hops (default: 2)
    SEARCH_TIMEOUT    — Timeout in seconds (default: 30)
    SERVER_PORT       — HTTP server port (default: 8081)
    CORS_ORIGINS      — Comma-separated allowed origins (default: *)
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from Environment Variables
# ---------------------------------------------------------------------------
class AppConfig:
    """Centralized configuration loaded from environment variables."""
    
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "3"))
    SEARCH_MAX_HOPS: int = int(os.getenv("SEARCH_MAX_HOPS", "2"))
    SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT", "30"))
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8081"))
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """Incoming search request payload."""
    search_query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language search query in Thai or English.",
        examples=["algorithmic trading strategies", "กลยุทธ์การเทรดอัตโนมัติ"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default top-K results count.",
    )
    max_hops: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Override default max graph traversal hops.",
    )


class SearchResponse(BaseModel):
    """Outgoing search response payload."""
    result: str = Field(..., description="Formatted Markdown context block with search results.")
    status: str = Field(..., description="'success' or 'error'.")
    query: str = Field(..., description="The original search query echoed back.")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    service: str = Field(default="search_knowledge_graph")
    version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Search Knowledge Graph API",
    description=(
        "HTTP wrapper for the Universal Knowledge Graph Search Tool. "
        "Provides hybrid graph + vector search over multi-domain knowledge "
        "including technical descriptions, trading strategies, and creative content."
    ),
    version="1.0.0",
)

# CORS Middleware — allow Open WebUI to reach this service from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Tool Instance (Lazy-loaded per request to avoid startup DB dependency)
# ---------------------------------------------------------------------------
def _get_tool():
    """Factory to create a SearchKnowledgeGraphTool instance."""
    from openwebui_tool import SearchKnowledgeGraphTool
    return SearchKnowledgeGraphTool()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Liveness/Readiness health check endpoint.
    
    Used by Docker HEALTHCHECK and orchestration systems to verify
    that the service is running and accepting requests.
    """
    return HealthResponse()


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """Execute a hybrid knowledge graph search.
    
    Accepts a natural language query in Thai or English and returns
    a formatted Markdown context block with relevant knowledge nodes
    discovered via hybrid vector + graph traversal search.
    
    **Timeout:** Configurable via SEARCH_TIMEOUT env var (default: 30s).
    **Error Handling:** Returns valid response with status='error' on failure.
    """
    top_k = request.top_k or AppConfig.SEARCH_TOP_K
    max_hops = request.max_hops or AppConfig.SEARCH_MAX_HOPS
    
    logger.info(
        "Search request: query=%r, top_k=%d, max_hops=%d",
        request.search_query, top_k, max_hops,
    )
    
    try:
        tool = _get_tool()
        result_text = tool.execute(request.search_query)
        
        return SearchResponse(
            result=result_text,
            status="success",
            query=request.search_query,
        )
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        
        error_result = (
            f"# Search Results for: \"{request.search_query}\"\n\n"
            f"> ❌ **Search error:** {str(exc)}\n"
        )
        
        return SearchResponse(
            result=error_result,
            status="error",
            query=request.search_query,
        )


@app.get("/api/schema", tags=["Metadata"])
async def get_tool_schema():
    """Return the JSON Schema tool definition for Open WebUI registration.
    
    This endpoint allows Open WebUI to dynamically discover the tool's
    function-calling schema without hardcoding it in the UI.
    """
    from openwebui_tool import TOOL_DEFINITION
    return JSONResponse(content=TOOL_DEFINITION)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    """Start the Uvicorn server with environment-configured settings."""
    import uvicorn
    
    uvicorn.run(
        "openwebui_server:app",
        host="0.0.0.0",
        port=AppConfig.SERVER_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
