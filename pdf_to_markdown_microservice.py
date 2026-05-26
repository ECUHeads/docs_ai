"""
Microservice for converting PDF to Markdown with streaming support.
Uses shared utilities from utils.py to avoid code duplication.
"""

import os
import sys
import tempfile
import json
import asyncio
from aiohttp import web
import logging

from utils import (
    setup_pytorch_memory,
    get_device,
    is_marker_available,
    initialize_pdf_converter,
    convert_pdf_to_markdown,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up PyTorch memory management
setup_pytorch_memory()

# Check if hybrid search dependencies are available
# Use lazy getters to avoid crashing when DBs are not ready
try:
    from graph_db import get_graph_db  # noqa: F401
    from vector_db import get_vector_db  # noqa: F401
    from hybrid_search import get_hybrid_search
    hybrid_search_available = True
    logger.info("Hybrid search libraries available")
except ImportError as e:
    hybrid_search_available = False
    logger.warning(f"Hybrid search libraries not available: {e}")

# --- Configuration constants ---
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
PDF_MAGIC_BYTES = b"%PDF"
STREAM_CHUNK_SIZE = 1024  # 1 KB chunks for streaming


def _initialize_converter() -> bool:
    """Initialize the PDF converter once at startup.

    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    if not is_marker_available():
        logger.error("Marker library not available. PDF processing will be disabled.")
        return False

    try:
        initialize_pdf_converter(device=get_device())
        logger.info("Marker models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False


def _validate_pdf_magic_bytes(header: bytes) -> bool:
    """Validate that the uploaded file is a PDF by checking magic bytes.

    Args:
        header: The first 5 bytes of the uploaded file.

    Returns:
        bool: True if the file appears to be a valid PDF.
    """
    return header[:4] == PDF_MAGIC_BYTES


async def extract_with_marker(pdf_path: str) -> str:
    """Extract PDF to Markdown using Marker library.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: The extracted Markdown text, or empty string on failure.
    """
    if not is_marker_available():
        logger.warning(f"Marker not available, skipping {os.path.basename(pdf_path)}")
        return ""

    try:
        logger.info(f"Processing with Marker: {os.path.basename(pdf_path)}")
        markdown_text, _ = convert_pdf_to_markdown(pdf_path)
        return markdown_text if markdown_text else ""
    except Exception as e:
        logger.error(f"Marker failed on {os.path.basename(pdf_path)}: {e}")
        return ""


async def handle_pdf_conversion(request: web.Request) -> web.Response:
    """Handle PDF to Markdown conversion request.

    Expects a multipart form upload with field name 'pdf_file'.
    Validates file type (PDF magic bytes) and enforces a maximum file size.
    Returns a streaming response with the converted Markdown content.

    Args:
        request: HTTP request containing the PDF file.

    Returns:
        web.Response: Streaming response with Markdown content or error JSON.
    """
    temp_file = None
    try:
        # Parse multipart upload
        reader = await request.multipart()

        pdf_part = None
        while True:
            part = await reader.next()
            if part.name == "pdf_file":
                pdf_part = part
                break
            if part is None:
                break

        if pdf_part is None:
            return web.json_response(
                {"error": "No PDF file provided. Use field name 'pdf_file'."},
                status=400,
            )

        # Create a secure temporary file using tempfile module
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix="pdf_upload_"
        )
        temp_path = temp_file.name

        # Read and validate the uploaded PDF
        total_size = 0
        header_read = False
        header_bytes = b""
        try:
            while True:
                chunk = await pdf_part.read_chunk()
                if not chunk:
                    break

                # Capture first chunk for magic bytes validation
                if not header_read:
                    header_bytes = chunk[:5]
                    header_read = True
                    if not _validate_pdf_magic_bytes(header_bytes):
                        temp_file.close()
                        os.unlink(temp_path)
                        return web.json_response(
                            {"error": "Invalid file type. Only PDF files are accepted."},
                            status=400,
                        )

                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    temp_file.close()
                    os.unlink(temp_path)
                    return web.json_response(
                        {
                            "error": (
                                f"File too large. Maximum allowed size is "
                                f"{MAX_FILE_SIZE // (1024 * 1024)} MB."
                            )
                        },
                        status=413,
                    )
                temp_file.write(chunk)
        finally:
            temp_file.close()

        # Convert PDF to Markdown
        markdown_text = await extract_with_marker(temp_path)

        # Build streaming response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/markdown; charset=utf-8",
                "Content-Disposition": 'inline; filename="output.md"',
            },
        )
        await response.prepare(request)

        # Stream the markdown content in fixed-size chunks.
        # Await the streaming directly to avoid race conditions
        # where the background task returns before streaming finishes.
        if markdown_text:
            for i in range(0, len(markdown_text), STREAM_CHUNK_SIZE):
                await response.write(markdown_text[i : i + STREAM_CHUNK_SIZE].encode("utf-8"))

        await response.write_eof()
        return response

    except Exception as e:
        logger.error(f"Error in handle_pdf_conversion: {e}")
        # REVIEW-03 FIX (Item #9): Hide internal error details from client responses.
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return web.json_response({"error": "Internal server error", "code": "CONVERSION_FAILED"}, status=500)
    finally:
        # Ensure temporary file is cleaned up even on unexpected failures
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError as cleanup_err:
                logger.debug(
                    f"Could not clean up temp file {temp_file.name}: {cleanup_err}"
                )


async def handle_hybrid_search(request: web.Request) -> web.Response:
    """Handle hybrid search request using both GraphDB and VectorDB.

    Expects a plain-text or JSON body containing the search query.

    Args:
        request: HTTP request containing the search query.

    Returns:
        web.Response: JSON response with hybrid search results or error.
    """
    try:
        if not hybrid_search_available:
            return web.json_response(
                {"error": "Hybrid search not available"},
                status=503,
            )

        # Accept both plain text and JSON body
        content_type = request.content_type or ""
        if "json" in content_type:
            data = await request.json()
            query = data.get("query", "") if isinstance(data, dict) else ""
        else:
            query = await request.text()

        if not query or not query.strip():
            return web.json_response(
                {"error": "No search query provided"},
                status=400,
            )

        # Perform hybrid search (lazy-init on first use)
        results = get_hybrid_search().search(query.strip())

        return web.json_response({"query": query.strip(), "results": results})

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        return web.json_response(
            {"error": "Internal server error", "code": "SEARCH_FAILED"},
            status=500,
        )


async def handle_store_document(request: web.Request) -> web.Response:
    """Store document in both GraphDB and VectorDB for hybrid search.

    Expects a JSON body with 'doc_id' (str), 'content' (str),
    and optional 'metadata' (dict).

    Args:
        request: HTTP request containing document data.

    Returns:
        web.Response: JSON response with storage result or error.
    """
    try:
        if not hybrid_search_available:
            return web.json_response(
                {"error": "Hybrid search not available"},
                status=503,
            )

        data = await request.json()
        doc_id = data.get("doc_id")
        content = data.get("content")
        metadata = data.get("metadata", {})

        if not doc_id or not content:
            return web.json_response(
                {"error": "Missing required fields: 'doc_id' and 'content'"},
                status=400,
            )

        # Store document (lazy-init on first use)
        success = get_hybrid_search().store_document(doc_id, content, metadata)

        return web.json_response({"doc_id": doc_id, "stored": success})

    except Exception as e:
        logger.error(f"Store document failed: {e}", exc_info=True)
        return web.json_response(
            {"error": "Internal server error", "code": "STORE_FAILED"},
            status=500,
        )


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint.

    Returns a JSON payload describing the service status and capabilities.

    Args:
        request: HTTP request (unused).

    Returns:
        web.Response: JSON response with health status.
    """
    return web.json_response(
        {
            "status": "healthy",
            "marker_available": is_marker_available(),
            "hybrid_search_available": hybrid_search_available,
        }
    )


def create_app() -> web.Application:
    """Create and configure the aiohttp application.

    Initializes the PDF converter at startup and registers all routes.

    Returns:
        web.Application: Configured aiohttp application instance.
    """
    app = web.Application()

    # Initialize the converter
    if not _initialize_converter():
        logger.warning(
            "Failed to initialize converter; "
            "service will work with limited functionality."
        )

    # Register routes
    app.router.add_get("/health", health_check)
    app.router.add_post("/convert", handle_pdf_conversion)
    app.router.add_post("/search/hybrid", handle_hybrid_search)
    app.router.add_post("/document/store", handle_store_document)

    return app


if __name__ == "__main__":
    # REVIEW-03 FIX (Item #4): Use environment variables for host/port instead of hardcoded values.
    host = os.getenv("MICROSERVICE_HOST", "localhost")
    port = int(os.getenv("MICROSERVICE_PORT", "8080"))
    # Run the server
    app = create_app()
    web.run_app(app, host=host, port=port)
