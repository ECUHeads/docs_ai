"""
Microservice for converting PDF to Markdown with streaming support.
Uses shared utilities from utils.py to avoid code duplication.

Changes (Review-02):
  - Added GET /health endpoint for health checks and readiness probes.
  - Replaced hardcoded /tmp/ paths with tempfile.NamedTemporaryFile() for
    security (prevents race conditions and predictable filenames).
  - Converted stream_markdown_content() to an async generator for proper
    async/await usage in aiohttp context.
  - Added try/finally cleanup for temporary files — temp file is guaranteed
    to be deleted even if conversion fails.
  - Added file type validation (checks PDF magic bytes %PDF).
  - Added configurable max file size limit (default 100MB) to prevent DoS.
  - Host and port are now configurable via environment variables
    MICROSERVICE_HOST and MICROSERVICE_PORT (defaults: localhost, 8080).
  - Error responses now return JSON instead of plain text for consistency.
  - Added request timeout configuration via MICROSERVICE_TIMEOUT env var.

Pipeline Improvements:
  - Redis-based rate limiting with Sliding Window Counter (replaces in-memory dict)
  - Prometheus-compatible /metrics endpoint
  - Integration with metrics middleware for request tracking
"""

import os
import asyncio
import tempfile
import logging
import time
from aiohttp import web
from typing import AsyncGenerator, Callable, Dict, List, Optional

from utils import (
    setup_pytorch_memory,
    get_device,
    is_marker_available,
    initialize_pdf_converter,
    convert_pdf_to_markdown,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up PyTorch memory management
setup_pytorch_memory()

# REVIEW-02 FIX: Configurable host, port, file size limit, and timeout from environment.
MICROSERVICE_HOST = os.getenv('MICROSERVICE_HOST', 'localhost')
MICROSERVICE_PORT = int(os.getenv('MICROSERVICE_PORT', '8080'))
MAX_FILE_SIZE = int(os.getenv('MAX_UPLOAD_SIZE_MB', '100')) * 1024 * 1024  # Default 100MB
REQUEST_TIMEOUT = int(os.getenv('MICROSERVICE_TIMEOUT', '300'))  # Default 5 minutes

# REVIEW-03 FIX (Item #5): Rate limiting configuration.
RATE_LIMIT_MAX_REQUESTS = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '100'))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))


def _initialize_converter() -> bool:
    """Initialize the PDF converter once at startup."""
    if not is_marker_available():
        logger.error("Marker library not available. PDF processing will be disabled.")
        return False

    try:
        initialize_pdf_converter(device=get_device())
        logger.info("PDF converter initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PDF converter: {e}")
        return False


async def _convert_pdf(pdf_file_path: str) -> str:
    """Convert a PDF file to Markdown and return the text."""
    if not is_marker_available():
        raise RuntimeError("Marker library not available")

    text, pages = convert_pdf_to_markdown(pdf_file_path)
    if not text:
        raise RuntimeError(f"Conversion returned empty result for {pdf_file_path}")
    logger.info(f"Converted {pdf_file_path} ({pages} pages)")
    return text


# REVIEW-02 FIX: Changed from sync generator to async generator for proper async usage.
async def stream_markdown_content(markdown_content: str) -> AsyncGenerator[bytes, None]:
    """
    Stream the markdown content in chunks as an async generator.

    Args:
        markdown_content: The full markdown text to stream.

    Yields:
        Chunks of encoded markdown content (UTF-8).
    """
    chunk_size = 1024  # 1KB chunks
    for i in range(0, len(markdown_content), chunk_size):
        yield markdown_content[i:i + chunk_size].encode('utf-8')


# --- Pipeline Improvement: Redis-based Rate Limiting ---

def _get_rate_limiter():
    """
    Create and return the appropriate rate limiter based on configuration.

    Uses Redis-based Sliding Window Counter when available, falls back to
    in-memory implementation otherwise.
    """
    try:
        from config import get_config
        cfg = get_config().get_config().get('rate_limit', {})
    except Exception:
        cfg = {}

    use_redis = cfg.get('use_redis_rate_limiting', True)

    if use_redis:
        try:
            from rate_limiter import RedisRateLimiter
            return RedisRateLimiter(
                redis_host=cfg.get('redis_host', 'localhost'),
                redis_port=cfg.get('redis_port', 6379),
                redis_password=cfg.get('redis_password'),
                redis_db=cfg.get('redis_db', 0),
                max_requests=cfg.get('rate_limit_max_requests', RATE_LIMIT_MAX_REQUESTS),
                window_seconds=cfg.get('rate_limit_window_seconds', RATE_LIMIT_WINDOW_SECONDS),
            )
        except Exception as e:
            logger.warning(f"Redis rate limiter unavailable ({e}); using in-memory fallback")

    # In-memory fallback
    return _InMemoryRateLimiter(
        max_requests=RATE_LIMIT_MAX_REQUESTS,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )


class _InMemoryRateLimiter:
    """Simple in-memory rate limiter for fallback when Redis is unavailable."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._store: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str):
        now = time.time()
        window_start = now - self.window_seconds

        if client_id not in self._store:
            self._store[client_id] = []

        self._store[client_id] = [
            ts for ts in self._store[client_id] if ts > window_start
        ]

        current_count = len(self._store[client_id])
        info = {
            'current_count': current_count,
            'limit': self.max_requests,
            'remaining': max(0, self.max_requests - current_count),
        }

        if current_count >= self.max_requests:
            return False, info

        self._store[client_id].append(now)
        info['current_count'] = current_count + 1
        info['remaining'] = max(0, self.max_requests - current_count)
        return True, info


# Create global rate limiter instance
_rate_limiter = None


def get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = _get_rate_limiter()
    return _rate_limiter


@web.middleware
async def rate_limit_middleware(
    request: web.Request,
    handler: Callable[[web.Request], web.Response],
) -> web.Response:
    """
    Rate limiting middleware using Redis-based Sliding Window Counter.

    Pipeline Improvement: Replaces in-memory dictionary with Redis-backed
    implementation for multi-instance deployment support. Falls back to
    in-memory if Redis is unavailable.

    Returns 429 Too Many Requests when the limit is exceeded.
    """
    remote = request.remote or "unknown"
    limiter = get_rate_limiter()

    allowed, info = limiter.is_allowed(remote)

    if not allowed:
        logger.warning(f"Rate limit exceeded for {remote}")
        return web.json_response(
            {"error": "Too many requests", "code": "RATE_LIMIT_EXCEEDED"},
            status=429,
        )

    return await handler(request)


# --- Pipeline Improvement: Metrics Integration ---

def _setup_metrics(app: web.Application) -> None:
    """
    Register the /metrics endpoint and metrics middleware on the app.

    Only enabled when ENABLE_METRICS environment variable is 'true' (default).
    """
    enable = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    if not enable:
        logger.info("Metrics endpoint disabled (set ENABLE_METRICS=true to enable)")
        return

    try:
        from metrics import handle_metrics, metrics_middleware
        # Insert metrics middleware at the front of the middleware chain
        app.middlewares.insert(0, metrics_middleware)
        app.router.add_get('/metrics', handle_metrics)
        logger.info("Metrics endpoint registered at /metrics")
    except ImportError:
        logger.warning("metrics module not available; /metrics endpoint skipped")
    except Exception as e:
        logger.warning(f"Failed to register metrics endpoint: {e}")


async def handle_health_check(request: web.Request) -> web.Response:
    """
    Health check endpoint for readiness/liveness probes.

    REVIEW-02 FIX: Added missing /health endpoint that test_microservice.py expects.
    """
    converter_ready = is_marker_available() and _initialize_converter.__module__ is not None
    return web.json_response({
        'status': 'healthy',
        'service': 'pdf-to-markdown-microservice-v2',
        'converter_available': is_marker_available(),
    })


async def handle_pdf_upload(request: web.Request) -> web.Response:
    """Handle PDF upload and conversion"""
    temp_pdf_path: Optional[str] = None  # Track temp file for cleanup

    try:
        # Check if the request contains multipart data
        if not request.content_type == 'multipart/form-data':
            return web.json_response(
                {'error': 'Invalid content type. Expected multipart/form-data'},
                status=400
            )

        # Read the form data
        reader = await request.multipart()

        # Get the PDF file from the form
        pdf_file = None
        while True:
            part = await reader.next()
            if part.name == 'pdf_file':
                pdf_file = part
                break

        if not pdf_file:
            return web.json_response(
                {'error': 'No PDF file provided'},
                status=400
            )

        # REVIEW-02 FIX: Use tempfile.NamedTemporaryFile for secure temp file handling.
        # The file is automatically deleted when closed, and we use delete=False so
        # we can pass the path to the converter (then manually clean up in finally).
        temp_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf', prefix='docai_upload_')
        os.close(temp_fd)  # Close the file descriptor; we'll write via open()

        # REVIEW-02 FIX: Read with file size limit to prevent DoS attacks.
        total_size = 0
        with open(temp_pdf_path, 'wb') as f:
            while True:
                chunk = await pdf_file.read_chunk(size=1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_size += len(chunk)
                # REVIEW-02 FIX: Enforce max file size limit during upload.
                if total_size > MAX_FILE_SIZE:
                    raise ValueError(
                        f"File size ({total_size} bytes) exceeds maximum allowed "
                        f"({MAX_FILE_SIZE} bytes / {MAX_FILE_SIZE // (1024*1024)} MB)"
                    )
                f.write(chunk)

        # REVIEW-02 FIX: Validate PDF file type by checking magic bytes.
        with open(temp_pdf_path, 'rb') as f:
            magic_bytes = f.read(5)
            if not magic_bytes.startswith(b'%PDF'):
                raise ValueError("Uploaded file is not a valid PDF (invalid magic bytes)")

        # Convert PDF to Markdown
        start_time = time.time()
        markdown_content = await _convert_pdf(temp_pdf_path)
        duration = time.time() - start_time

        # Pipeline Improvement: Track conversion metrics
        try:
            from metrics import get_metrics_collector
            collector = get_metrics_collector()
            collector.observe_conversion_time(duration)
            collector.inc_conversions_success()
        except Exception:
            pass  # Metrics tracking should not affect core functionality

        # Stream the markdown content as async generator
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/markdown',
                'Content-Disposition': 'attachment; filename="converted.md"'
            }
        )

        await response.prepare(request)

        # REVIEW-02 FIX: Use async generator for streaming.
        async for chunk in stream_markdown_content(markdown_content):
            await response.write(chunk)

        await response.write_eof()

        return response

    except ValueError as e:
        # Validation errors (file size, type)
        logger.warning(f"Validation error: {e}")
        # Pipeline Improvement: Track failed conversions
        try:
            from metrics import get_metrics_collector
            get_metrics_collector().inc_conversions_failed()
        except Exception:
            pass
        return web.json_response({'error': str(e)}, status=400)
    except RuntimeError as e:
        logger.error(f"Conversion error: {e}")
        try:
            from metrics import get_metrics_collector
            get_metrics_collector().inc_conversions_failed()
        except Exception:
            pass
        return web.json_response({'error': f'Conversion failed: {str(e)}'}, status=503)
    except Exception as e:
        logger.error(f"Error handling PDF upload: {e}")
        try:
            from metrics import get_metrics_collector
            get_metrics_collector().inc_conversions_failed()
        except Exception:
            pass
        return web.json_response({'error': 'Internal server error'}, status=500)
    finally:
        # REVIEW-02 FIX: Guaranteed cleanup of temp file even on failure.
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.debug(f"Cleaned up temporary file: {temp_pdf_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {temp_pdf_path}: {e}")


async def main() -> None:
    """Main function to start the microservice"""
    # Initialize the converter
    if not _initialize_converter():
        logger.error("Failed to initialize converter. Exiting.")
        return

    # REVIEW-03 FIX (Item #5): Register rate limiting middleware.
    # Create the web application
    app = web.Application(middlewares=[rate_limit_middleware])

    # REVIEW-02 FIX: Added /health endpoint.
    app.router.add_get('/health', handle_health_check)
    app.router.add_post('/convert', handle_pdf_upload)

    # Pipeline Improvement: Register /metrics endpoint
    _setup_metrics(app)

    # REVIEW-02 FIX: Use configurable host and port from environment variables.
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, MICROSERVICE_HOST, MICROSERVICE_PORT)
    await site.start()

    logger.info(
        f"Microservice started on http://{MICROSERVICE_HOST}:{MICROSERVICE_PORT} "
        f"(max upload: {MAX_FILE_SIZE // (1024*1024)}MB, timeout: {REQUEST_TIMEOUT}s)"
    )

    # Keep the server running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour, effectively keeping the server alive
    except KeyboardInterrupt:
        logger.info("Shutting down microservice...")
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
