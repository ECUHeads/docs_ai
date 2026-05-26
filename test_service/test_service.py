"""
Test suite for the PDF-to-Markdown microservice.

Launches the microservice as a subprocess and runs smoke tests against
every public endpoint: /health, /convert, /search/hybrid, /document/store.

Usage:
    python test_service.py
    python test_service.py --base-url http://localhost:9090
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required. Install it with: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:8080"
STARTUP_TIMEOUT = 30  # seconds to wait for the service to become healthy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_healthy(base_url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll the /health endpoint until the service responds or *timeout* expires.

    Args:
        base_url: Base URL of the running microservice.
        timeout: Maximum seconds to wait.

    Returns:
        bool: True if the service became healthy within the timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _safe_terminate(process: subprocess.Popen, timeout: int = 5) -> None:
    """Terminate a subprocess, falling back to SIGKILL if needed.

    Args:
        process: The subprocess handle.
        timeout: Seconds to wait after SIGTERM before sending SIGKILL.
    """
    if process.poll() is not None:
        return  # Already exited
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# ---------------------------------------------------------------------------
# Individual test functions — each returns (passed: bool, message: str)
# ---------------------------------------------------------------------------

def test_health_check(base_url: str) -> Tuple[bool, str]:
    """Test GET /health endpoint.

    Expects HTTP 200 with a JSON body containing ``"status": "healthy"``.
    """
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            return False, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        if body.get("status") != "healthy":
            return False, f"Expected status 'healthy', got '{body.get('status')}'"
        return True, "Health check passed"
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def test_convert_no_file(base_url: str) -> Tuple[bool, str]:
    """Test POST /convert with no file — should return 400."""
    try:
        resp = requests.post(f"{base_url}/convert", timeout=10)
        if resp.status_code == 400:
            return True, "Correctly rejected request without PDF file"
        return (
            False,
            f"Expected 400, got {resp.status_code}",
        )
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def test_convert_invalid_file(base_url: str) -> Tuple[bool, str]:
    """Test POST /convert with a non-PDF file — should return 400."""
    try:
        # Create a temporary file that is NOT a PDF
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, mode="w"
        ) as tmp:
            tmp.write("This is not a valid PDF file")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as fh:
                resp = requests.post(
                    f"{base_url}/convert",
                    files={"pdf_file": ("test.pdf", fh, "application/pdf")},
                    timeout=10,
                )
            if resp.status_code == 400:
                return True, "Correctly rejected non-PDF file"
            return (
                False,
                f"Expected 400 for invalid PDF, got {resp.status_code}",
            )
        finally:
            os.unlink(tmp_path)
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def test_hybrid_search_unavailable(base_url: str) -> Tuple[bool, str]:
    """Test POST /search/hybrid — expects 503 when DBs are not configured.

    If hybrid search *is* available, a plain-text body should return 200
    or at least not crash the server.
    """
    try:
        resp = requests.post(
            f"{base_url}/search/hybrid",
            data="test query",
            headers={"Content-Type": "text/plain"},
            timeout=10,
        )
        # Acceptable responses: 200 (DBs available) or 503 (graceful degradation)
        if resp.status_code in (200, 503):
            return True, f"Hybrid search endpoint responded with {resp.status_code}"
        return (
            False,
            f"Expected 200 or 503, got {resp.status_code}",
        )
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def test_store_document_missing_fields(base_url: str) -> Tuple[bool, str]:
    """Test POST /document/store with missing fields — should return 400 (or 503)."""
    try:
        resp = requests.post(
            f"{base_url}/document/store",
            json={},
            timeout=10,
        )
        if resp.status_code in (400, 503):
            return True, f"Correctly rejected incomplete document store request ({resp.status_code})"
        return (
            False,
            f"Expected 400 or 503, got {resp.status_code}",
        )
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def test_store_document_valid_payload(base_url: str) -> Tuple[bool, str]:
    """Test POST /document/store with valid payload.

    Expects 200 when DBs are available, or 503 when gracefully degraded.
    """
    try:
        resp = requests.post(
            f"{base_url}/document/store",
            json={
                "doc_id": "test-doc-001",
                "content": "This is test document content for hybrid search.",
                "metadata": {"source": "test_service.py"},
            },
            timeout=10,
        )
        if resp.status_code in (200, 503):
            return True, f"Document store endpoint responded with {resp.status_code}"
        return (
            False,
            f"Expected 200 or 503, got {resp.status_code}",
        )
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("GET /health", test_health_check),
    ("POST /convert (no file)", test_convert_no_file),
    ("POST /convert (invalid PDF)", test_convert_invalid_file),
    ("POST /search/hybrid", test_hybrid_search_unavailable),
    ("POST /document/store (missing fields)", test_store_document_missing_fields),
    ("POST /document/store (valid payload)", test_store_document_valid_payload),
]


def run_tests(base_url: str) -> List[Tuple[str, bool, str]]:
    """Execute all smoke tests against a running service.

    Args:
        base_url: Base URL of the microservice.

    Returns:
        list of (test_name, passed, message) tuples.
    """
    results: List[Tuple[str, bool, str]] = []
    for name, func in ALL_TESTS:
        passed, message = func(base_url)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {message}")
        results.append((name, passed, message))
    return results


# ---------------------------------------------------------------------------
# Service lifecycle management
# ---------------------------------------------------------------------------

def start_service() -> Optional[subprocess.Popen]:
    """Start the microservice as a background subprocess.

    Returns:
        subprocess.Popen handle, or None if startup failed.
    """
    print("Starting microservice ...")
    try:
        process = subprocess.Popen(
            [sys.executable, "pdf_to_markdown_microservice.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process
    except Exception as exc:
        print(f"Failed to start service: {exc}")
        return None


def stop_service(process: subprocess.Popen) -> None:
    """Stop the microservice subprocess."""
    print("Stopping microservice ...")
    _safe_terminate(process)
    print("Service stopped.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test_service_startup(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Full integration test: start service, run tests, stop service.

    Args:
        base_url: Base URL of the microservice (used for health polling).

    Returns:
        bool: True if all tests passed.
    """
    process = start_service()
    if process is None:
        return False

    try:
        if not _wait_for_healthy(base_url):
            print("Service did not become healthy within timeout.")
            return False

        print("\nRunning smoke tests ...")
        results = run_tests(base_url)

        passed = sum(1 for _, p, _ in results if p)
        total = len(results)
        print(f"\n{'='*50}")
        print(f"Results: {passed}/{total} tests passed")
        print(f"{'='*50}")
        return passed == total

    finally:
        stop_service(process)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the PDF-to-Markdown microservice."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the microservice (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        default=False,
        help=(
            "Do not start/stop the service — assume it is already running. "
            "Only runs the test suite against the given --base-url."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_start:
        # Service is assumed to be already running externally.
        print(f"Running tests against existing service at {args.base_url} ...")
        results = run_tests(args.base_url)
        passed = sum(1 for _, p, _ in results if p)
        total = len(results)
        print(f"\n{'='*50}")
        print(f"Results: {passed}/{total} tests passed")
        print(f"{'='*50}")
        sys.exit(0 if passed == total else 1)
    else:
        success = test_service_startup(args.base_url)
        sys.exit(0 if success else 1)
