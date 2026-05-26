"""
Client for testing the PDF-to-Markdown microservice.

Usage:
    python pdf_client.py                          # Test with default ./test.pdf
    python pdf_client.py --pdf-path ./my_doc.pdf  # Test with a specific file
    python pdf_client.py --health-only             # Only test health endpoint
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

import aiohttp

# Default service URL (overridable via environment variable)
DEFAULT_SERVICE_URL = os.getenv("MICROSERVICE_URL", "http://localhost:8080")


async def test_health_check(base_url: str = DEFAULT_SERVICE_URL) -> bool:
    """Test the health check endpoint.

    Args:
        base_url: Base URL of the microservice.

    Returns:
        bool: True if the health check succeeded.
    """
    url = f"{base_url}/health"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(f"Health check — Status: {response.status}")
                body = await response.text()
                if response.status == 200:
                    print(f"Response: {body}")
                    return True
                else:
                    print(f"Error: {body}")
                    return False
    except Exception as exc:
        print(f"Error connecting to service: {exc}")
        return False


async def test_pdf_conversion(
    pdf_path: str,
    base_url: str = DEFAULT_SERVICE_URL,
) -> bool:
    """Test the PDF-to-Markdown conversion endpoint.

    Args:
        pdf_path: Path to the local PDF file to upload.
        base_url: Base URL of the microservice.

    Returns:
        bool: True if the conversion succeeded.
    """
    if not os.path.exists(pdf_path):
        print(f"Test PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")
        return False

    url = f"{base_url}/convert"

    try:
        async with aiohttp.ClientSession() as session:
            with open(pdf_path, "rb") as fh:
                data = aiohttp.FormData()
                data.add_field("pdf_file", fh, filename=os.path.basename(pdf_path))

                async with session.post(url, data=data) as response:
                    print(f"Conversion — Status: {response.status}")

                    if response.status == 200:
                        print("Streaming markdown content:")
                        async for chunk in response.content.iter_chunked(1024):
                            if chunk:
                                print(chunk.decode("utf-8"), end="", flush=True)
                        print()  # trailing newline
                        return True
                    else:
                        error_body = await response.text()
                        print(f"Error: {error_body}")
                        return False
    except Exception as exc:
        print(f"Error connecting to service: {exc}")
        return False


async def run_tests(
    pdf_path: Optional[str],
    health_only: bool,
    base_url: str = DEFAULT_SERVICE_URL,
) -> None:
    """Run the selected test suite.

    Args:
        pdf_path: Path to a PDF file for conversion testing (or None).
        health_only: If True, only run the health check.
        base_url: Base URL of the microservice.
    """
    print(f"Testing PDF-to-Markdown microservice at {base_url} ...")

    # 1. Health check
    print("\n--- 1. Testing health check ---")
    await test_health_check(base_url)

    # 2. PDF conversion (optional)
    if not health_only:
        print("\n--- 2. Testing PDF conversion ---")
        if pdf_path is None:
            pdf_path = "./test.pdf"
        await test_pdf_conversion(pdf_path, base_url)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test client for the PDF-to-Markdown microservice."
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to a test PDF file (default: ./test.pdf).",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        default=False,
        help="Only run the health check test.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
        help=f"Base URL of the microservice (default: {DEFAULT_SERVICE_URL}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_tests(args.pdf_path, args.health_only, args.base_url))
