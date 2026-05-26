"""
Marker Processor Module.

Provides the MarkerProcessor class for PDF/PUB document processing
using the Marker library, with support for external and local LLM endpoints.
"""

import os
import gc
import logging
import argparse
import subprocess
from typing import Optional, Tuple

import torch

# --- 1. Environment & Hardware Fixes ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Configure module-level logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration constants (overridable via constructor)
# ---------------------------------------------------------------------------
DEFAULT_BATCH_MULTIPLIER = 4
DEFAULT_VRAM_THRESHOLD_GB = 2.0
DEFAULT_FORCE_OCR = False
DEFAULT_PAGINATE_OUTPUT = False
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_PAGES = 5000
DEFAULT_MEMORY_LIMIT_MB = 8192
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MAX_WORKERS = 4
DEFAULT_ENABLE_CACHE = True
DEFAULT_PRECISION = "fp16"

# Default LLM settings
DEFAULT_OPENAI_MODEL = "llama3"
# REVIEW-03 FIX (Item #1): Use environment variable instead of hardcoded API key.
# Falls back to a placeholder that prevents accidental real API calls.
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")


class MarkerProcessor:
    """A class to handle PDF and document processing with Marker library and LLMs.

    Supports three modes:
      1. Internal Marker models (default)
      2. External LLM endpoint (OpenAI-compatible protocol)
      3. Local LLM endpoint via Marker's OpenAIService
    """

    def __init__(
        self,
        external_llm_url: Optional[str] = None,
        local_llm_url: Optional[str] = None,
        device: Optional[str] = None,
        # Converter configuration (optional overrides)
        batch_multiplier: int = DEFAULT_BATCH_MULTIPLIER,
        vram_threshold_gb: float = DEFAULT_VRAM_THRESHOLD_GB,
        force_ocr: bool = DEFAULT_FORCE_OCR,
        paginate_output: bool = DEFAULT_PAGINATE_OUTPUT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_pages: int = DEFAULT_MAX_PAGES,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        enable_cache: bool = DEFAULT_ENABLE_CACHE,
        precision: str = DEFAULT_PRECISION,
        # LLM configuration
        openai_model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        """Initialize the MarkerProcessor.

        Args:
            external_llm_url: External LLM endpoint URL (OpenAI-compatible).
            local_llm_url: Local LLM endpoint URL for Marker's OpenAIService.
            device: Device to use ('cuda', 'xpu', 'cpu'). Auto-detected if None.
            batch_multiplier: Batch size multiplier for Marker converter.
            vram_threshold_gb: VRAM threshold in GB before offloading.
            force_ocr: Force OCR even on text-based PDFs.
            paginate_output: Paginate the output Markdown.
            batch_size: Base batch size for inference.
            max_pages: Maximum number of pages to process per document.
            memory_limit_mb: Memory limit in MB before triggering cleanup.
            chunk_size: Number of pages per processing chunk.
            max_workers: Maximum number of worker threads.
            enable_cache: Enable caching of intermediate results.
            precision: Model precision ('fp16', 'bf16', 'fp32').
            openai_model: Model name for external/local LLM calls.
        """
        self.external_llm_url = external_llm_url
        self.local_llm_url = local_llm_url
        self.device = device or self._get_device()
        self.pdf_converter = None
        self.client = None
        self.marker_available = False
        self.config_parser_available = False
        self.openai_model = openai_model

        # Store converter configuration for potential re-initialization
        self.converter_config = {
            "batch_multiplier": batch_multiplier,
            "vram_threshold_gb": vram_threshold_gb,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "batch_size": batch_size,
            "max_pages": max_pages,
            "memory_limit_mb": memory_limit_mb,
            "chunk_size": chunk_size,
            "max_workers": max_workers,
            "enable_cache": enable_cache,
            "precision": precision,
        }

        # Initialize the processor
        self._initialize()

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    def _get_device(self) -> str:
        """Determine the appropriate device to use.

        Returns:
            str: 'cuda', 'xpu', or 'cpu'.
        """
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        logger.warning("CUDA/XPU not available, falling back to CPU")
        return "cpu"

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available.

        Returns:
            bool: True if core dependencies are present.
        """
        logger.info("Checking system dependencies ...")

        # Check PyTorch
        try:
            import torch as _torch  # noqa: F811
            logger.info("PyTorch version: %s", _torch.__version__)
        except ImportError as exc:
            logger.error("PyTorch not available: %s", exc)
            return False

        # Check Marker library
        try:
            from marker.models import create_model_dict  # noqa: F401
            from marker.converters.pdf import PdfConverter  # noqa: F401
            from marker.services.openai import OpenAIService  # noqa: F401

            self.marker_available = True
            logger.info("Marker library available")
        except ImportError as exc:
            logger.warning("Required library 'marker' not available: %s", exc)
            self.marker_available = False

        # Check config_parser (optional)
        try:
            import config_parser  # noqa: F401
            self.config_parser_available = True
            logger.info("config_parser available")
        except ImportError:
            logger.info("config_parser not available, will use fallback")

        return True

    # ------------------------------------------------------------------
    # LLM client initialization
    # ------------------------------------------------------------------
    def _initialize_openai_client(self) -> bool:
        """Initialize OpenAI client for external LLM if specified.

        Returns:
            bool: True on success or if no external LLM is configured.
        """
        if not self.external_llm_url:
            return True

        logger.info("Using external LLM endpoint: %s", self.external_llm_url)
        try:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.external_llm_url,
                api_key=DEFAULT_API_KEY,
            )
            return True
        except Exception as exc:
            logger.error("Failed to initialize external LLM client: %s", exc)
            return False

    def _initialize_local_llm_service(self) -> bool:
        """Initialize local LLM service via Marker's OpenAIService.

        Returns:
            bool: True on success or if no local LLM is configured.
        """
        if not self.local_llm_url:
            return True

        logger.info("Local LLM mode enabled — using OpenAIService with local LLM")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.services.openai import OpenAIService

            local_llm_service = OpenAIService(
                config={
                    "openai_base_url": self.local_llm_url,
                    "openai_api_key": DEFAULT_API_KEY,
                    "openai_model": self.openai_model,
                }
            )

            self.pdf_converter = PdfConverter(
                artifact_dict={},
                llm_service=local_llm_service,
                config=self.converter_config,
            )
            logger.info("Local LLM service initialized successfully")
            return True
        except Exception as exc:
            logger.error("Failed to initialize local LLM service: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Marker model initialization
    # ------------------------------------------------------------------
    def _initialize_marker_models(self) -> bool:
        """Initialize internal Marker models.

        Returns:
            bool: True on success or if Marker is not available.
        """
        if not self.marker_available:
            return True  # Not a failure; just skipped.

        logger.info("Loading Marker models (device=%s) ...", self.device)
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            if self.device == "xpu":
                torch.xpu.set_device(0)

            marker_model_dict = create_model_dict(
                device=self.device, dtype=torch.float16
            )

            self.pdf_converter = PdfConverter(
                artifact_dict=marker_model_dict,
                config=self.converter_config,
            )
            logger.info("Marker models loaded successfully")
            return True
        except Exception as exc:
            logger.error("Failed to initialize models: %s", exc)
            self.pdf_converter = None
            return False

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _initialize(self) -> bool:
        """Initialize all components.

        Returns:
            bool: True if initialization completed without blocking failures.
        """
        if not self._check_dependencies():
            return False
        if not self._initialize_openai_client():
            return False
        if not self._initialize_local_llm_service():
            return False
        if not self._initialize_marker_models():
            return False
        return True

    # ------------------------------------------------------------------
    # Memory cleanup helper
    # ------------------------------------------------------------------
    def _clear_memory(self) -> None:
        """Force GPU/CPU memory cleanup after processing."""
        try:
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception as exc:
            logger.debug("Could not clear GPU memory: %s", exc)

    # ------------------------------------------------------------------
    # Public API — document conversion
    # ------------------------------------------------------------------
    def process_pub_to_pdf(self, pub_path: str) -> Optional[str]:
        """Convert a .pub file to .pdf using LibreOffice.

        Args:
            pub_path: Path to the .pub file.

        Returns:
            str | None: Path to the converted PDF file, or None on failure.
        """
        logger.info("Converting: %s -> PDF", os.path.basename(pub_path))
        out_dir = os.path.dirname(pub_path) or "."

        try:
            subprocess.run(
                ["soffice", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    pub_path,
                    "--outdir",
                    out_dir,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            pdf_path = pub_path.rsplit(".", 1)[0] + ".pdf"
            logger.info("Converted to: %s", pdf_path)
            return pdf_path
        except subprocess.CalledProcessError as exc:
            logger.error("Converting .pub with LibreOffice failed: %s", exc)
            return None
        except FileNotFoundError:
            logger.error(
                "LibreOffice (soffice) not found. "
                "Please install LibreOffice to convert .pub files."
            )
            return None
        except Exception as exc:
            logger.error("Converting .pub failed: %s", exc)
            return None

    def extract_with_marker(self, pdf_path: str) -> Tuple[str, int]:
        """Extract PDF content to Markdown.

        Uses either an external LLM or internal Marker models depending
        on how the processor was configured.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            tuple: (markdown_text, pages_count)
        """
        try:
            if self.external_llm_url:
                return self._extract_via_external_llm(pdf_path)
            else:
                return self._extract_via_marker(pdf_path)
        except Exception as exc:
            logger.error("Unexpected error during extraction: %s", exc)
            return "", 0

    def _extract_via_external_llm(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text using an external LLM endpoint.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            tuple: (markdown_text, pages_count)
        """
        logger.info(
            "Processing with external LLM: %s", os.path.basename(pdf_path)
        )
        text_content = ""

        # Try pdftotext first
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            text_content = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning(
                "pdftotext not available or failed (%s); using placeholder.", exc
            )
            text_content = (
                f"Extracted content from {os.path.basename(pdf_path)} — "
                "placeholder for actual PDF text extraction"
            )

        # Send to external LLM
        if self.client:
            try:
                # REVIEW-03 FIX (Item #3): Use configured model name instead of empty string.
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that extracts "
                                "and formats content from PDF documents."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Please extract and format the following "
                                f"PDF content:\n\n{text_content}"
                            ),
                        },
                    ],
                    max_tokens=2048,
                    temperature=0.3,
                )
                processed_text = response.choices[0].message.content
                return processed_text, 0
            except Exception as exc:
                logger.error("Calling external LLM failed: %s", exc)
                return text_content, 0
        else:
            return text_content, 0

    def _extract_via_marker(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text using internal Marker models.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            tuple: (markdown_text, pages_count)
        """
        if self.pdf_converter is None:
            logger.warning("Marker not available, skipping %s", os.path.basename(pdf_path))
            return "", 0

        logger.info("Processing with Marker: %s", os.path.basename(pdf_path))
        try:
            result = self.pdf_converter(pdf_path)
            markdown_text = result.markdown
            pages = len(result.pages) if hasattr(result, "pages") else 0
            return markdown_text.strip(), pages
        except Exception as exc:
            error_msg = str(exc).lower()
            logger.error("Marker failed on %s: %s", os.path.basename(pdf_path), exc)
            if "out of memory" in error_msg:
                self._clear_memory()
            return "", 0
        finally:
            self._clear_memory()

    # ------------------------------------------------------------------
    # Public API — single document processing
    # ------------------------------------------------------------------
    def process_document(
        self, file_path: str
    ) -> Tuple[bool, Optional[str], str, int]:
        """Process a single document (PDF or PUB).

        Args:
            file_path: Path to the document file.

        Returns:
            tuple: (success, output_file_path, text, pages)
                - success: True if processing completed without fatal error.
                - output_file_path: Path to the intermediate PDF (for .pub)
                  or the original path (for .pdf), or None on failure.
                - text: Extracted Markdown text.
                - pages: Number of pages extracted.
        """
        try:
            ext = file_path.lower().split(".")[-1]
            text: str = ""
            pages: int = 0
            output_file: Optional[str] = None

            if ext == "pub":
                pdf = self.process_pub_to_pdf(file_path)
                if pdf:
                    output_file = pdf
                    text, pages = self.extract_with_marker(pdf)
            elif ext == "pdf":
                output_file = file_path
                text, pages = self.extract_with_marker(file_path)
            else:
                logger.warning(
                    "Unsupported file extension '.%s' for %s",
                    ext,
                    os.path.basename(file_path),
                )

            return True, output_file, text, pages

        except Exception as exc:
            logger.error(
                "Processing document %s failed: %s",
                os.path.basename(file_path),
                exc,
            )
            return False, None, "", 0

    # ------------------------------------------------------------------
    # Public API — batch processing
    # ------------------------------------------------------------------
    def run_step_1(
        self, docs_folder: str, output_folder: str
    ) -> int:
        """Process all supported documents in *docs_folder*.

        For each PDF or PUB found, the extracted Markdown is written to
        *output_folder* as ``<base_name>.md``.

        Args:
            docs_folder: Path to the folder containing documents.
            output_folder: Path to the output folder for Markdown files.

        Returns:
            int: Number of successfully processed documents.
        """
        logger.info(
            ">>> STEP 1: Processing documents from '%s'", docs_folder
        )
        success_count = 0

        # --- Validate / create directories ----------------------------------------
        if not os.path.exists(docs_folder):
            try:
                os.makedirs(docs_folder, exist_ok=True)
                logger.info("Created directory: %s", docs_folder)
            except Exception as exc:
                logger.error("Cannot create directory %s: %s", docs_folder, exc)
                return 0
        elif not os.path.isdir(docs_folder):
            logger.error("%s exists but is not a directory", docs_folder)
            return 0

        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                logger.info("Created output directory: %s", output_folder)
            except Exception as exc:
                logger.error(
                    "Cannot create output directory %s: %s", output_folder, exc
                )
                return 0

        try:
            files = sorted(
                [
                    os.path.join(docs_folder, f)
                    for f in os.listdir(docs_folder)
                    if os.path.isfile(os.path.join(docs_folder, f))
                ]
            )
        except Exception as exc:
            logger.error("Cannot list files in %s: %s", docs_folder, exc)
            return 0

        for file_path in files:
            try:
                success, output_file, text, pages = self.process_document(
                    file_path
                )

                if success and text:
                    base_name = os.path.splitext(
                        os.path.basename(file_path)
                    )[0]
                    md_path = os.path.join(output_folder, f"{base_name}.md")

                    with open(md_path, "w", encoding="utf-8") as fh:
                        fh.write(text)

                    logger.info(
                        "Success: %s -> %s (~%d pages)",
                        os.path.basename(file_path),
                        os.path.basename(md_path),
                        pages,
                    )
                    success_count += 1
                elif not success:
                    logger.error("Failed: %s", os.path.basename(file_path))

            except Exception as exc:
                logger.error(
                    "Processing file %s failed: %s",
                    os.path.basename(file_path),
                    exc,
                )
                continue

        logger.info("Step 1 Complete! Total items processed: %d", success_count)
        return success_count


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    """Main function to run the marker processor from the command line."""
    parser = argparse.ArgumentParser(
        description="PDF/PUB to Markdown Converter with Marker library"
    )
    parser.add_argument(
        "--external-llm-url",
        type=str,
        default=None,
        help=(
            "External LLM endpoint URL (OpenAI-compatible). "
            "If specified, uses external LLM instead of internal models."
        ),
    )
    parser.add_argument(
        "--local-llm-url",
        type=str,
        default=None,
        help=(
            "Local LLM endpoint URL (OpenAI-compatible). "
            "If specified, uses local LLM via Marker's OpenAIService."
        ),
    )
    parser.add_argument(
        "--docs-folder",
        type=str,
        default="./data",
        help="Path to the folder containing PDF/PUB documents (default: ./data).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./output",
        help="Path to the output folder for Markdown files (default: ./output).",
    )
    args = parser.parse_args()

    processor = MarkerProcessor(
        external_llm_url=args.external_llm_url,
        local_llm_url=args.local_llm_url,
    )

    if processor._check_dependencies():
        processor.run_step_1(args.docs_folder, args.output_folder)
    else:
        logger.error("Failed to initialize MarkerProcessor")


if __name__ == "__main__":
    main()
