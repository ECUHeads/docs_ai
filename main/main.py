"""
Main entry point for PDF-to-Markdown batch processing.
Uses shared utilities from utils.py to avoid code duplication.

Pipeline Improvements:
  - Parallel processing in run_step_1() via ProcessPoolExecutor (CPU-bound)
    with automatic worker count and error handling per file.
  - Configurable parallel strategy via environment variables.
"""

import os
import json
import pathlib
import subprocess
import sys
import argparse
import logging
import torch

from config import domain_detection_mode
from utils import (
    setup_pytorch_memory,
    get_device,
    is_marker_available,
    initialize_pdf_converter,
    convert_pdf_to_markdown,
    detect_language,
    detect_domain,
    inject_frontmatter,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Configurable Constants ---
# Default instruction template (configurable via environment variable)
DEFAULT_INSTRUCTION = os.getenv(
    "DOC_AI_INSTRUCTION",
    "จงสรุปและวิเคราะห์ข้อมูลจากเนื้อหาต่อไปนี้"
)

# Maximum file size allowed for processing (default: 500 MB)
MAX_FILE_SIZE_MB = int(os.getenv("DOC_AI_MAX_FILE_SIZE_MB", "500"))


# --- 1. Environment Setup ---

def setup_environment() -> str:
    """
    Initialize PyTorch memory, detect device, and load Marker models.

    Returns:
        str: The detected device string (e.g., 'cuda', 'cpu').
    """
    setup_pytorch_memory()

    # Check system dependencies
    logger.info("Checking system dependencies...")

    try:
        logger.info(f"PyTorch version: {torch.__version__}")
    except Exception as e:
        logger.error(f"PyTorch not available: {e}")
        sys.exit(1)

    device = get_device()
    logger.info(f"Initializing on device: {device}")

    # Initialize Marker models
    if is_marker_available():
        try:
            initialize_pdf_converter(device=device)
            logger.info("Marker models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.warning("Continuing without Marker models. PDF processing will be disabled.")
    else:
        logger.warning("Marker library not available. PDF processing will be disabled.")

    return device


def is_model_available() -> bool:
    """Check whether Marker models were successfully initialized."""
    from utils import get_pdf_converter
    converter = get_pdf_converter()
    return converter is not None


# --- 2. Validation Helpers ---

# REVIEW-03 FIX (Item #2): Path traversal protection.
def validate_safe_path(path: str, base_dir: str = ".") -> str:
    """
    Validate that the resolved path is within the allowed base directory.

    Args:
        path: The path to validate.
        base_dir: The base directory that paths must be contained within.

    Returns:
        str: The resolved, validated path string.

    Raises:
        ValueError: If the resolved path is outside the allowed base directory.
    """
    resolved = pathlib.Path(path).resolve()
    base = pathlib.Path(base_dir).resolve()
    if not str(resolved).startswith(str(base)):
        raise ValueError(f"Path '{path}' is outside the allowed directory '{base}'")
    return str(resolved)


def check_file_size(file_path: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """
    Validate that a file does not exceed the maximum allowed size.

    Args:
        file_path: Path to the file to check.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        True if the file size is within limits, False otherwise.
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.warning(
                f"Skipping {os.path.basename(file_path)}: "
                f"size ({size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)"
            )
            return False
        return True
    except OSError as e:
        logger.error(f"Cannot check file size for {file_path}: {e}")
        return False


# --- 3. Conversion Functions ---

def process_pub_to_pdf(pub_path: str):
    """Convert .pub to .pdf using LibreOffice."""
    logger.info(f"Converting: {os.path.basename(pub_path)} -> PDF")
    out_dir = os.path.dirname(pub_path) or '.'
    try:
        subprocess.run(['soffice', '--version'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf',
                        pub_path, '--outdir', out_dir],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return pub_path.rsplit('.', 1)[0] + '.pdf'
    except subprocess.CalledProcessError as e:
        logger.error(f"LibreOffice conversion failed for {pub_path}: {e}")
        return None
    except FileNotFoundError:
        logger.error("LibreOffice (soffice) not found. Install LibreOffice to convert .pub files.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting {pub_path}: {e}")
        return None


# --- 4. Processing Logic ---

def _process_single_file(
    file_path: str,
    instruction: str,
    output_md_dir: str,
    max_file_size_mb: int,
) -> dict:
    """
    Process a single PDF/PUB file and return the result.

    This function is designed to be called by ProcessPoolExecutor workers.
    All arguments must be picklable for multiprocessing.

    Pipeline Improvement:
      After Marker returns the Markdown string, this function performs language
      detection, domain classification (heuristic or LLM-based), and injects
      YAML frontmatter before writing the final output to disk.

    Args:
        file_path: Path to the input file.
        instruction: Instruction text for dataset entry.
        output_md_dir: Directory for Markdown output.
        max_file_size_mb: Maximum allowed file size in MB.

    Returns:
        Dict with keys: 'success', 'dataset_entry', 'file_path', 'error'.
    """
    result = {
        'success': False,
        'dataset_entry': None,
        'file_path': file_path,
        'error': None,
    }

    ext = file_path.lower().split('.')[-1]
    text, pages = "", 0

    try:
        # Validate file size before processing
        if not check_file_size(file_path, max_file_size_mb):
            result['error'] = 'File size exceeded'
            return result

        if ext == 'pub':
            pdf = process_pub_to_pdf(file_path)
            if pdf:
                if not check_file_size(pdf, max_file_size_mb):
                    result['error'] = 'Converted PDF size exceeded'
                    return result
                text, pages = convert_pdf_to_markdown(pdf)
        elif ext == 'pdf':
            text, pages = convert_pdf_to_markdown(file_path)

        if text:
            # --- YAML Frontmatter Injection ---
            source_filename = os.path.basename(file_path)

            # Language detection (heuristic).
            language, lang_confidence = detect_language(text)
            logger.info(
                "Detected language='%s' (confidence=%.2f) for '%s'",
                language, lang_confidence, source_filename,
            )

            # Domain detection: LLM mode or heuristic fallback.
            domain, domain_confidence = detect_domain(text, mode=domain_detection_mode)
            logger.info(
                "Detected domain='%s' (confidence=%.2f) for '%s'",
                domain, domain_confidence, source_filename,
            )

            # Inject frontmatter into the markdown content.
            text = inject_frontmatter(
                content=text,
                domain=domain,
                language=language,
                source=source_filename,
            )
            # ----------------------------------

            result['dataset_entry'] = {
                "instruction": instruction,
                "input": "",
                "output": text,
            }

            # Save individual Markdown file to output directory
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            md_file_path = os.path.join(output_md_dir, f"{base_name}.md")
            try:
                with open(md_file_path, "w", encoding="utf-8") as md_f:
                    md_f.write(text)
                logger.info(f"Saved Markdown: {md_file_path}")
            except OSError as e:
                logger.error(f"Failed to save Markdown file {md_file_path}: {e}")

            result['success'] = True
            logger.info(f"Success: {os.path.basename(file_path)} (~{pages} pages)")
        else:
            result['error'] = 'Empty conversion result'

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {os.path.basename(file_path)}: {e}")

    return result


def run_step_1(
    docs_folder: str,
    output_json: str,
    output_md_dir: str = "./output",
    instruction: str = DEFAULT_INSTRUCTION,
    max_file_size_mb: int = MAX_FILE_SIZE_MB,
) -> None:
    """
    Batch-process all PDF/PUB files in docs_folder and save JSON dataset.

    Also saves individual Markdown files to output_md_dir.

    Pipeline Improvement:
      Uses parallel processing via ProcessPoolExecutor for CPU-bound PDF
      conversion tasks. Worker count is auto-detected from system resources.
      Individual file failures are handled gracefully without aborting the
      entire batch.

    Args:
        docs_folder: Path to the folder containing PDF/PUB files.
        output_json: Path to the output JSON dataset file.
        output_md_dir: Directory where individual Markdown files are saved.
        instruction: Instruction text for the dataset entries.
        max_file_size_mb: Maximum allowed file size in megabytes.
    """
    # REVIEW-03 FIX (Item #2): Validate input paths to prevent path traversal attacks.
    try:
        docs_folder = validate_safe_path(docs_folder)
        output_json = validate_safe_path(output_json)
        output_md_dir = validate_safe_path(output_md_dir)
    except ValueError as e:
        logger.error(f"Invalid path: {e}")
        return

    logger.info(f"STEP 1: Processing documents from '{docs_folder}'")
    dataset = []

    if not os.path.exists(docs_folder):
        try:
            os.makedirs(docs_folder, exist_ok=True)
            logger.info(f"Created directory: {docs_folder}")
        except OSError as e:
            logger.error(f"Cannot create directory {docs_folder}: {e}")
            return
    elif not os.path.isdir(docs_folder):
        logger.error(f"{docs_folder} exists but is not a directory")
        return

    try:
        files = sorted(
            os.path.join(docs_folder, f)
            for f in os.listdir(docs_folder)
            if os.path.isfile(os.path.join(docs_folder, f))
        )
    except OSError as e:
        logger.error(f"Cannot list files in {docs_folder}: {e}")
        return

    # Create output directory for Markdown files
    try:
        os.makedirs(output_md_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory {output_md_dir}: {e}")

    # Filter to only processable files
    processable_files = [
        f for f in files
        if f.lower().split('.')[-1] in ('pdf', 'pub')
    ]

    if not processable_files:
        logger.warning(f"No PDF or PUB files found in {docs_folder}")
    else:
        # Pipeline Improvement: Determine parallel processing strategy
        use_parallel = os.getenv('PARALLEL_PROCESSING', 'true').lower() == 'true'
        max_workers = int(os.getenv('PARALLEL_MAX_WORKERS', '0'))

        if use_parallel and len(processable_files) > 1:
            logger.info(
                f"Processing {len(processable_files)} files in parallel "
                f"(workers={max_workers or 'auto'})..."
            )

            # Pipeline Improvement: Use ProcessPoolExecutor for CPU-bound PDF conversion
            import concurrent.futures

            worker_count = max_workers if max_workers > 0 else (os.cpu_count() or 4)
            worker_count = min(worker_count, len(processable_files))

            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(
                            _process_single_file,
                            file_path,
                            instruction,
                            output_md_dir,
                            max_file_size_mb,
                        ): file_path
                        for file_path in processable_files
                    }

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            if result['success'] and result['dataset_entry']:
                                dataset.append(result['dataset_entry'])
                            elif result['error']:
                                logger.warning(
                                    f"File {os.path.basename(file_path)} failed: {result['error']}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error processing {os.path.basename(file_path)}: {e}"
                            )

            except Exception as e:
                logger.error(f"Parallel processing failed: {e}. Falling back to sequential.")
                # Fallback to sequential processing
                for file_path in processable_files:
                    result = _process_single_file(
                        file_path, instruction, output_md_dir, max_file_size_mb
                    )
                    if result['success'] and result['dataset_entry']:
                        dataset.append(result['dataset_entry'])
        else:
            # Sequential processing (original behavior)
            logger.info(f"Processing {len(processable_files)} files sequentially...")
            for file_path in processable_files:
                result = _process_single_file(
                    file_path, instruction, output_md_dir, max_file_size_mb
                )
                if result['success'] and result['dataset_entry']:
                    dataset.append(result['dataset_entry'])

    try:
        output_dir = os.path.dirname(output_json) or '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        logger.info(f"Step 1 Complete! Total items stored: {len(dataset)}")
    except OSError as e:
        logger.error(f"Failed to save output to {output_json}: {e}")


# --- 5. Main Entry Point ---

def main():
    """Main entry point for batch PDF-to-Markdown processing."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='PDF to Markdown Converter with CUDA support'
    )
    parser.add_argument(
        "--docs-folder",
        type=str,
        default="./data",
        help="Path to the folder containing PDF/PUB files (default: ./data)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="unsloth_dataset.json",
        help="Path to the output JSON dataset file (default: unsloth_dataset.json)"
    )
    parser.add_argument(
        "--output-md-dir",
        type=str,
        default="./output",
        help="Directory for individual Markdown output files (default: ./output)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction text for dataset entries (default: from DOC_AI_INSTRUCTION env or Thai default)"
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=MAX_FILE_SIZE_MB,
        help=f"Maximum file size in MB (default: {MAX_FILE_SIZE_MB})"
    )
    args = parser.parse_args()

    # Initialize environment (PyTorch, device, Marker models)
    setup_environment()

    # Check if models are available before processing
    if not is_model_available():
        logger.error("No PDF conversion model available. Aborting.")
        sys.exit(1)

    # Run batch processing
    run_step_1(
        docs_folder=args.docs_folder,
        output_json=args.output_json,
        output_md_dir=args.output_md_dir,
        instruction=args.instruction,
        max_file_size_mb=args.max_file_size_mb,
    )


if __name__ == "__main__":
    main()
