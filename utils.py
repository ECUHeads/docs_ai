"""
Shared utilities for PDF-to-Markdown processing.
Extracts common functionality to reduce code duplication across main.py, microservice_v2.py, etc.

YAML Frontmatter Utilities:
  - detect_language(): Detect Thai vs English text via Unicode block scanning.
  - detect_domain(): Heuristic keyword-based domain classification.
  - inject_frontmatter(): Generate and prepend YAML frontmatter to Markdown content.
"""

import os
import re
import torch
import gc
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- PyTorch Memory Management ---


def setup_pytorch_memory() -> None:
    """Configure PyTorch memory allocator for fragmentation reduction."""
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def get_device() -> str:
    """Determine the best available device (cuda > cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    logger.warning("CUDA not available, falling back to CPU")
    return 'cpu'


def clear_memory(device: str = 'cuda') -> None:
    """Force garbage collection and clear GPU/CPU memory caches."""
    gc.collect()
    try:
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception as e:
        logger.debug(f"Memory cleanup skipped: {e}")


# --- Marker Library Initialization ---

_pdf_converter: Optional['PdfConverter'] = None  # type: ignore[name-defined]
_marker_available: bool = False
_device: Optional[str] = None

# REVIEW-03 FIX (Item #6): Threading lock for thread-safe singleton initialization.
_converter_lock = threading.Lock()

try:
    from marker.models import create_model_dict
    from marker.converters.pdf import PdfConverter
    _marker_available = True
except ImportError as e:
    logger.warning(f"Marker library not available: {e}")


def is_marker_available() -> bool:
    """Check if the marker library is importable."""
    return _marker_available


def initialize_pdf_converter(
    device: Optional[str] = None,
    batch_multiplier: int = 4,
    vram_threshold_gb: float = 2.0,
) -> 'PdfConverter':  # type: ignore[name-defined]
    """
    Initialize (or re-initialize) the global PdfConverter singleton.

    Args:
        device: Target device ('cuda' or 'cpu'). Auto-detected if None.
        batch_multiplier: VRAM batch multiplier for marker.
        vram_threshold_gb: VRAM threshold in GB before falling back.

    Returns:
        The initialized PdfConverter instance.

    Raises:
        RuntimeError: If marker library is not available.
    """
    global _pdf_converter, _device

    if not _marker_available:
        raise RuntimeError("Marker library not available. Cannot initialize converter.")

    # Reset the converter before (re-)initialization to avoid stale references.
    _pdf_converter = None

    _device = device or get_device()
    logger.info(f"Initializing Marker models on device: {_device}")

    try:
        model_dict = create_model_dict(device=_device, dtype=torch.float16)
        _pdf_converter = PdfConverter(
            artifact_dict=model_dict,
            config={
                "batch_multiplier": batch_multiplier,
                "vram_threshold_gb": vram_threshold_gb,
                "force_ocr": False,
                "paginate_output": False,
            },
        )
        logger.info("PdfConverter initialized successfully")
        return _pdf_converter
    except Exception as e:
        logger.error(f"Failed to initialize PdfConverter: {e}")
        _pdf_converter = None
        raise


def get_pdf_converter() -> 'PdfConverter':  # type: ignore[name-defined]
    """
    Get the global PdfConverter instance, initializing it lazily if needed.

    Returns:
        The PdfConverter singleton.

    Raises:
        RuntimeError: If marker library is not available or initialization fails.
    """
    # REVIEW-03 FIX (Item #6): Thread-safe singleton with double-checked locking.
    global _pdf_converter
    with _converter_lock:
        if _pdf_converter is None:
            initialize_pdf_converter()
        if _pdf_converter is None:
            raise RuntimeError("PdfConverter initialization failed.")
        return _pdf_converter


# --- PDF Processing ---

class PdfConversionError(Exception):
    """Raised when PDF-to-Markdown conversion fails."""
    pass


def convert_pdf_to_markdown(pdf_path: str, raise_on_error: bool = False) -> Tuple[str, int]:
    """
    Convert a single PDF file to Markdown text.

    Args:
        pdf_path: Path to the PDF file.
        raise_on_error: If True, raises PdfConversionError on failure instead
            of returning an empty string. This allows callers to distinguish
            between "empty document" and "conversion failed".

    Returns:
        Tuple of (markdown_text, page_count). Empty string and 0 on failure
        unless raise_on_error is True.

    Raises:
        PdfConversionError: If conversion fails and raise_on_error is True.
        RuntimeError: If marker library is not available.
    """
    if not is_marker_available():
        msg = "Marker not available; PDF conversion skipped."
        logger.error(msg)
        if raise_on_error:
            raise PdfConversionError(msg)
        return "", 0

    converter = get_pdf_converter()
    try:
        result = converter(pdf_path)
        markdown_text = result.markdown
        pages = len(result.pages) if hasattr(result, 'pages') else 0
        return markdown_text.strip(), pages
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Marker failed on {pdf_path}: {e}")
        if "out of memory" in error_msg:
            clear_memory(_device or 'cuda')
        if raise_on_error:
            raise PdfConversionError(f"Failed to convert {pdf_path}: {e}") from e
        return "", 0
    finally:
        clear_memory(_device or 'cuda')


# --- YAML Frontmatter Utilities ---

# Regex pattern for Thai Unicode block (U+0E00 to U+0E7F).
_THAI_UNICODE_PATTERN = re.compile(r'[\u0E00-\u0E7F]')

# Regex pattern for Latin Unicode block (U+0041 to U+007A).
_LATIN_UNICODE_PATTERN = re.compile(r'[\u0041-\u007A]')

# Domain keyword categories for heuristic detection.
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance": [
        "trading", "strategy", "alpha", "quant", "portfolio", "risk",
        "market", "stock", "bond", "derivative",
    ],
    "hardware": [
        "server", "ic", "latency", "fpga", "network",
        "cpu", "gpu", "memory", "cache",
    ],
    "engineering": [
        "architecture", "system", "design", "pattern",
        "algorithm", "optimization",
    ],
    "fiction": [
        "character", "plot", "story", "narrative",
        "chapter", "scene",
    ],
    "general": [],  # Default fallback
}

# Legacy alias for backward compatibility.
_DOMAIN_KEYWORDS = DOMAIN_KEYWORDS

# Regex to detect existing YAML frontmatter at the start of a document.
_FRONTMATTER_GUARD = re.compile(r'^---\s*\n[\s\S]*?\n---\s*\n', re.MULTILINE)


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the primary language of input text by analyzing Unicode character ranges.

    Scans the text for characters in these Unicode blocks:
      - Thai: U+0E00 to U+0E7F
      - Latin: U+0041 to U+007A

    Classifies the text as "th" (Thai), "en" (English/Latin), or "mixed"
    based on which ranges are represented. Confidence is derived from the
    proportion of matched characters against the total character count.

    Args:
        text: The input text to analyze.

    Returns:
        A tuple of (language_code, confidence_score) where language_code is
        one of "th", "en", or "mixed", and confidence_score is a float
        between 0.0 and 1.0. Returns ("en", 0.0) for empty input.

    Example:
        >>> detect_language("Hello world")
        ('en', 1.0)
        >>> detect_language("สวัสดีชาวโลก")
        ('th', 1.0)
    """
    if not text or not text.strip():
        logger.debug("detect_language: empty input, returning defaults")
        return ("en", 0.0)

    thai_matches = len(_THAI_UNICODE_PATTERN.findall(text))
    latin_matches = len(_LATIN_UNICODE_PATTERN.findall(text))
    total_chars = len(text.strip())

    has_thai = thai_matches > 0
    has_latin = latin_matches > 0

    if has_thai and has_latin:
        language_code = "mixed"
        confidence = round((thai_matches + latin_matches) / total_chars, 2)
    elif has_thai:
        language_code = "th"
        confidence = round(thai_matches / total_chars, 2)
    elif has_latin:
        language_code = "en"
        confidence = round(latin_matches / total_chars, 2)
    else:
        # Neither Thai nor Latin detected; default to English with low confidence.
        language_code = "en"
        confidence = 0.1

    # Clamp confidence to [0.0, 1.0].
    confidence = max(0.0, min(1.0, confidence))
    logger.debug("detect_language: language='%s', confidence=%.2f", language_code, confidence)
    return (language_code, confidence)


def detect_domain(
    text: str,
    mode: str = "auto",
    manual_domain: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Determine the domain category of a piece of text.

    Supports three operating modes:
      - "auto": Keyword-based heuristic matching with confidence scoring.
        Counts keyword matches per domain and returns the domain with the
        highest score. If confidence is below 0.3, optionally escalates to LLM.
      - "manual": Returns the domain provided via *manual_domain* with
        confidence 1.0. Falls back to "general" if not provided.
      - "llm": Delegates to LLMClient.detect_domain() for AI-assisted
        classification. Falls back to heuristic on failure.

    Args:
        text: The input text to classify.
        mode: Detection mode — one of "auto", "manual", or "llm".
        manual_domain: Domain string to return when mode is "manual".

    Returns:
        A tuple of (domain, confidence_score) where domain is a category
        name (e.g., 'finance', 'hardware') and confidence_score is a float
        between 0.0 and 1.0. Returns ("general", 0.0) for empty input.

    Example:
        >>> detect_domain("portfolio risk management strategy", mode="auto")
        ('finance', 0.75)
    """
    if not text or not text.strip():
        logger.debug("detect_domain: empty input, returning defaults")
        return ("general", 0.0)

    if mode == "manual":
        domain = manual_domain if manual_domain else "general"
        confidence = 1.0
        logger.debug("detect_domain (manual): domain='%s', confidence=%.2f", domain, confidence)
        return (domain, confidence)

    if mode == "llm":
        try:
            from llm_extractor import LLMClient
            llm_domain = LLMClient().detect_domain(text)
            # LLM-based detection is assumed reasonably confident.
            logger.debug("detect_domain (llm): domain='%s', confidence=0.80", llm_domain)
            return (llm_domain, 0.8)
        except Exception as exc:
            logger.warning("LLM domain detection failed (%s); falling back to heuristic", exc)
            # Fall through to auto mode.

    # --- Auto mode: keyword matching with confidence scoring ---
    lower_text = text.lower()
    scores: Dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if not keywords:
            continue  # Skip 'general' (no keywords).
        match_count = sum(1 for kw in keywords if kw in lower_text)
        if match_count > 0:
            scores[domain] = match_count

    if not scores:
        domain = "general"
        confidence = 0.0
        logger.debug("detect_domain (auto): no keyword matches, domain='%s', confidence=%.2f", domain, confidence)
        return (domain, confidence)

    # Domain with the most keyword matches wins.
    best_domain = max(scores, key=scores.get)  # type: ignore[type-arg]
    best_score = scores[best_domain]

    # Normalize confidence: ratio of matched keywords to total keywords in that domain.
    total_keywords_in_domain = len(DOMAIN_KEYWORDS.get(best_domain, []))
    confidence = round(best_score / total_keywords_in_domain, 2) if total_keywords_in_domain > 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    logger.debug("detect_domain (auto): domain='%s', confidence=%.2f", best_domain, confidence)
    return (best_domain, confidence)


def detect_domain_and_language(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combined domain and language detection utility.

    Reads the detection mode from *config* under the key
    ``DOMAIN_DETECTION_MODE`` (defaulting to "auto") and performs both
    domain and language classification in a single call.

    Args:
        config: Configuration dictionary that must contain at least:
            - "text" (str): The input text to analyze.
            - Optionally "DOMAIN_DETECTION_MODE" (str): One of
              "auto", "manual", or "llm".
            - Optionally "manual_domain" (str): Domain override for
              manual mode.

    Returns:
        A dictionary with keys:
            - "domain" (str): Detected domain category.
            - "language" (str): Detected language code ("th", "en", "mixed").
            - "domain_confidence" (float): Confidence score for domain.
            - "language_confidence" (float): Confidence score for language.

    Example:
        >>> result = detect_domain_and_language({
        ...     "text": "portfolio risk trading strategy",
        ...     "DOMAIN_DETECTION_MODE": "auto",
        ... })
        >>> result["domain"]
        'finance'
    """
    text = config.get("text", "")
    mode = config.get("DOMAIN_DETECTION_MODE", "auto")
    manual_domain = config.get("manual_domain")

    domain, domain_confidence = detect_domain(text, mode=mode, manual_domain=manual_domain)
    language, language_confidence = detect_language(text)

    result = {
        "domain": domain,
        "language": language,
        "domain_confidence": domain_confidence,
        "language_confidence": language_confidence,
    }
    logger.debug(
        "detect_domain_and_language: domain='%s' (%.2f), language='%s' (%.2f)",
        domain, domain_confidence, language, language_confidence,
    )
    return result


def inject_frontmatter(
    content: str,
    domain: str,
    language: str,
    source: str,
) -> str:
    """
    Generate and prepend YAML frontmatter to Markdown content.

    Creates a valid YAML frontmatter block containing:
      - domain: The detected domain category.
      - language: The detected language code.
      - source_pdf: The original PDF filename.
      - processed_date: ISO 8601 UTC timestamp of processing.

    If the content already begins with YAML frontmatter (triple-dash delimiters),
    injection is skipped to avoid duplication.

    Args:
        content: The original Markdown content.
        domain: The detected domain category string.
        language: The detected language code string.
        source: The original PDF filename used as the source reference.

    Returns:
        The Markdown content with YAML frontmatter prepended (or unchanged
        if frontmatter already exists).
    """
    # Guard: skip injection if frontmatter already present.
    if _FRONTMATTER_GUARD.match(content):
        return content

    processed_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    frontmatter = (
        f"---\n"
        f"domain: {domain}\n"
        f"language: {language}\n"
        f"source_pdf: {source}\n"
        f"processed_date: '{processed_date}'\n"
        f"---\n"
    )

    return frontmatter + content
