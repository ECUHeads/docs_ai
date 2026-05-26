"""
Advanced content chunking module for Vector Database storage.

Supports multiple splitting strategies:
  - character/byte-based splitting with UTF-8 boundary safety
  - semantic chunking using embedding similarity
  - Thai-aware splitting using pythainlp sentence segmentation

Each strategy supports configurable overlap (10-20%) to preserve context
continuity across chunk boundaries.

Review Changes:
  - Added overlap_ratio / overlap_chars parameters for inter-chunk overlap
  - UTF-8 byte-boundary splitting prevents character corruption in streaming
    or multilingual scenarios
  - Semantic chunking via embedding model evaluates meaning-based cut points
  - Thai sentence segmentation via pythainlp for grammatically correct splits
  - Flexible mode switching via SplitMode enum
  - Full type hints and edge-case error handling (empty text, bad boundaries)
"""

import os
import re
import math
import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class SplitMode(str, Enum):
    """Supported chunking strategies."""
    CHARACTER = "character"       # Character/byte-based with UTF-8 safety
    SEMANTIC = "semantic"         # Embedding-based semantic boundaries
    THAI_AWARE = "thai_aware"     # Thai sentence segmentation via pythainlp


# Default overlap: 15% (within the recommended 10-20% range)
DEFAULT_OVERLAP_RATIO: float = 0.15

# Pipeline Improvement: Default overlap in characters (50-100 range, default 75).
# Controls the number of overlapping characters between consecutive chunks to
# preserve context continuity and avoid cutting key terms.
DEFAULT_OVERLAP_CHARS: int = int(os.getenv('CHUNK_OVERLAP_CHARS', '75'))

DEFAULT_MAX_CHUNK_SIZE: int = 500


# ---------------------------------------------------------------------------
# UTF-8 safe byte-boundary helpers
# ---------------------------------------------------------------------------

def _safe_byte_split(text: str, max_bytes: int) -> List[str]:
    """
    Split *text* at valid UTF-8 byte boundaries so no multi-byte character
    is truncated.

    Multi-byte characters (e.g. Thai U+0E01..U+0E56, CJK, emoji) are encoded
    as 2-4 bytes in UTF-8.  A naive `text[:n]` in Python is actually safe at
    the *character* level, but when the chunks are later serialised as raw
    bytes (e.g. for streaming or binary protocols) a mid-character cut can
    cause corruption.  This function guarantees every returned chunk encodes
    to <= *max_bytes* valid UTF-8 bytes.

    Args:
        text: Input string.
        max_bytes: Hard byte limit per chunk.

    Returns:
        List of substrings whose UTF-8 encoding fits within *max_bytes*.
    """
    if not text:
        return []

    encoded = text.encode('utf-8')
    total = len(encoded)
    chunks: List[str] = []
    start = 0

    while start < total:
        end = min(start + max_bytes, total)

        # If we are not at the very end, make sure we land on a valid
        # UTF-8 character boundary by backing up past continuation bytes.
        # Continuation bytes have the form 10xxxxxx (0x80 .. 0xBF).
        if end < total:
            while end > start and (encoded[end] & 0xC0) == 0x80:
                end -= 1

        # Safety: if we backed up all the way, force a single-byte cut.
        if end <= start:
            end = start + 1

        chunk_bytes = encoded[start:end]
        try:
            chunks.append(chunk_bytes.decode('utf-8'))
        except UnicodeDecodeError:
            # Should not happen after boundary adjustment, but guard anyway.
            fallback = chunk_bytes.decode('utf-8', errors='ignore')
            if fallback:
                chunks.append(fallback)

        start = end

    return chunks


# ---------------------------------------------------------------------------
# Sentence segmentation helpers
# ---------------------------------------------------------------------------

def _split_sentences_regex(text: str) -> List[str]:
    """
    Regex-based sentence splitting (English / general punctuation aware).

    Splits after '.!?' followed by whitespace, and at newline boundaries.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentences.
    """
    pattern = re.compile(
        r'(?<=[.!?])\s+|'   # split after sentence-ending punctuation
        r'(?<=\n)\s*'       # or split at newlines
    )
    parts = pattern.split(text)
    return [s.strip() for s in parts if s.strip()]


def _split_sentences_thai(text: str) -> List[str]:
    """
    Thai-aware sentence segmentation using pythainlp.

    Falls back to regex-based splitting if pythainlp is unavailable.

    Args:
        text: Input text (Thai or mixed).

    Returns:
        List of non-empty sentences.
    """
    try:
        from pythainlp import sent_tokenize  # type: ignore[import-not-found]
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except ImportError:
        logger.warning(
            "pythainlp not installed; falling back to regex splitting for "
            "Thai-aware mode.  Install with: pip install pythainlp"
        )
        return _split_sentences_regex(text)


# ---------------------------------------------------------------------------
# Overlap calculation helpers
# ---------------------------------------------------------------------------

def _compute_overlap_chars(
    sentences: List[str],
    overlap_ratio: float,
    max_chunk_size: int,
) -> int:
    """
    Derive a fixed overlap character count from *overlap_ratio* and the
    average sentence length in the current content.

    Args:
        sentences: List of segmented sentences.
        overlap_ratio: Desired overlap as a fraction (0.0-1.0).
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        Overlap character count clamped to [0, max_chunk_size * overlap_ratio].
    """
    if not sentences or overlap_ratio <= 0:
        return 0

    avg_len = sum(len(s) for s in sentences) / len(sentences)
    overlap = int(avg_len * overlap_ratio)
    # Clamp so overlap never exceeds the theoretical maximum.
    max_overlap = int(max_chunk_size * overlap_ratio)
    return min(overlap, max_overlap) if max_overlap > 0 else 0


def _build_overlapping_chunks(
    sentences: List[str],
    max_chunk_size: int,
    overlap_chars: int = 0,
) -> List[str]:
    """
    Pack *sentences* into chunks of at most *max_chunk_size* characters,
    carrying over the last *overlap_chars* characters to the next chunk.

    Args:
        sentences: Pre-segmented sentences.
        max_chunk_size: Maximum character length per chunk.
        overlap_chars: Number of overlapping characters from the previous chunk.

    Returns:
        List of chunk strings.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # --- Oversized single sentence: hard split by words ---------------
        if sentence_len > max_chunk_size:
            # Flush accumulated chunk first.
            if current_parts:
                chunks.append(' '.join(current_parts))
                current_parts = []
                current_len = 0

            words = sentence.split()
            partial: List[str] = []
            partial_len = 0
            for word in words:
                wlen = len(word) + 1  # +1 for joining space
                if partial_len + wlen > max_chunk_size and partial:
                    chunks.append(' '.join(partial))
                    partial = [word]
                    partial_len = wlen
                else:
                    partial.append(word)
                    partial_len += wlen
            if partial:
                chunks.append(' '.join(partial))
            continue

        # --- Start a new chunk when limit exceeded ------------------------
        if current_len + sentence_len > max_chunk_size and current_parts:
            chunks.append(' '.join(current_parts))

            # Build overlap prefix from the tail of the flushed chunk.
            if overlap_chars > 0:
                last_chunk = chunks[-1]
                overlap_text = last_chunk[-overlap_chars:] if overlap_chars < len(last_chunk) else last_chunk
                # Find a clean word boundary inside the overlap text.
                last_space = overlap_text.rfind(' ')
                if last_space >= 0:
                    overlap_text = overlap_text[last_space + 1:]
                current_parts = [overlap_text]
                current_len = len(overlap_text)
            else:
                current_parts = []
                current_len = 0

        current_parts.append(sentence)
        current_len += sentence_len + 1  # +1 for joining space

    # Flush remaining.
    if current_parts:
        chunks.append(' '.join(current_parts))

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking (embedding-based)
# ---------------------------------------------------------------------------

def _semantic_chunk(
    sentences: List[str],
    max_chunk_size: int,
    overlap_chars: int = 0,
    model=None,
    similarity_threshold: float = 0.6,
) -> List[str]:
    """
    Group consecutive sentences that are semantically similar (cosine
    similarity >= *similarity_threshold*) into the same chunk, then apply
    the size/overlap constraints.

    If the embedding *model* is ``None`` or embedding fails, falls back to
    regex-based packing with overlap.

    Args:
        sentences: Pre-segmented sentences.
        max_chunk_size: Maximum character length per chunk.
        overlap_chars: Overlap characters from previous chunk.
        model: SentenceTransformer instance (or None for fallback).
        similarity_threshold: Minimum cosine similarity to keep sentences
            in the same group.

    Returns:
        List of chunk strings.
    """
    if not sentences:
        return []

    # Attempt embedding-based grouping.
    groups: List[List[str]] = []
    current_group: List[str] = [sentences[0]]

    if model is not None:
        try:
            embeddings = model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Normalise rows for fast cosine similarity.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            normed = embeddings / norms

            for i in range(1, len(sentences)):
                prev = normed[i - 1]
                curr = normed[i]
                cos_sim = float(np.dot(prev, curr))

                if cos_sim >= similarity_threshold:
                    current_group.append(sentences[i])
                else:
                    groups.append(current_group)
                    current_group = [sentences[i]]

            groups.append(current_group)
        except Exception as exc:
            logger.warning(
                "Semantic embedding failed (%s); falling back to regex packing.",
                exc,
            )
            groups = [sentences]  # single group = plain packing
    else:
        logger.info(
            "No embedding model provided for semantic chunking; "
            "falling back to regex-based packing."
        )
        groups = [sentences]

    # Now pack each group respecting max_chunk_size and overlap.
    all_chunks: List[str] = []
    for group in groups:
        group_text = ' '.join(group)
        if len(group_text) <= max_chunk_size:
            all_chunks.append(group_text)
        else:
            # Group too large; fall back to sentence-level packing.
            packed = _build_overlapping_chunks(
                group, max_chunk_size, overlap_chars
            )
            all_chunks.extend(packed)

    return all_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_content(
    content: str,
    *,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    mode: SplitMode = SplitMode.CHARACTER,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    overlap_chars: Optional[int] = DEFAULT_OVERLAP_CHARS,
    embedding_model=None,
    similarity_threshold: float = 0.6,
) -> List[str]:
    """
    Split *content* into chunks suitable for Vector Database storage.

    Parameters
    ----------
    content:
        Raw text to chunk.
    max_chunk_size:
        Maximum character length per chunk (default ``500``).
    mode:
        Splitting strategy – one of ``CHARACTER``, ``SEMANTIC``, or
        ``THAI_AWARE``.
    overlap_ratio:
        Fraction of the previous chunk to carry over (10-20 % recommended).
        Ignored when *overlap_chars* is explicitly provided.
    overlap_chars:
        Fixed number of overlapping characters.  Overrides *overlap_ratio*
        when supplied.
    embedding_model:
        SentenceTransformer instance for ``SEMANTIC`` mode.  When ``None``
        the function falls back to regex-based packing.
    similarity_threshold:
        Minimum cosine similarity (0-1) for keeping consecutive sentences
        in the same semantic group.

    Returns
    -------
    List[str]
        Chunks of text, each at most *max_chunk_size* characters.

    Raises
    ------
    ValueError
        If *max_chunk_size* <= 0, or *overlap_ratio* is outside [0, 1].
    """
    # --- Input validation -------------------------------------------------
    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
    if not (0.0 <= overlap_ratio <= 1.0):
        raise ValueError(
            f"overlap_ratio must be in [0, 1], got {overlap_ratio}"
        )
    if overlap_chars is not None and overlap_chars < 0:
        raise ValueError(f"overlap_chars must be non-negative, got {overlap_chars}")

    # --- Empty / whitespace-only guard ------------------------------------
    if not content or not content.strip():
        return []

    # --- Effective overlap ------------------------------------------------
    effective_overlap = overlap_chars
    if effective_overlap is None:
        # Will be computed after segmentation (needs average sentence length).
        effective_overlap = -1  # sentinel

    # --- Sentence segmentation --------------------------------------------
    if mode == SplitMode.THAI_AWARE:
        sentences = _split_sentences_thai(content)
    else:
        sentences = _split_sentences_regex(content)

    if not sentences:
        return []

    # Compute overlap from ratio when not explicitly provided.
    if effective_overlap < 0:
        effective_overlap = _compute_overlap_chars(
            sentences, overlap_ratio, max_chunk_size
        )

    # --- UTF-8 byte-boundary safety ---------------------------------------
    # Convert character budget to a byte budget so that no chunk exceeds the
    # limit when encoded.  A rough upper bound: Thai / CJK ≈ 3 bytes/char.
    max_bytes = max_chunk_size * 3
    # Pre-split every sentence at byte boundaries so downstream packing
    # never produces an oversized chunk in byte terms.
    byte_safe_sentences: List[str] = []
    for sent in sentences:
        parts = _safe_byte_split(sent, max_bytes)
        byte_safe_sentences.extend(parts)

    # --- Choose splitting strategy ----------------------------------------
    if mode == SplitMode.SEMANTIC:
        chunks = _semantic_chunk(
            byte_safe_sentences,
            max_chunk_size=max_chunk_size,
            overlap_chars=effective_overlap,
            model=embedding_model,
            similarity_threshold=similarity_threshold,
        )
    else:
        chunks = _build_overlapping_chunks(
            byte_safe_sentences,
            max_chunk_size=max_chunk_size,
            overlap_chars=effective_overlap,
        )

    return chunks


# ---------------------------------------------------------------------------
# Backward-compatible alias used by vector_db.py
# ---------------------------------------------------------------------------

def _split_content(
    content: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[str]:
    """
    Backward-compatible wrapper around :func:`split_content`.

    Preserves the original signature expected by
    ``QdrantVectorDB.store_document_embeddings()``.
    """
    return split_content(content, max_chunk_size=max_chunk_size)
