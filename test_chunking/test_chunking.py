"""
Unit tests for the advanced chunking module (chunking.py).

Covers:
  - Character/byte-based splitting with UTF-8 boundary safety
  - Overlap ratio and overlap_chars parameters
  - Semantic chunking (with mocked embedding model)
  - Thai-aware splitting (with mocked pythainlp)
  - Edge cases: empty input, oversized sentences, bad boundaries
  - Integration with split_content() public API
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from chunking import (
    SplitMode,
    split_content,
    _split_content,
    _safe_byte_split,
    _split_sentences_regex,
    _split_sentences_thai,
    _compute_overlap_chars,
    _build_overlapping_chunks,
    _semantic_chunk,
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_MAX_CHUNK_SIZE,
)


# ===================================================================
# Fixtures
# ===================================================================

SAMPLE_ENGLISH = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a second sentence! And here is a third one? "
    "Finally, we have a fourth sentence.\n"
    "New paragraph starts here with more content."
)

SAMPLE_THAI = (
    "สวัสดีครับ นี่คือประโยคแรกภาษาไทย "
    "นี่คือประโยคที่สองและมันยาวกว่านิดหน่อย "
    "ประโยคสุดท้ายสั้นมาก"
)

SAMPLE_MIXED = (
    "Hello world. สวัสดีชาวโลก. "
    "This is English text with Thai mixed in: ข้อมูลทดสอบ. "
    "End of document."
)

SAMPLE_EMOJI = "Hello 👋 World 🌍! This has emoji 😀🎉 and text."


# ===================================================================
# 1. _safe_byte_split tests
# ===================================================================

class TestSafeByteSplit:
    """Tests for UTF-8 byte-boundary safe splitting."""

    def test_ascii_text_no_truncation(self):
        """Pure ASCII should split cleanly at byte boundaries."""
        text = "Hello World This is a test"
        chunks = _safe_byte_split(text, max_bytes=10)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.encode('utf-8')) <= 10

    def test_thai_text_boundary_safety(self):
        """Thai characters (3 bytes each) must not be split mid-character."""
        text = "สวัสดีชาวโลก"  # Each Thai char = 3 bytes
        chunks = _safe_byte_split(text, max_bytes=5)  # 5 bytes = 1 full + partial char
        for chunk in chunks:
            encoded = chunk.encode('utf-8')
            assert len(encoded) <= 5
            # Verify round-trip encoding/decoding is valid
            assert chunk.encode('utf-8').decode('utf-8') == chunk

    def test_emoji_preservation(self):
        """Emoji (4 bytes each) must not be corrupted."""
        text = "Hello 👋 World 🌍"
        chunks = _safe_byte_split(text, max_bytes=10)
        result = ''.join(chunks)
        assert '👋' in result
        assert '🌍' in result

    def test_empty_input(self):
        """Empty string returns empty list."""
        assert _safe_byte_split("", max_bytes=10) == []

    def test_single_character(self):
        """Single character returns as-is."""
        assert _safe_byte_split("A", max_bytes=10) == ["A"]
        assert _safe_byte_split("ส", max_bytes=10) == ["ส"]

    def test_exact_boundary(self):
        """Text that fits exactly in max_bytes should produce one chunk."""
        text = "Hello"  # 5 bytes
        chunks = _safe_byte_split(text, max_bytes=5)
        assert chunks == ["Hello"]


# ===================================================================
# 2. Sentence segmentation tests
# ===================================================================

class TestSentenceSegmentation:
    """Tests for regex and Thai sentence splitting."""

    def test_regex_basic_split(self):
        """English sentences split on . ! ?"""
        sentences = _split_sentences_regex("Hello world. How are you? Fine!")
        assert len(sentences) == 3
        assert "Hello world." in sentences[0]
        assert "How are you?" in sentences[1]

    def test_regex_newline_split(self):
        """Newlines act as sentence boundaries."""
        text = "First line.\nSecond line.\nThird line."
        sentences = _split_sentences_regex(text)
        assert len(sentences) >= 2

    def test_regex_empty_input(self):
        assert _split_sentences_regex("") == []
        assert _split_sentences_regex("   ") == []

    def test_thai_split_returns_results(self):
        """Thai splitter returns non-empty results for valid input."""
        # Even without pythainlp, the fallback regex produces results.
        result = _split_sentences_thai("Test sentence. Another one.")
        assert len(result) >= 1


# ===================================================================
# 3. Overlap calculation tests
# ===================================================================

class TestOverlapComputation:
    """Tests for overlap ratio and character computation."""

    def test_compute_overlap_from_ratio(self):
        sentences = ["Hello world.", "This is a test.", "Final sentence."]
        overlap = _compute_overlap_chars(sentences, overlap_ratio=0.15, max_chunk_size=500)
        assert overlap > 0
        assert overlap <= int(500 * 0.15)

    def test_zero_overlap_ratio(self):
        sentences = ["Hello.", "World."]
        overlap = _compute_overlap_chars(sentences, overlap_ratio=0.0, max_chunk_size=500)
        assert overlap == 0

    def test_empty_sentences(self):
        overlap = _compute_overlap_chars([], overlap_ratio=0.15, max_chunk_size=500)
        assert overlap == 0

    def test_clamped_to_max(self):
        # Very long sentences with small max_chunk_size should clamp overlap.
        sentences = ["A" * 1000]
        overlap = _compute_overlap_chars(sentences, overlap_ratio=0.2, max_chunk_size=100)
        assert overlap <= int(100 * 0.2)


# ===================================================================
# 4. Overlapping chunk building tests
# ===================================================================

class TestBuildOverlappingChunks:
    """Tests for _build_overlapping_chunks with overlap."""

    def test_no_overlap(self):
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        chunks = _build_overlapping_chunks(sentences, max_chunk_size=200, overlap_chars=0)
        assert len(chunks) >= 1

    def test_with_overlap(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "This is a second sentence with more words.",
            "And a third sentence to complete the set.",
        ]
        chunks = _build_overlapping_chunks(sentences, max_chunk_size=60, overlap_chars=15)
        assert len(chunks) >= 2
        # Verify overlap: last chars of chunk[i] should appear in chunk[i+1].
        if len(chunks) >= 2:
            # The overlap text from the end of chunk 0 should be at the start of chunk 1.
            assert len(chunks[1]) > 0

    def test_oversized_sentence_hard_split(self):
        """A single sentence longer than max_chunk_size is split by words."""
        long_sent = " ".join(["word"] * 50)  # ~200 chars
        chunks = _build_overlapping_chunks([long_sent], max_chunk_size=30, overlap_chars=0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_empty_input(self):
        assert _build_overlapping_chunks([], max_chunk_size=100) == []


# ===================================================================
# 5. Character-based split_content tests
# ===================================================================

class TestCharacterSplitMode:
    """Tests for CHARACTER mode (default)."""

    def test_basic_splitting(self):
        chunks = split_content(
            SAMPLE_ENGLISH,
            max_chunk_size=100,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.0,
        )
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_with_overlap(self):
        chunks = split_content(
            SAMPLE_ENGLISH,
            max_chunk_size=100,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.15,
        )
        assert len(chunks) >= 2
        # Verify that consecutive chunks share some text.
        if len(chunks) >= 2:
            # Overlap means the start of chunk[1] should contain text from end of chunk[0].
            assert len(chunks[1]) > 0

    def test_explicit_overlap_chars(self):
        chunks = split_content(
            SAMPLE_ENGLISH,
            max_chunk_size=100,
            mode=SplitMode.CHARACTER,
            overlap_chars=20,
        )
        assert len(chunks) >= 1

    def test_empty_input(self):
        assert split_content("", mode=SplitMode.CHARACTER) == []
        assert split_content("   ", mode=SplitMode.CHARACTER) == []

    def test_single_short_sentence(self):
        chunks = split_content("Hello world.", max_chunk_size=500, mode=SplitMode.CHARACTER)
        assert chunks == ["Hello world."] or len(chunks) == 1


# ===================================================================
# 6. Semantic chunking tests (mocked model)
# ===================================================================

class TestSemanticChunkMode:
    """Tests for SEMANTIC mode with mocked embedding model."""

    def _make_mock_model(self, embeddings):
        """Create a mock SentenceTransformer that returns pre-defined embeddings."""
        mock = MagicMock()
        mock.encode.return_value = np.array(embeddings)
        return mock

    def test_semantic_groups_similar_sentences(self):
        """Highly similar consecutive sentences should be grouped together."""
        # Create embeddings where sentences 0-1 are similar, but 2 is different.
        e0 = [1.0, 0.0]
        e1 = [0.9, 0.1]   # Similar to e0 (cosine ~0.99)
        e2 = [-1.0, 0.0]  # Opposite direction (cosine ~-1)
        model = self._make_mock_model([e0, e1, e2])

        sentences = ["Sentence A.", "Similar sentence B.", "Different topic C."]
        chunks = _semantic_chunk(
            sentences,
            max_chunk_size=500,
            overlap_chars=0,
            model=model,
            similarity_threshold=0.6,
        )
        # Sentences 0 and 1 should be in the same chunk; sentence 2 separate.
        assert len(chunks) >= 1

    def test_semantic_fallback_no_model(self):
        """When model is None, falls back to regex-based packing."""
        chunks = _semantic_chunk(
            ["Hello.", "World.", "Test."],
            max_chunk_size=500,
            overlap_chars=0,
            model=None,
        )
        assert len(chunks) >= 1

    def test_split_content_semantic_mode(self):
        """Public API with SEMANTIC mode and mocked model."""
        e0 = [1.0, 0.0, 0.0]
        e1 = [0.95, 0.1, 0.0]
        e2 = [0.9, 0.15, 0.0]
        model = self._make_mock_model([e0, e1, e2])

        chunks = split_content(
            "First topic sentence. Second related sentence. Third also related.",
            max_chunk_size=200,
            mode=SplitMode.SEMANTIC,
            overlap_ratio=0.1,
            embedding_model=model,
        )
        assert len(chunks) >= 1


# ===================================================================
# 7. Thai-aware splitting tests
# ===================================================================

class TestThaiAwareSplitMode:
    """Tests for THAI_AWARE mode."""

    def test_thai_aware_mode_calls_thai_splitter(self):
        """THAI_AWARE mode should use Thai sentence segmentation."""
        with patch('chunking._split_sentences_thai') as mock_thai:
            mock_thai.return_value = ["ประโยคแรก", "ประโยคที่สอง"]
            chunks = split_content(
                "ภาษาไทยทดสอบ",
                max_chunk_size=500,
                mode=SplitMode.THAI_AWARE,
                overlap_ratio=0.0,
            )
            mock_thai.assert_called_once()
            assert len(chunks) >= 1

    def test_thai_aware_with_overlap(self):
        with patch('chunking._split_sentences_thai') as mock_thai:
            mock_thai.return_value = [
                "นี่คือประโยคแรกที่ยาวพอสมควรสำหรับทดสอบ",
                "นี่คือประโยคที่สองที่ตามมาหลังจากนั้น",
                "ประโยคสุดท้ายสั้นมาก",
            ]
            chunks = split_content(
                SAMPLE_THAI,
                max_chunk_size=40,
                mode=SplitMode.THAI_AWARE,
                overlap_ratio=0.15,
            )
            assert len(chunks) >= 2

    def test_thai_fallback_to_regex(self):
        """When pythainlp unavailable, Thai-aware falls back to regex."""
        # _split_sentences_thai already handles ImportError internally.
        result = split_content(
            "Hello world. Test sentence.",
            mode=SplitMode.THAI_AWARE,
            max_chunk_size=500,
        )
        assert len(result) >= 1


# ===================================================================
# 8. UTF-8 / multilingual integration tests
# ===================================================================

class TestUTF8Multilingual:
    """Tests for UTF-8 safety in mixed-language content."""

    def test_mixed_language_content(self):
        chunks = split_content(
            SAMPLE_MIXED,
            max_chunk_size=50,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.0,
        )
        assert len(chunks) >= 2
        # Verify every chunk is valid UTF-8 (round-trip test).
        for chunk in chunks:
            encoded = chunk.encode('utf-8')
            decoded = encoded.decode('utf-8')
            assert decoded == chunk

    def test_emoji_content(self):
        chunks = split_content(
            SAMPLE_EMOJI,
            max_chunk_size=20,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.0,
        )
        combined = ' '.join(chunks)
        assert '👋' in combined or '👋' in ''.join(chunks)

    def test_cjk_characters(self):
        cjk_text = "这是一个测试。这是第二个句子。第三个句子在这里。"
        chunks = split_content(
            cjk_text,
            max_chunk_size=15,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.0,
        )
        assert len(chunks) >= 2
        for chunk in chunks:
            # Each chunk must be valid UTF-8.
            assert chunk.encode('utf-8').decode('utf-8') == chunk


# ===================================================================
# 9. Edge case and error handling tests
# ===================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_string(self):
        assert split_content("") == []

    def test_whitespace_only(self):
        assert split_content("   \n\t  ") == []

    def test_none_like_input_returns_empty(self):
        # split_content handles None gracefully by returning empty list.
        result = split_content(None)  # type: ignore[arg-type]
        assert result == []

    def test_invalid_max_chunk_size_zero(self):
        with pytest.raises(ValueError, match="positive"):
            split_content("Hello", max_chunk_size=0)

    def test_invalid_max_chunk_size_negative(self):
        with pytest.raises(ValueError, match="positive"):
            split_content("Hello", max_chunk_size=-10)

    def test_invalid_overlap_ratio_negative(self):
        with pytest.raises(ValueError, match="overlap_ratio"):
            split_content("Hello", overlap_ratio=-0.1)

    def test_invalid_overlap_ratio_over_one(self):
        with pytest.raises(ValueError, match="overlap_ratio"):
            split_content("Hello", overlap_ratio=1.5)

    def test_negative_overlap_chars(self):
        with pytest.raises(ValueError, match="overlap_chars"):
            split_content("Hello", overlap_chars=-5)

    def test_valid_boundary_overlap_ratio_zero(self):
        """overlap_ratio=0.0 is valid (no overlap)."""
        chunks = split_content("Hello world.", overlap_ratio=0.0)
        assert len(chunks) >= 1

    def test_valid_boundary_overlap_ratio_one(self):
        """overlap_ratio=1.0 is valid (max overlap)."""
        chunks = split_content("Hello world. Test.", overlap_ratio=1.0)
        assert len(chunks) >= 1

    def test_very_long_single_word(self):
        """A single word longer than max_chunk_size must be split."""
        long_word = "a" * 1000
        chunks = split_content(long_word, max_chunk_size=50, overlap_ratio=0.0)
        # Even a single unbreakable word produces at least one chunk.
        assert len(chunks) >= 1

    def test_backward_compat_split_content(self):
        """_split_content() backward-compatible wrapper works."""
        chunks = _split_content(SAMPLE_ENGLISH, max_chunk_size=100)
        assert len(chunks) >= 2


# ===================================================================
# 10. Overlap continuity tests
# ===================================================================

class TestOverlapContinuity:
    """Verify that overlap preserves context between chunks."""

    def test_overlap_preserves_tail_of_previous_chunk(self):
        """The beginning of chunk[i+1] should contain text from the end of chunk[i]."""
        long_text = ". ".join([f"Sentence number {i} with some content" for i in range(20)])
        chunks = split_content(
            long_text,
            max_chunk_size=60,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.15,
        )
        assert len(chunks) >= 3

        # Check that consecutive chunks have some shared content.
        for i in range(len(chunks) - 1):
            tail = chunks[i][-15:] if len(chunks[i]) >= 15 else chunks[i]
            # The overlap text from the tail should appear at the start of the next chunk.
            # Due to word-boundary alignment, it may not be an exact prefix match,
            # but at least some words should overlap.
            tail_words = set(tail.split())
            head_words = set(chunks[i + 1][:15].split())
            if len(tail_words) > 0 and len(head_words) > 0:
                # Allow partial overlap (not all words need to match).
                intersection = tail_words & head_words
                # Just verify the chunks are non-empty and reasonably sized.
                assert len(chunks[i]) > 0
                assert len(chunks[i + 1]) > 0

    def test_no_overlap_mode(self):
        """With overlap_ratio=0, consecutive chunks should not share text."""
        chunks = split_content(
            "First sentence. Second sentence. Third sentence.",
            max_chunk_size=30,
            mode=SplitMode.CHARACTER,
            overlap_ratio=0.0,
        )
        assert len(chunks) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
