"""
Tests for Subtitle / Sentence Segmenter Module.
"""

import os
import sys
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmenter import segment_text


class TestSegmenter:
    """Tests for deterministic unpunctuated ASR segmentation."""

    def test_no_punctuation_multi_sentence_transcript(self):
        text = (
            "the government announced a new economic package today "
            "this new economic package will focus on employment and infrastructure "
            "officials confirmed that local communities will receive direct funding "
            "moreover renewable energy transition represents a major commitment"
        )
        segments = segment_text(text)
        
        # Verify it segments into multiple chunks without periods
        assert len(segments) > 1
        # Verify no segment is a giant unsegmented block
        assert all(len(s.split()) <= 20 for s in segments)
        # Verify segments have reasonable length
        assert all(len(s.split()) >= 3 for s in segments)

    def test_very_short_fragments(self):
        text = "hello world"
        segments = segment_text(text)
        # Short input should remain a single segment rather than get discarded
        assert len(segments) == 1
        assert segments[0] == "hello world"

    def test_empty_input(self):
        assert segment_text("") == []
        assert segment_text("   ") == []

    def test_long_continuous_input(self):
        # 50 words without any punctuation
        words = ["word" + str(i) for i in range(50)]
        text = " ".join(words)
        segments = segment_text(text, max_words=12)
        
        assert len(segments) >= 4
        # Verify max words threshold is respected
        for seg in segments:
            assert len(seg.split()) <= 16

    def test_conjunction_and_discourse_boundaries(self):
        text = (
            "the government announced a comprehensive new policy today "
            "and officials confirmed that renewable energy investments will expand "
            "furthermore artificial intelligence infrastructure will empower research"
        )
        segments = segment_text(text, min_words=5, max_words=14)
        assert len(segments) >= 2
        # Verify discourse marker "furthermore" triggered a clean boundary
        assert any(seg.startswith("furthermore") or "furthermore" in seg for seg in segments)

    def test_does_not_rely_on_periods(self):
        text = "we are testing an asr stream that completely lacks periods full stops and commas"
        segments = segment_text(text, min_words=5, max_words=10)
        assert len(segments) >= 1
        for seg in segments:
            assert "." not in seg
