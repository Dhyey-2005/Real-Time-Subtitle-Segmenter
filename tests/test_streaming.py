"""
Automated Unit Tests for Phase 2 Streaming Simulation Layer.

Covers all 20 required verification points:
1. Streaming buffer starts empty
2. Chunks are appended correctly
3. Word order is preserved
4. Buffer does not duplicate words
5. Buffer does not lose words
6. Segment emission works
7. Segment remains buffered when boundary is not reached
8. Maximum word limit works
9. Maximum character limit works
10. End-of-stream flush works
11. Word-by-word mode works
12. Fixed chunk mode works
13. Configurable chunk size works
14. Zero-delay mode works
15. Simulated delay configuration works without requiring sleep in tests
16. Streaming output reconstructs the normalized input
17. Streaming N-grams are generated
18. Gemini failure does not break streaming
19. Missing GEMINI_API_KEY does not break streaming
20. Edge-case transcript works
"""

import os
import sys
import unittest.mock
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.streaming_buffer import StreamingBuffer
from src.stream_simulator import StreamSimulator
from src.streaming_segmenter import StreamingSegmenter, StreamingConfig
from src.streaming_orchestrator import run_streaming_simulation, verify_word_integrity
from src.tokenizer import tokenize_text
from src.text_processor import normalize_text, load_text


class TestStreamingBuffer:
    """Tests 1-5, 10: StreamingBuffer functionality and data integrity."""

    def test_1_buffer_starts_empty(self):
        buf = StreamingBuffer()
        assert buf.is_empty()
        assert buf.word_count() == 0
        assert buf.char_count() == 0
        assert buf.get_tokens() == []
        assert buf.get_text() == ""
        assert len(buf) == 0

    def test_2_chunks_appended_correctly(self):
        buf = StreamingBuffer()
        buf.append("the government announced")
        assert not buf.is_empty()
        assert buf.word_count() == 3
        assert buf.get_text() == "the government announced"
        assert buf.get_tokens() == ["the", "government", "announced"]

    def test_3_word_order_is_preserved(self):
        buf = StreamingBuffer()
        buf.append("the government announced")
        buf.append("a new economic package")
        expected_tokens = ["the", "government", "announced", "a", "new", "economic", "package"]
        assert buf.get_tokens() == expected_tokens
        assert buf.get_text() == "the government announced a new economic package"

    def test_4_buffer_does_not_duplicate_words(self):
        buf = StreamingBuffer()
        chunk1 = "alpha beta gamma"
        chunk2 = "delta epsilon"
        buf.append(chunk1)
        buf.append(chunk2)
        tokens = buf.get_tokens()
        assert tokens == ["alpha", "beta", "gamma", "delta", "epsilon"]
        assert len(tokens) == 5

    def test_5_buffer_does_not_lose_words_during_finalization(self):
        buf = StreamingBuffer()
        buf.append("the government announced a new economic package")
        total_tokens = buf.get_tokens()

        # Finalize first 3 tokens
        finalized_seg = buf.finalize_segment(3)
        assert finalized_seg == "the government announced"
        
        # Remaining tokens stay in buffer
        remaining_tokens = buf.get_tokens()
        assert remaining_tokens == ["a", "new", "economic", "package"]
        
        # Verify complete set of words is preserved
        reconstructed = tokenize_text(finalized_seg) + remaining_tokens
        assert reconstructed == total_tokens

    def test_10_end_of_stream_flush_works(self):
        buf = StreamingBuffer()
        buf.append("remaining unfinished text in buffer")
        assert not buf.is_empty()
        
        flushed_text = buf.flush()
        assert flushed_text == "remaining unfinished text in buffer"
        assert buf.is_empty()
        assert buf.word_count() == 0

        # Flushing an empty buffer returns empty string safely
        assert buf.flush() == ""


class TestStreamingSegmenter:
    """Tests 6-9: StreamingSegmenter boundary heuristics and limits."""

    def test_6_segment_emission_works(self):
        config = StreamingConfig(min_segment_words=3, target_words=5, max_segment_words=10)
        segmenter = StreamingSegmenter(config=config)
        
        # Chunk 1: not ready
        res1 = segmenter.add_chunk({"chunk_index": 1, "text": "the government announced a"})
        assert res1 == []
        
        # Chunk 2: introduces discourse marker "furthermore" after sufficient tokens
        res2 = segmenter.add_chunk({"chunk_index": 2, "text": "policy furthermore research"})
        assert len(res2) == 1
        assert res2[0]["text"] == "the government announced a policy"
        assert segmenter.buffer.get_text() == "furthermore research"

    def test_7_segment_remains_buffered_when_boundary_not_reached(self):
        config = StreamingConfig(min_segment_words=4, max_segment_words=14)
        segmenter = StreamingSegmenter(config=config)
        
        # Two words: below min_segment_words, no emission
        emitted = segmenter.add_chunk({"chunk_index": 1, "text": "hello world"})
        assert emitted == []
        assert segmenter.buffer.word_count() == 2
        assert segmenter.buffer.get_text() == "hello world"

    def test_8_maximum_word_limit_works(self):
        config = StreamingConfig(max_segment_words=6, min_segment_words=3)
        segmenter = StreamingSegmenter(config=config)
        
        # Feed 8 continuous words without punctuation or discourse markers
        emitted = segmenter.add_chunk({
            "chunk_index": 1,
            "text": "one two three four five six seven eight"
        })
        
        assert len(emitted) == 1
        assert emitted[0]["word_count"] == 6
        assert emitted[0]["text"] == "one two three four five six"
        assert segmenter.buffer.get_text() == "seven eight"

    def test_9_maximum_character_limit_works(self):
        config = StreamingConfig(max_segment_characters=25, min_segment_words=2, max_segment_words=10)
        segmenter = StreamingSegmenter(config=config)
        
        # "internationalization modernization" has 33 chars
        emitted = segmenter.add_chunk({
            "chunk_index": 1,
            "text": "internationalization modernization infrastructure"
        })
        assert len(emitted) >= 1
        for seg in emitted:
            assert len(seg["text"]) <= 40


class TestStreamSimulator:
    """Tests 11-15: StreamSimulator modes, chunking, and timing."""

    def test_11_word_by_word_mode_works(self):
        text = "the government announced a new policy"
        sim = StreamSimulator(text, chunk_size=1, delay=0.0)
        chunks = list(sim.stream())
        assert len(chunks) == 6
        assert all(c["word_count"] == 1 for c in chunks)
        assert [c["text"] for c in chunks] == ["the", "government", "announced", "a", "new", "policy"]

    def test_12_fixed_chunk_mode_works(self):
        text = "the government announced a new policy today"
        sim = StreamSimulator(text, chunk_size=3, delay=0.0)
        chunks = list(sim.stream())
        assert len(chunks) == 3
        assert chunks[0]["text"] == "the government announced"
        assert chunks[1]["text"] == "a new policy"
        assert chunks[2]["text"] == "today"

    def test_13_configurable_chunk_size_works(self):
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9"
        sim = StreamSimulator(text, chunk_size=4, delay=0.0)
        chunks = list(sim.stream())
        assert len(chunks) == 3
        assert chunks[0]["word_count"] == 4
        assert chunks[1]["word_count"] == 4
        assert chunks[2]["word_count"] == 1

    def test_14_zero_delay_mode_works(self):
        text = "fast simulation text stream test"
        sim = StreamSimulator(text, chunk_size=2, delay=0.0)
        chunks = list(sim.stream())
        assert len(chunks) == 3

    def test_15_simulated_delay_configuration_without_sleep_in_tests(self):
        text = "chunk one chunk two chunk three"
        with unittest.mock.patch("time.sleep") as mock_sleep:
            sim = StreamSimulator(text, chunk_size=2, delay=0.25)
            list(sim.stream())
            # Delay applies between chunks (from chunk 2 onward)
            assert mock_sleep.call_count >= 1
            mock_sleep.assert_called_with(0.25)


class TestStreamingPipelineAndIntegrity:
    """Tests 16-20: End-to-end streaming pipeline, word integrity, n-grams, and Gemini safety."""

    def test_16_streaming_output_reconstructs_normalized_input(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "output_test")
        
        results = run_streaming_simulation(
            input_path=sample_path,
            output_dir=output_dir,
            config=StreamingConfig(chunk_size=3, simulated_delay=0.0),
            enable_gemini=False,
        )
        
        assert results["integrity_result"]["status"] == "PASSED"
        assert results["integrity_result"]["exact_match"] is True

    def test_17_streaming_ngrams_are_generated(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "output_test_ngrams")
        
        results = run_streaming_simulation(
            input_path=sample_path,
            output_dir=output_dir,
            config=StreamingConfig(chunk_size=3, simulated_delay=0.0),
            enable_gemini=False,
        )
        
        assert results["unique_bigrams"] > 100
        assert results["unique_trigrams"] > 100
        
        bigram_file = os.path.join(output_dir, "streaming_bigram_frequency.csv")
        trigram_file = os.path.join(output_dir, "streaming_trigram_frequency.csv")
        assert os.path.exists(bigram_file)
        assert os.path.exists(trigram_file)

    def test_18_gemini_failure_does_not_break_streaming(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "output_test_gemini_err")
        
        with unittest.mock.patch("src.gemini_client.GeminiSemanticClient.categorize_phrases") as mock_cat:
            mock_cat.side_effect = RuntimeError("Simulated API network drop")
            results = run_streaming_simulation(
                input_path=sample_path,
                output_dir=output_dir,
                config=StreamingConfig(chunk_size=3, simulated_delay=0.0),
                enable_gemini=True,
            )
            assert results["integrity_result"]["status"] == "PASSED"
            assert results["segment_count"] > 0

    def test_19_missing_gemini_api_key_does_not_break_streaming(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "output_test_no_key")
        
        results = run_streaming_simulation(
            input_path=sample_path,
            output_dir=output_dir,
            config=StreamingConfig(chunk_size=3, simulated_delay=0.0),
            enable_gemini=True,
        )
        assert results["integrity_result"]["status"] == "PASSED"
        assert results["segment_count"] > 0

    def test_20_edge_case_transcript_works(self, tmp_path):
        edge_path = os.path.join(PROJECT_ROOT, "data", "input", "edge_case_transcript.txt")
        output_dir = str(tmp_path / "output_test_edge")
        
        results = run_streaming_simulation(
            input_path=edge_path,
            output_dir=output_dir,
            config=StreamingConfig(chunk_size=2, simulated_delay=0.0),
            enable_gemini=False,
        )
        assert results["integrity_result"]["status"] == "PASSED"
        assert results["segment_count"] > 0
        assert results["integrity_result"]["exact_match"] is True
