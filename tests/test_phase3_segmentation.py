"""
Automated Unit Tests for Phase 3 Improved Subtitle Segmentation.

Covers all 22 required verification points:
1. Existing Phase 1 tests pass (tested via suite)
2. Existing Phase 2 tests pass (tested via suite)
3. Boundary scoring works
4. Strong discourse transition gets higher score than a weak boundary
5. Conjunction alone does not always cause segmentation
6. Multi-word phrase protection works
7. Phrase is not split unnecessarily
8. Maximum word limit remains enforced
9. Maximum character limit remains enforced
10. Minimum segment size behavior works
11. End-of-stream flush works
12. Streaming behavior remains incremental
13. Boundary reason is recorded
14. Boundary score is recorded
15. Word order remains unchanged
16. No words are lost
17. No words are duplicated
18. Sample transcript passes
19. Edge-case transcript passes
20. Phase 3 comparison is generated
21. Hard-limit violations are zero
22. Missing Gemini API key does not break Phase 3
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phrase_protection import PhraseProtector
from src.boundary_scorer import BoundaryScorer
from src.improved_segmenter import ImprovedStreamingSegmenter
from src.streaming_segmenter import StreamingConfig
from src.segmentation_comparator import run_phase3_pipeline, evaluate_segment_quality
from src.tokenizer import tokenize_text
from src.text_processor import normalize_text, load_text


class TestBoundaryScoring:
    """Tests 3-5, 10: Deterministic boundary scoring logic."""

    def test_3_boundary_scoring_works(self):
        scorer = BoundaryScorer()
        tokens = ["the", "government", "announced", "a", "policy"]
        evaluation = scorer.evaluate_boundary(tokens, 3)
        assert isinstance(evaluation.score, float)
        assert isinstance(evaluation.reason, str)
        assert isinstance(evaluation.is_valid, bool)

    def test_4_strong_discourse_transition_higher_score_than_weak(self):
        scorer = BoundaryScorer()
        tokens_discourse = ["the", "policy", "improved", "however", "challenges", "remain"]
        tokens_weak = ["the", "policy", "improved", "and", "challenges", "remain"]

        eval_discourse = scorer.evaluate_boundary(tokens_discourse, 3)
        eval_weak = scorer.evaluate_boundary(tokens_weak, 3)

        assert eval_discourse.score > eval_weak.score
        assert eval_discourse.reason == "discourse_transition"

    def test_5_conjunction_alone_does_not_always_cause_segmentation(self):
        scorer = BoundaryScorer(emission_threshold=40.0)
        # Short clause with conjunction
        tokens = ["officials", "met", "and", "agreed", "on", "terms"]
        eval_res = scorer.evaluate_boundary(tokens, 2)
        # Should not reach emission threshold alone because clause is short
        assert eval_res.score < scorer.emission_threshold
        assert not eval_res.is_valid

    def test_10_minimum_segment_size_behavior(self):
        scorer = BoundaryScorer(min_words=3)
        tokens = ["hello", "world", "again", "today"]
        eval_short = scorer.evaluate_boundary(tokens, 2)
        assert not eval_short.is_valid
        assert eval_short.score == -999.0
        assert eval_short.reason == "below_min_length"


class TestPhraseProtection:
    """Tests 6-7: Multi-word phrase protection."""

    def test_6_multi_word_phrase_protection_detects_cut(self):
        protector = PhraseProtector()
        tokens = ["we", "need", "artificial", "intelligence", "infrastructure", "today"]
        # Splitting after "artificial" (index 3) cuts inside "artificial intelligence infrastructure"
        cut_result = protector.is_split_inside_phrase(tokens, 3)
        assert cut_result is not None
        phrase_name, start, end = cut_result
        assert "artificial intelligence" in phrase_name

        # Splitting after "infrastructure" (index 5) is safe
        safe_cut = protector.is_split_inside_phrase(tokens, 5)
        assert safe_cut is None

    def test_7_phrase_not_split_unnecessarily_by_scorer(self):
        scorer = BoundaryScorer()
        tokens = ["we", "need", "new", "economic", "package", "immediately"]
        # Candidate split inside "new economic package"
        cut_eval = scorer.evaluate_boundary(tokens, 3)
        assert not cut_eval.is_valid
        assert cut_eval.phrase_broken is not None
        assert cut_eval.score < 0


class TestImprovedStreamingSegmenter:
    """Tests 8-9, 11-17: Incremental streaming, limits, and integrity."""

    def test_8_maximum_word_limit_remains_enforced(self):
        config = StreamingConfig(max_segment_words=10, min_segment_words=3)
        seg = ImprovedStreamingSegmenter(config=config)
        # Add 15 words
        seg.add_chunk({"chunk_index": 1, "text": "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"})
        assert len(seg.emitted_segments) >= 1
        for s in seg.emitted_segments:
            assert s["word_count"] <= 10

    def test_9_maximum_character_limit_remains_enforced(self):
        config = StreamingConfig(max_segment_characters=40, min_segment_words=2, max_segment_words=10)
        seg = ImprovedStreamingSegmenter(config=config)
        seg.add_chunk({"chunk_index": 1, "text": "extraordinarily longwordcombination infrastructure modernization"})
        for s in seg.emitted_segments:
            assert s["character_count"] <= 40

    def test_11_end_of_stream_flush_works(self):
        seg = ImprovedStreamingSegmenter()
        seg.add_chunk({"chunk_index": 1, "text": "some buffered text waiting"})
        flushed = seg.flush(last_chunk_index=1)
        assert len(flushed) == 1
        assert flushed[0]["boundary_reason"] == "end_of_stream"
        assert flushed[0]["boundary_score"] == 100.0
        assert seg.buffer.is_empty()

    def test_12_streaming_behavior_remains_incremental(self):
        seg = ImprovedStreamingSegmenter()
        # Chunk 1: buffer small, no emission
        res1 = seg.add_chunk({"chunk_index": 1, "text": "the government"})
        assert res1 == []
        # Chunk 2: still buffering
        res2 = seg.add_chunk({"chunk_index": 2, "text": "announced today"})
        # Progresses incrementally
        assert seg.buffer.word_count() > 0

    def test_13_14_boundary_reason_and_score_recorded(self):
        config = StreamingConfig(min_segment_words=3)
        seg = ImprovedStreamingSegmenter(config=config)
        seg.add_chunk({"chunk_index": 1, "text": "good morning everyone welcome to today"})
        assert len(seg.emitted_segments) >= 1
        first_seg = seg.emitted_segments[0]
        assert "boundary_reason" in first_seg
        assert "boundary_score" in first_seg
        assert first_seg["boundary_reason"] in [
            "discourse_transition",
            "clause_boundary",
            "phrase_complete",
            "length_limit",
            "character_limit",
            "end_of_stream",
            "balanced_boundary",
        ]

    def test_15_16_17_word_order_zero_loss_zero_duplication(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "integrity_test")
        res = run_phase3_pipeline(sample_path, output_dir)
        
        assert res["integrity_result"]["status"] == "PASSED"
        assert res["integrity_result"]["exact_match"] is True
        assert res["integrity_result"]["original_word_count"] == res["integrity_result"]["reconstructed_word_count"]


class TestPhase3TranscriptsAndComparison:
    """Tests 18-22: Pipeline runs, comparison artifacts, limits, and safety."""

    def test_18_sample_transcript_passes(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "sample_out")
        res = run_phase3_pipeline(sample_path, output_dir)
        assert res["phase3_quality"]["segment_count"] > 0
        assert res["integrity_result"]["exact_match"] is True

    def test_19_edge_case_transcript_passes(self, tmp_path):
        edge_path = os.path.join(PROJECT_ROOT, "data", "input", "edge_case_transcript.txt")
        output_dir = str(tmp_path / "edge_out")
        res = run_phase3_pipeline(edge_path, output_dir)
        assert res["phase3_quality"]["segment_count"] > 0
        assert res["integrity_result"]["exact_match"] is True

    def test_20_phase3_comparison_is_generated(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "compare_out")
        res = run_phase3_pipeline(sample_path, output_dir)
        
        for f in res["output_files"]:
            assert os.path.exists(f)

    def test_21_hard_limit_violations_are_zero(self, tmp_path):
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        test_path = os.path.join(PROJECT_ROOT, "data", "input", "segmentation_test_transcript.txt")
        output_dir = str(tmp_path / "limits_out")
        
        res1 = run_phase3_pipeline(sample_path, output_dir)
        assert res1["phase3_quality"]["hard_limit_violations"] == 0

        res2 = run_phase3_pipeline(test_path, output_dir)
        assert res2["phase3_quality"]["hard_limit_violations"] == 0

    def test_22_missing_gemini_api_key_does_not_break_phase3(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        output_dir = str(tmp_path / "no_key_out")
        res = run_phase3_pipeline(sample_path, output_dir, enable_gemini=True)
        assert res["integrity_result"]["exact_match"] is True
        assert res["phase3_quality"]["segment_count"] > 0
