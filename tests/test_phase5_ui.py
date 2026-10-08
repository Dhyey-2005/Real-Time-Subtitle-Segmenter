"""
Tests for Phase 5 UI Support and Simulation Engine.

Validates:
1. UI and simulation session modules import cleanly
2. Existing Phase 1, Phase 2, Phase 3, and Phase 4 pipelines remain functional
3. Sample transcript end-to-end simulation processing
4. Availability of subtitle segments and Phase 4 phrase rankings in session state
5. Cross-segment phrase capture in UI session
6. Simulation lifecycle (start, step, reset)
7. Session export generation (JSON and CSV)
8. Empty and small transcript handling
9. Missing Gemini key fault-tolerance
10. OpenAI exclusion verification
11. Deterministic NLP scoring consistency
12. HTML phrase highlighting generation
"""

import os
import sys
import tempfile
import json
import csv
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce project-local NLTK data directory
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR):
    import nltk
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

from src.text_processor import load_text
from src.streaming_segmenter import StreamingConfig
from src.ui_session import StreamingSimulationSession, highlight_phrases_in_text
from src.gemini_client import GeminiSemanticClient


class TestPhase5UISessionAndSimulation:
    """Tests for UI session state management and simulation engine."""

    def test_01_modules_import_cleanly(self):
        """Verify UI session, highlight utilities, and app modules import cleanly."""
        import src.ui_session as ui_sess
        assert hasattr(ui_sess, "StreamingSimulationSession")
        assert hasattr(ui_sess, "highlight_phrases_in_text")

    def test_01b_existing_phase1_remains_functional(self):
        """Verify Phase 1 segmenter and text processor operate normally."""
        from src.segmenter import segment_text
        from src.text_processor import normalize_text
        norm = normalize_text("this is a test sentence and this is another clause")
        segs = segment_text(norm)
        assert len(segs) >= 1

    def test_01c_existing_phase2_remains_functional(self):
        """Verify Phase 2 streaming buffer operates normally."""
        from src.streaming_buffer import StreamingBuffer
        buf = StreamingBuffer()
        buf.append("hello world")
        assert buf.word_count() == 2
        popped = buf.finalize_segment(2)
        assert popped == "hello world"

    def test_01d_existing_phase3_remains_functional(self):
        """Verify Phase 3 improved segmenter operates normally."""
        from src.improved_segmenter import ImprovedStreamingSegmenter
        seg = ImprovedStreamingSegmenter()
        res = seg.add_chunk({"chunk_index": 1, "text": "hello world"})
        assert isinstance(res, list)

    def test_01e_existing_phase4_remains_functional(self):
        """Verify Phase 4 collocation scoring and ranking operate normally."""
        from src.collocation_scorer import calculate_pmi_bigram
        from src.phrase_ranker import analyze_and_rank_phrases
        from collections import Counter
        pmi = calculate_pmi_bigram(("word1", "word2"), Counter({("word1", "word2"): 2}), Counter({"word1": 2, "word2": 2}), 2, 4)
        assert pmi > 0.0
        analysis = analyze_and_rank_phrases(["renewable", "energy", "policy", "renewable", "energy"], min_frequency=2)
        assert "ranked_phrases" in analysis

    def test_02_sample_transcript_processing(self):
        """Verify the session processes sample_transcript.txt completely."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        raw_text = load_text(sample_path)
        sess = StreamingSimulationSession(transcript_text=raw_text, min_frequency=2, top_k=20)
        
        assert sess.total_chunks > 0
        summary = sess.run_all()
        
        assert sess.status == "COMPLETED"
        assert summary["emitted_segments_count"] > 0
        assert summary["word_loss"] == 0
        assert len(sess.emitted_segments) == summary["emitted_segments_count"]

    def test_03_subtitle_segments_available_to_ui(self):
        """Verify emitted subtitle segments retain all Phase 3 metadata for UI display."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        raw_text = load_text(sample_path)
        sess = StreamingSimulationSession(transcript_text=raw_text)
        sess.run_all()

        for seg in sess.emitted_segments:
            assert "segment_id" in seg
            assert "text" in seg
            assert "word_count" in seg
            assert "character_count" in seg
            assert "boundary_reason" in seg
            assert "boundary_score" in seg

    def test_04_phrase_extraction_available_to_ui(self):
        """Verify Phase 4 ranked phrases are available to the UI session."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        raw_text = load_text(sample_path)
        sess = StreamingSimulationSession(transcript_text=raw_text, min_frequency=2, top_k=20)
        sess.run_all()

        rankings = sess.current_rankings
        assert "ranked_phrases" in rankings
        assert len(rankings["ranked_phrases"]) > 0

        top_phrase = rankings["ranked_phrases"][0]
        assert "phrase" in top_phrase
        assert "quality_score" in top_phrase
        assert "quality_category" in top_phrase
        assert top_phrase["quality_category"] in ("HIGH", "MEDIUM", "LOW")

    def test_05_cross_segment_phrases_available(self):
        """Verify phrases spanning across subtitle boundaries are detected."""
        text = "we encourage renewable energy transition because renewable energy transition matters"
        sess = StreamingSimulationSession(transcript_text=text, min_frequency=2)
        sess.run_all()

        ranked = [p["phrase"] for p in sess.current_rankings.get("ranked_phrases", [])]
        assert any("renewable energy" in p or "energy transition" in p for p in ranked)

    def test_06_session_reset_logic_works(self):
        """Verify resetting session resets stream position and segments without losing text."""
        raw_text = "the government announced a new economic package today"
        sess = StreamingSimulationSession(transcript_text=raw_text)
        
        # Step through half
        sess.step()
        assert sess.chunk_index > 0
        assert sess.status == "PROCESSING"

        # Reset
        sess.reset()
        assert sess.chunk_index == 0
        assert sess.status == "READY"
        assert len(sess.emitted_segments) == 0
        assert sess.progressive_asr_text == ""
        assert len(sess.all_chunks) > 0

    def test_07_export_generation_works(self):
        """Verify export produces valid JSON and CSV artifacts."""
        raw_text = "the government announced a new economic package today and another new economic package tomorrow"
        sess = StreamingSimulationSession(transcript_text=raw_text, min_frequency=2)
        sess.run_all()

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir:
            paths = sess.export_session(output_dir=tmpdir)
            
            assert os.path.exists(paths["segments_json"])
            assert os.path.exists(paths["phrases_csv"])
            assert os.path.exists(paths["summary_json"])

            with open(paths["summary_json"], "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["status"] == "COMPLETED"
                assert data["emitted_segments_count"] > 0

    def test_08_empty_input_handled(self):
        """Verify empty transcript initializes safely without crashing."""
        sess = StreamingSimulationSession(transcript_text="")
        assert sess.total_chunks == 0
        res = sess.step()
        assert res["is_complete"] is True
        assert sess.status == "COMPLETED"

    def test_09_small_input_handled(self):
        """Verify single word input runs cleanly."""
        sess = StreamingSimulationSession(transcript_text="hello")
        assert sess.total_chunks == 1
        summary = sess.run_all()
        assert summary["status"] == "COMPLETED"
        assert summary["total_tokens"] == 1

    def test_10_missing_gemini_key_does_not_crash(self):
        """Verify Gemini client handles missing key gracefully for UI."""
        client = GeminiSemanticClient(api_key=None, use_env=False)
        assert client.is_available() is False
        cats = client.categorize_phrases(["economic package", "clean energy"])
        assert "economic package" in cats

    def test_11_openai_exclusion_verified(self):
        """Verify no OpenAI dependencies or imports exist in Phase 5."""
        import pkgutil
        installed = {m.name.lower() for m in pkgutil.iter_modules()}
        assert "openai" not in installed

        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "src")):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        assert "import openai" not in content
                        assert "from openai" not in content

    def test_12_deterministic_backend_results_consistent(self):
        """Verify running session twice on same transcript produces identical rankings."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        raw_text = load_text(sample_path)
        
        sess1 = StreamingSimulationSession(transcript_text=raw_text, min_frequency=2, top_k=10)
        sess1.run_all()
        ranks1 = [(p["rank"], p["phrase"], p["quality_score"]) for p in sess1.current_rankings["ranked_phrases"]]

        sess2 = StreamingSimulationSession(transcript_text=raw_text, min_frequency=2, top_k=10)
        sess2.run_all()
        ranks2 = [(p["rank"], p["phrase"], p["quality_score"]) for p in sess2.current_rankings["ranked_phrases"]]

        assert ranks1 == ranks2

    def test_13_phrase_highlighting_html(self):
        """Verify phrase highlighting wraps recurring phrases in styled HTML spans."""
        text = "the minister announced a new economic package today"
        phrases = [
            {"phrase": "new economic package", "quality_category": "HIGH"},
        ]
        res = highlight_phrases_in_text(text, phrases)
        assert "<span style=" in res
        assert "new economic package</span>" in res
