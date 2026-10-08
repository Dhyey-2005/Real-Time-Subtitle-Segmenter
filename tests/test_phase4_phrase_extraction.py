"""
Tests for Phase 4 Improved N-gram Phrase Extraction.

Validates all 26 Phase 4 requirements:
- Preservation of bigram/trigram generation and frequency counting
- Stopword penalties and content-word prioritization
- POS tag pattern scoring and content word ratio
- Safe PMI and Tri-PMI association calculations (zero division, zero probability)
- Configurable minimum frequency filtering
- Quality categories (HIGH, MEDIUM, LOW)
- Case normalization and overlapping/parent-phrase detection
- Continuous cross-segment boundary phrase capture
- Online streaming incremental statistics accumulation
- Complete pipeline determinism
- Robustness on empty, single-word, and edge-case transcripts
- Missing Gemini key fault-tolerance
"""

import os
import sys
import tempfile
import pytest
from collections import Counter

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

from src.ngram_extractor import extract_bigrams, extract_trigrams, generate_ngrams
from src.collocation_scorer import calculate_pmi_bigram, calculate_pmi_trigram
from src.phrase_ranker import (
    compute_content_word_ratio,
    assign_quality_category,
    score_phrase_candidate,
    analyze_and_rank_phrases,
)
from src.streaming_phrase_extractor import StreamingPhraseExtractor
from src.phrase_orchestrator import run_phase4_pipeline
from src.streaming_segmenter import StreamingConfig


class TestPhase4NgramBasics:
    """Verifies existing n-gram extraction and frequency counting are preserved."""

    def test_bigram_generation_still_works(self):
        tokens = ["renewable", "energy", "transition", "policy"]
        bigrams = extract_bigrams(tokens)
        assert bigrams[("renewable", "energy")] == 1
        assert bigrams[("energy", "transition")] == 1
        assert bigrams[("transition", "policy")] == 1

    def test_trigram_generation_still_works(self):
        tokens = ["artificial", "intelligence", "infrastructure", "growth"]
        trigrams = extract_trigrams(tokens)
        assert trigrams[("artificial", "intelligence", "infrastructure")] == 1
        assert trigrams[("intelligence", "infrastructure", "growth")] == 1

    def test_frequency_counting_works(self):
        tokens = ["economic", "growth", "requires", "economic", "growth", "strategy"]
        bigrams = extract_bigrams(tokens)
        assert bigrams[("economic", "growth")] == 2
        assert bigrams[("growth", "requires")] == 1


class TestPhase4CollocationAndPMI:
    """Tests deterministic PMI and Tri-PMI association calculations with numerical safety."""

    def test_pmi_calculation_works(self):
        bigram = ("renewable", "energy")
        bigram_counts = Counter({bigram: 4, ("other", "words"): 10})
        unigram_counts = Counter({"renewable": 4, "energy": 5, "other": 10, "words": 10})
        total_bigrams = 14
        total_unigrams = 29

        pmi = calculate_pmi_bigram(bigram, bigram_counts, unigram_counts, total_bigrams, total_unigrams)
        assert isinstance(pmi, float)
        assert pmi > 0.0

    def test_pmi_handles_zero_probabilities_safely(self):
        bigram = ("nonexistent", "phrase")
        bigram_counts = Counter()
        unigram_counts = Counter({"word": 5})
        pmi = calculate_pmi_bigram(bigram, bigram_counts, unigram_counts, 10, 20)
        assert pmi == 0.0

    def test_pmi_does_not_divide_by_zero(self):
        # Empty totals
        pmi_bg = calculate_pmi_bigram(("word1", "word2"), Counter(), Counter(), 0, 0)
        assert pmi_bg == 0.0

        pmi_tg = calculate_pmi_trigram(("w1", "w2", "w3"), Counter(), Counter(), 0, 0)
        assert pmi_tg == 0.0

    def test_trigram_pmi_calculation_works(self):
        trigram = ("clean", "energy", "transition")
        trigram_counts = Counter({trigram: 3})
        unigram_counts = Counter({"clean": 4, "energy": 5, "transition": 4})
        pmi_tg = calculate_pmi_trigram(trigram, trigram_counts, unigram_counts, 10, 30)
        assert isinstance(pmi_tg, float)
        assert pmi_tg > 0.0


class TestPhase4PhraseScoringAndCategories:
    """Tests content-word ratio, stopword penalties, POS scoring, and quality categories."""

    def test_stopword_heavy_phrases_penalized(self):
        stopword_tuple = ("of", "the")
        tagged_stopword = [("of", "IN"), ("the", "DT")]
        score_stop, _, _ = score_phrase_candidate(stopword_tuple, 5, 0.5, tagged_stopword)

        content_tuple = ("renewable", "energy")
        tagged_content = [("renewable", "JJ"), ("energy", "NN")]
        score_content, _, _ = score_phrase_candidate(content_tuple, 5, 2.5, tagged_content)

        assert score_content > score_stop
        assert score_stop < 2.0  # heavily penalized

    def test_content_word_phrases_receive_better_scores(self):
        tokens = [
            "the", "minister", "discussed", "clean", "energy", "transition",
            "and", "the", "clean", "energy", "transition", "plan",
            "for", "the", "clean", "energy", "transition",
        ]
        results = analyze_and_rank_phrases(tokens, min_frequency=2)
        phrases = {p["phrase"]: p for p in results["ranked_phrases"]}

        assert "clean energy transition" in phrases
        assert phrases["clean energy transition"]["quality_score"] > 6.0
        assert phrases["clean energy transition"]["quality_category"] in ("HIGH", "MEDIUM")

    def test_pos_information_generated(self):
        tagged = [("artificial", "JJ"), ("intelligence", "NN")]
        ratio = compute_content_word_ratio(tagged)
        assert ratio == 1.0

        score, pos_pat, c_ratio = score_phrase_candidate(("artificial", "intelligence"), 3, 3.0, tagged)
        assert pos_pat == "JJ-NN"
        assert c_ratio == 1.0

    def test_quality_categories_thresholds(self):
        assert assign_quality_category(7.5) == "HIGH"
        assert assign_quality_category(6.5) == "HIGH"
        assert assign_quality_category(6.4) == "MEDIUM"
        assert assign_quality_category(4.0) == "MEDIUM"
        assert assign_quality_category(3.9) == "LOW"
        assert assign_quality_category(-1.0) == "LOW"


class TestPhase4RankingAndFiltering:
    """Tests minimum frequency filtering, ranking order, and normalization."""

    def test_minimum_frequency_filter_works(self):
        tokens = ["rare", "event", "happens", "once", "common", "word", "and", "common", "word"]
        results_min2 = analyze_and_rank_phrases(tokens, min_frequency=2)
        phrases_min2 = [p["phrase"] for p in results_min2["ranked_phrases"]]

        assert "common word" in phrases_min2
        assert "rare event" not in phrases_min2

        results_min1 = analyze_and_rank_phrases(tokens, min_frequency=1)
        phrases_min1 = [p["phrase"] for p in results_min1["ranked_phrases"]]
        assert "rare event" in phrases_min1

    def test_recurring_phrase_definition_works(self):
        tokens = ["clean", "water", "access", "requires", "clean", "water", "infrastructure"]
        results = analyze_and_rank_phrases(tokens, min_frequency=2)
        assert results["recurring_bigrams"] >= 1
        recurring_phrases = [p["phrase"] for p in results["ranked_phrases"]]
        assert "clean water" in recurring_phrases

    def test_case_normalization_works(self):
        from src.text_processor import normalize_text
        from src.tokenizer import tokenize_text
        raw_text = "Artificial Intelligence is key and ARTIFICIAL INTELLIGENCE is powerful and artificial intelligence works"
        tokens = tokenize_text(normalize_text(raw_text))
        results = analyze_and_rank_phrases(tokens, min_frequency=2)
        phrases = [p["phrase"] for p in results["ranked_phrases"]]
        assert "artificial intelligence" in phrases
        # Verify count is merged to 3
        ai_phrase = next(p for p in results["ranked_phrases"] if p["phrase"] == "artificial intelligence")
        assert ai_phrase["frequency"] == 3

    def test_phrase_ranking_deterministic_order(self):
        tokens = [
            "economic", "growth", "leads", "to", "economic", "growth",
            "digital", "economy", "fuels", "digital", "economy", "growth",
            "digital", "economy",
        ]
        res1 = analyze_and_rank_phrases(tokens, min_frequency=2)
        res2 = analyze_and_rank_phrases(tokens, min_frequency=2)

        ranks1 = [(p["rank"], p["phrase"], p["quality_score"]) for p in res1["ranked_phrases"]]
        ranks2 = [(p["rank"], p["phrase"], p["quality_score"]) for p in res2["ranked_phrases"]]
        assert ranks1 == ranks2

    def test_overlapping_phrases_handled(self):
        tokens = [
            "artificial", "intelligence", "infrastructure", "expands",
            "with", "artificial", "intelligence", "infrastructure",
        ]
        results = analyze_and_rank_phrases(tokens, min_frequency=2)
        phrases = {p["phrase"]: p for p in results["ranked_phrases"]}

        # Both the trigram and the bigram occur twice
        assert "artificial intelligence infrastructure" in phrases
        assert "artificial intelligence" in phrases
        # Bigram should indicate the trigram parent
        assert phrases["artificial intelligence"]["parent_phrase"] == "artificial intelligence infrastructure"


class TestPhase4StreamingAndCrossSegment:
    """Tests incremental online statistics and cross-segment phrase preservation."""

    def test_incremental_ngram_statistics_work(self):
        extractor = StreamingPhraseExtractor()
        extractor.add_segment("the government announced")
        stats1 = extractor.get_incremental_stats()
        assert stats1["segments_processed"] == 1
        assert stats1["online_unique_bigrams"] == 2

        extractor.add_segment("a new economic package")
        stats2 = extractor.get_incremental_stats()
        assert stats2["segments_processed"] == 2
        assert stats2["online_unique_bigrams"] > stats1["online_unique_bigrams"]

    def test_cross_segment_phrases_detected(self):
        extractor = StreamingPhraseExtractor()
        # "renewable energy" is at end of seg 1, "transition is accelerating" at start of seg 2
        extractor.add_segment("we support renewable energy")
        extractor.add_segment("transition is accelerating rapidly")

        # The cross-segment transition must be captured
        bigrams = extractor._bigram_counts
        trigrams = extractor._trigram_counts

        assert bigrams[("energy", "transition")] >= 1
        assert trigrams[("renewable", "energy", "transition")] >= 1


class TestPhase4EdgeCasesAndPipeline:
    """Tests edge cases, missing API keys, and pipeline execution on transcripts."""

    def test_empty_input_handled_safely(self):
        results = analyze_and_rank_phrases([], min_frequency=2)
        assert results["ranked_phrases"] == []
        assert results["total_unigrams"] == 0

    def test_very_small_input_handled_safely(self):
        results = analyze_and_rank_phrases(["hello"], min_frequency=2)
        assert results["ranked_phrases"] == []
        assert results["total_unigrams"] == 1

        results2 = analyze_and_rank_phrases(["hello", "world"], min_frequency=2)
        assert results2["ranked_phrases"] == []

    def test_pipeline_on_sample_transcript(self):
        input_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir:
            res = run_phase4_pipeline(
                input_path=input_path,
                output_dir=tmpdir,
                min_frequency=2,
                top_k=10,
                enable_gemini=False,
            )
            assert res["integrity_result"]["status"] == "PASSED"
            assert os.path.exists(res["paths"]["bigram_csv"])
            assert os.path.exists(res["paths"]["trigram_csv"])
            assert os.path.exists(res["paths"]["ranked_csv"])
            assert os.path.exists(res["paths"]["json"])
            assert os.path.exists(res["paths"]["report"])

    def test_pipeline_on_edge_case_transcript(self):
        input_path = os.path.join(PROJECT_ROOT, "data", "input", "edge_case_transcript.txt")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir:
            res = run_phase4_pipeline(
                input_path=input_path,
                output_dir=tmpdir,
                min_frequency=2,
                top_k=10,
                enable_gemini=False,
            )
            assert res["integrity_result"]["status"] == "PASSED"

    def test_pipeline_on_phrase_extraction_test_transcript(self):
        input_path = os.path.join(PROJECT_ROOT, "data", "input", "phrase_extraction_test_transcript.txt")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir:
            res = run_phase4_pipeline(
                input_path=input_path,
                output_dir=tmpdir,
                min_frequency=2,
                top_k=20,
                enable_gemini=False,
            )
            assert res["integrity_result"]["status"] == "PASSED"
            ranked = res["ranking_results"]["ranked_phrases"]
            top_texts = [p["phrase"] for p in ranked[:10]]

            # Ensure high-value repeated content phrases appear near top
            assert any("energy" in t or "intelligence" in t or "economic" in t for t in top_texts)

    def test_missing_gemini_api_key_does_not_break_phrase_extraction(self):
        # Even with enable_gemini=True, missing key should gracefully degrade
        input_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir:
            res = run_phase4_pipeline(
                input_path=input_path,
                output_dir=tmpdir,
                min_frequency=2,
                top_k=5,
                enable_gemini=True,
            )
            assert res["integrity_result"]["status"] == "PASSED"
            assert "gemini_status" in res["stats"]

    def test_pipeline_determinism_across_runs(self):
        input_path = os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir1, tempfile.TemporaryDirectory(dir=PROJECT_ROOT, ignore_cleanup_errors=True) as tmpdir2:
            res1 = run_phase4_pipeline(input_path=input_path, output_dir=tmpdir1, min_frequency=2, top_k=10, enable_gemini=False)
            res2 = run_phase4_pipeline(input_path=input_path, output_dir=tmpdir2, min_frequency=2, top_k=10, enable_gemini=False)

            phrases1 = [(p["rank"], p["phrase"], p["quality_score"], p["quality_category"]) for p in res1["ranking_results"]["ranked_phrases"]]
            phrases2 = [(p["rank"], p["phrase"], p["quality_score"], p["quality_category"]) for p in res2["ranking_results"]["ranked_phrases"]]
            assert phrases1 == phrases2

    def test_openai_exclusion_verified(self):
        import pkgutil
        installed_modules = {mod.name.lower() for mod in pkgutil.iter_modules()}
        assert "openai" not in installed_modules
        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "src")):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        assert "import openai" not in content
                        assert "from openai" not in content
