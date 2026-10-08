"""
Phase 10 Evaluation & Benchmarking Automated Test Suite.

Verifies the 14 Phase 10 evaluation requirements:
1. WER calculation (Levenshtein distance, substitutions, deletions, insertions).
2. CER calculation (character distance).
3. Text normalization (lowercasing, punctuation stripping, whitespace normalization).
4. Speaker-label permutation handling (bijective mapping invariance).
5. Speaker assignment scoring (accuracy and DER calculation).
6. Boundary tolerance calculation (within ±tolerance window, avg error).
7. Phrase precision.
8. Phrase recall.
9. Phrase F1.
10. RTF calculation (total time / duration, None on 0 duration).
11. Benchmark result serialization (valid JSON loading and structure).
12. Deterministic benchmark output (identical inputs yield identical metrics).
13. Missing reference handling (graceful empty/zero handling).
14. Invalid evaluation input handling (null/empty inputs do not crash).
"""

import json
import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.scripts.metrics import (
    normalize_eval_text,
    compute_wer,
    compute_cer,
    compute_speaker_accuracy_with_permutation,
    compute_boundary_metrics,
    compute_phrase_metrics,
    compute_rtf,
)


def test_1_wer_calculation():
    """Requirement 1: Test Word Error Rate (WER) computation."""
    # Exact match
    res_exact = compute_wer("hello world", "hello world")
    assert res_exact["wer"] == 0.0
    assert res_exact["substitutions"] == 0
    assert res_exact["deletions"] == 0
    assert res_exact["insertions"] == 0
    assert res_exact["ref_words"] == 2

    # Substitution
    res_sub = compute_wer("hello world", "hello earth")
    assert res_sub["wer"] == 0.5
    assert res_sub["substitutions"] == 1

    # Deletion
    res_del = compute_wer("the quick brown fox", "the brown fox")
    assert res_del["wer"] == 0.25
    assert res_del["deletions"] == 1

    # Insertion
    res_ins = compute_wer("renewable energy", "the renewable energy")
    assert res_ins["wer"] == 0.5
    assert res_ins["insertions"] == 1


def test_2_cer_calculation():
    """Requirement 2: Test Character Error Rate (CER) computation."""
    res_exact = compute_cer("climate", "climate")
    assert res_exact["cer"] == 0.0

    res_diff = compute_cer("climate", "climat")
    assert res_diff["cer"] == round(1 / 7, 4)
    assert res_diff["distance"] == 1
    assert res_diff["ref_chars"] == 7


def test_3_text_normalization():
    """Requirement 3: Test evaluation text normalization."""
    raw = "  Today, We're DISCUSSING   Artificial-Intelligence!  "
    norm = normalize_eval_text(raw)
    assert norm == "today were discussing artificial intelligence"

    assert normalize_eval_text("") == ""
    assert normalize_eval_text(None) == ""


def test_4_speaker_label_permutation_handling():
    """Requirement 4: Model label permutation (Speaker 1 vs Speaker 2) evaluates correctly."""
    ref_segs = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "speaker": "Speaker 2"},
    ]
    # Inverted labels from model
    hyp_segs_inverted = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 2"},
        {"start": 2.0, "end": 4.0, "speaker": "Speaker 1"},
    ]
    res = compute_speaker_accuracy_with_permutation(ref_segs, hyp_segs_inverted, time_step=0.1)
    assert res["speaker_accuracy"] == 1.0
    assert res["der"] == 0.0
    assert res["best_mapping"] == {"Speaker 2": "Speaker 1", "Speaker 1": "Speaker 2"}


def test_5_speaker_assignment_scoring():
    """Requirement 5: Test partial speaker assignment accuracy and DER."""
    ref_segs = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "speaker": "Speaker 2"},
    ]
    # Hypothesis correctly labels first 2s, but incorrectly labels second 2s
    hyp_segs = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "speaker": "Speaker 1"},
    ]
    res = compute_speaker_accuracy_with_permutation(ref_segs, hyp_segs, time_step=0.1)
    # 50% accurate
    assert 0.45 <= res["speaker_accuracy"] <= 0.55
    assert 0.45 <= res["der"] <= 0.55


def test_6_boundary_tolerance_calculation():
    """Requirement 6: Subtitle boundary hit testing with timing tolerance."""
    ref_b = [5.0, 10.0, 15.0]
    # Hyp boundaries within ±0.5s: 5.2s (diff 0.2s), 9.8s (diff 0.2s), 16.0s (diff 1.0s -> missed)
    hyp_b = [5.2, 9.8, 16.0]

    res = compute_boundary_metrics(ref_b, hyp_b, tolerance_sec=0.50)
    assert res["num_ref_boundaries"] == 3
    assert res["matched_boundaries"] == 2
    assert res["boundary_accuracy"] == round(2 / 3, 4)
    assert res["average_boundary_error_sec"] == 0.20


def test_7_phrase_precision():
    """Requirement 7: Phrase Precision calculation."""
    expected = ["renewable energy transition", "clean energy infrastructure"]
    # Predicted 3 phrases, 2 are correct, 1 is false positive
    predicted = ["renewable energy transition", "clean energy infrastructure", "random phrase"]

    res = compute_phrase_metrics(expected, predicted)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["precision"] == round(2 / 3, 4)


def test_8_phrase_recall():
    """Requirement 8: Phrase Recall calculation."""
    expected = ["renewable energy transition", "clean energy infrastructure"]
    # Predicted only 1 of the 2
    predicted = ["renewable energy transition"]

    res = compute_phrase_metrics(expected, predicted)
    assert res["true_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["recall"] == 0.5


def test_9_phrase_f1():
    """Requirement 9: Phrase F1 calculation."""
    expected = ["renewable energy transition", "clean energy infrastructure"]
    predicted = ["renewable energy transition"]

    res = compute_phrase_metrics(expected, predicted)
    # P = 1.0, R = 0.5 -> F1 = 2 * 1.0 * 0.5 / 1.5 = 0.6667
    assert res["precision"] == 1.0
    assert res["recall"] == 0.5
    assert res["f1"] == 0.6667


def test_10_rtf_calculation():
    """Requirement 10: Real-Time Factor (RTF) calculation."""
    rtf_fast = compute_rtf(total_processing_time_sec=2.0, audio_duration_sec=20.0)
    assert rtf_fast == 0.10

    rtf_slow = compute_rtf(total_processing_time_sec=15.0, audio_duration_sec=10.0)
    assert rtf_slow == 1.50

    # 0 duration edge case
    assert compute_rtf(total_processing_time_sec=5.0, audio_duration_sec=0.0) is None


def test_11_benchmark_result_serialization():
    """Requirement 11: Benchmark JSON result file exists, is valid JSON, and has required keys."""
    res_path = os.path.join(PROJECT_ROOT, "evaluation", "results", "benchmark_results.json")
    assert os.path.exists(res_path), f"Benchmark JSON missing at {res_path}"

    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmark_metadata" in data
    assert "aggregate_metrics" in data
    assert "benchmarks" in data
    assert "robustness" in data
    assert "mean_wer" in data["aggregate_metrics"]
    assert "mean_rtf" in data["aggregate_metrics"]


def test_12_deterministic_benchmark_output():
    """Requirement 12: Benchmark metrics produce deterministic results on identical inputs."""
    ref = "Today we discuss artificial intelligence and clean energy infrastructure."
    hyp = "Today we discuss artificial intelligence and clean energy infrastructure."

    wer_1 = compute_wer(ref, hyp)
    wer_2 = compute_wer(ref, hyp)
    assert wer_1 == wer_2

    cer_1 = compute_cer(ref, hyp)
    cer_2 = compute_cer(ref, hyp)
    assert cer_1 == cer_2


def test_13_missing_reference_handling():
    """Requirement 13: Handle missing references without unhandled exceptions."""
    # Empty reference
    wer_res = compute_wer("", "hypothesis text")
    assert wer_res["ref_words"] == 0
    assert wer_res["hyp_words"] == 2

    # Empty phrase reference
    phrase_res = compute_phrase_metrics([], ["predicted phrase"])
    assert phrase_res["expected_count"] == 0
    assert phrase_res["precision"] == 0.0

    # Empty boundary reference
    boundary_res = compute_boundary_metrics([], [5.0, 10.0])
    assert boundary_res["num_ref_boundaries"] == 0
    assert boundary_res["matched_boundaries"] == 0


def test_14_invalid_evaluation_input_handling():
    """Requirement 14: Null or empty inputs are handled cleanly across all metric functions."""
    assert normalize_eval_text("") == ""
    assert compute_wer("", "")["wer"] == 0.0
    assert compute_cer("", "")["cer"] == 0.0
    assert compute_speaker_accuracy_with_permutation([], [])["speaker_accuracy"] == 1.0
    assert compute_boundary_metrics([], [])["boundary_accuracy"] == 1.0
    assert compute_phrase_metrics([], [])["f1"] == 1.0
    assert compute_rtf(0.0, 0.0) is None
