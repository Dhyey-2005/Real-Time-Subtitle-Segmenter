"""
Evaluation Metrics Module for Phase 10.

Implements standard, deterministic evaluation metrics:
- Word Error Rate (WER) via dynamic programming Levenshtein distance on words
- Character Error Rate (CER) via dynamic programming Levenshtein distance on characters
- Text normalization for fair evaluation
- Speaker label permutation handling & optimal assignment scoring
- Subtitle boundary accuracy and temporal error calculation
- Phrase extraction Precision, Recall, and F1
- Real-Time Factor (RTF)
"""

import itertools
import re
import string
from typing import List, Dict, Any, Tuple, Optional, Set


def normalize_eval_text(text: str) -> str:
    """
    Standardize text for objective evaluation.
    Converts to lowercase, removes punctuation, collapses whitespace.
    """
    if not text:
        return ""
    # Lowercase
    t = text.lower()
    # Replace hyphens with spaces to match word boundaries
    t = t.replace("-", " ")
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> Tuple[int, int, int, int]:
    """
    Calculate Levenshtein edit distance between two sequences.
    
    Returns:
        Tuple of (substitutions, deletions, insertions, total_distance)
    """
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # Substitution
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                )

    # Backtrace to count individual operations
    i, j = n, m
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    total_dist = dp[n][m]
    return subs, dels, ins, total_dist


def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Compute Word Error Rate (WER).
    
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    ref_norm = normalize_eval_text(reference)
    hyp_norm = normalize_eval_text(hypothesis)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    if not ref_words and not hyp_words:
        return {
            "wer": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "ref_words": 0,
            "hyp_words": 0,
            "correct_words": 0,
        }
    if not ref_words:
        return {
            "wer": float(len(hyp_words)),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "correct_words": 0,
        }

    subs, dels, ins, dist = levenshtein_distance(ref_words, hyp_words)
    wer = round(dist / len(ref_words), 4)
    hits = max(0, len(ref_words) - (subs + dels))

    return {
        "wer": wer,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "correct_words": hits,
    }


def compute_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Compute Character Error Rate (CER).
    
    CER = Levenshtein Distance / Total Reference Characters
    """
    ref_norm = normalize_eval_text(reference)
    hyp_norm = normalize_eval_text(hypothesis)

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if not ref_chars and not hyp_chars:
        return {
            "cer": 0.0,
            "distance": 0,
            "ref_chars": 0,
            "hyp_chars": 0,
        }
    if not ref_chars:
        return {
            "cer": float(len(hyp_chars)),
            "distance": len(hyp_chars),
            "ref_chars": 0,
            "hyp_chars": len(hyp_chars),
        }

    _, _, _, dist = levenshtein_distance(ref_chars, hyp_chars)
    cer = round(dist / len(ref_chars), 4)

    return {
        "cer": cer,
        "distance": dist,
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
    }


def compute_speaker_accuracy_with_permutation(
    ref_segments: List[Dict[str, Any]],
    hyp_segments: List[Dict[str, Any]],
    time_step: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluate speaker diarization accuracy accounting for anonymous speaker label permutation.
    Discretizes the timeline into frame slices of length `time_step` seconds.
    Tests all bijective mappings between predicted and reference speaker labels to find optimal alignment.
    
    Returns:
        Dict with optimal speaker assignment accuracy, DER approximation, and label mapping.
    """
    if not ref_segments:
        return {
            "speaker_accuracy": 1.0 if not hyp_segments else 0.0,
            "der": 0.0 if not hyp_segments else 1.0,
            "best_mapping": {},
            "total_evaluated_duration_sec": 0.0,
        }

    max_time = max(
        max(s.get("end_time", s.get("end", 0.0)) for s in ref_segments),
        max((s.get("end_time", s.get("end", 0.0)) for s in hyp_segments), default=0.0),
    )
    if max_time <= 0.0:
        return {
            "speaker_accuracy": 1.0,
            "der": 0.0,
            "best_mapping": {},
            "total_evaluated_duration_sec": 0.0,
        }

    num_frames = int(max_time / time_step) + 1
    ref_frames: List[Optional[str]] = [None] * num_frames
    hyp_frames: List[Optional[str]] = [None] * num_frames

    for seg in ref_segments:
        st = seg.get("start_time", seg.get("start", 0.0))
        et = seg.get("end_time", seg.get("end", 0.0))
        spk = seg.get("speaker", "Speaker 1")
        i_start = max(0, int(st / time_step))
        i_end = min(num_frames, int(et / time_step))
        for i in range(i_start, i_end):
            ref_frames[i] = spk

    for seg in hyp_segments:
        st = seg.get("start_time", seg.get("start", 0.0))
        et = seg.get("end_time", seg.get("end", 0.0))
        spk = seg.get("speaker", "Speaker 1")
        i_start = max(0, int(st / time_step))
        i_end = min(num_frames, int(et / time_step))
        for i in range(i_start, i_end):
            hyp_frames[i] = spk

    ref_speakers = sorted(list({s for s in ref_frames if s is not None}))
    hyp_speakers = sorted(list({s for s in hyp_frames if s is not None}))

    # If no speech in reference
    speech_frame_indices = [i for i, r in enumerate(ref_frames) if r is not None]
    if not speech_frame_indices:
        return {
            "speaker_accuracy": 1.0,
            "der": 0.0,
            "best_mapping": {},
            "total_evaluated_duration_sec": 0.0,
        }

    # Evaluate label permutations
    best_accuracy = 0.0
    best_mapping = {}

    all_permutations = list(itertools.permutations(hyp_speakers, len(ref_speakers))) if len(hyp_speakers) >= len(ref_speakers) else []
    if not all_permutations:
        # Fewer hyp speakers than ref speakers: pad with dummy labels
        padded_hyp = hyp_speakers + [f"Unassigned_{k}" for k in range(len(ref_speakers) - len(hyp_speakers))]
        all_permutations = list(itertools.permutations(padded_hyp, len(ref_speakers)))

    for perm in all_permutations:
        mapping = {hyp_s: ref_s for hyp_s, ref_s in zip(perm, ref_speakers)}
        correct = 0
        for idx in speech_frame_indices:
            h_spk = hyp_frames[idx]
            r_spk = ref_frames[idx]
            mapped_h = mapping.get(h_spk, h_spk)
            if mapped_h == r_spk:
                correct += 1
        acc = correct / len(speech_frame_indices)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_mapping = mapping

    der = round(max(0.0, 1.0 - best_accuracy), 4)

    return {
        "speaker_accuracy": round(best_accuracy, 4),
        "der": der,
        "best_mapping": best_mapping,
        "total_evaluated_duration_sec": round(len(speech_frame_indices) * time_step, 2),
    }


def compute_boundary_metrics(
    ref_boundaries: List[float],
    hyp_boundaries: List[float],
    tolerance_sec: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate subtitle boundary alignment against ground truth with temporal tolerance.
    
    Args:
        ref_boundaries: List of reference boundary timestamps in seconds.
        hyp_boundaries: List of predicted boundary timestamps in seconds.
        tolerance_sec: Allowed error window in seconds (default: ±0.5s).
        
    Returns:
        Dict with boundary accuracy, average boundary error, and matches.
    """
    if not ref_boundaries:
        return {
            "num_ref_boundaries": 0,
            "num_hyp_boundaries": len(hyp_boundaries),
            "matched_boundaries": 0,
            "boundary_accuracy": 1.0 if not hyp_boundaries else 0.0,
            "average_boundary_error_sec": 0.0,
        }

    matched = 0
    errors = []
    used_hyp = set()

    for ref_b in ref_boundaries:
        best_diff = None
        best_idx = None
        for idx, hyp_b in enumerate(hyp_boundaries):
            if idx in used_hyp:
                continue
            diff = abs(hyp_b - ref_b)
            if diff <= tolerance_sec:
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = idx
        if best_idx is not None:
            matched += 1
            errors.append(best_diff)
            used_hyp.add(best_idx)

    accuracy = round(matched / len(ref_boundaries), 4)
    avg_error = round(sum(errors) / len(errors), 4) if errors else 0.0

    return {
        "num_ref_boundaries": len(ref_boundaries),
        "num_hyp_boundaries": len(hyp_boundaries),
        "matched_boundaries": matched,
        "boundary_accuracy": accuracy,
        "average_boundary_error_sec": avg_error,
        "tolerance_sec": tolerance_sec,
    }


def compute_phrase_metrics(
    expected_phrases: List[str],
    predicted_phrases: List[str],
) -> Dict[str, Any]:
    """
    Evaluate precision, recall, and F1 of extracted recurring phrases against expected phrases.
    """
    exp_set = {normalize_eval_text(p) for p in expected_phrases if normalize_eval_text(p)}
    pred_set = {normalize_eval_text(p) for p in predicted_phrases if normalize_eval_text(p)}

    if not exp_set and not pred_set:
        return {
            "expected_count": 0,
            "predicted_count": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "expected_phrases_found": [],
            "expected_phrases_missed": [],
        }

    tp_set = exp_set.intersection(pred_set)
    fp_set = pred_set.difference(exp_set)
    fn_set = exp_set.difference(pred_set)

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else (1.0 if not exp_set else 0.0)
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else (1.0 if not pred_set else 0.0)
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "expected_count": len(exp_set),
        "predicted_count": len(pred_set),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expected_phrases_found": sorted(list(tp_set)),
        "expected_phrases_missed": sorted(list(fn_set)),
    }


def compute_rtf(total_processing_time_sec: float, audio_duration_sec: float) -> Optional[float]:
    """
    Calculate Real-Time Factor (RTF).
    RTF = total_processing_time / audio_duration
    RTF < 1.0 indicates faster-than-playback processing.
    """
    if audio_duration_sec <= 0.0:
        return None
    return round(total_processing_time_sec / audio_duration_sec, 4)
