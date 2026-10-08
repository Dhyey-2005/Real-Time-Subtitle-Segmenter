"""
Streaming Phrase Extractor Module for Phase 4.

Supports incremental online n-gram statistics accumulation as subtitle segments
are emitted, while preserving cross-segment continuity across the reconstructed
token stream for comprehensive post-stream ranking.
"""

import time
from collections import Counter
from typing import List, Dict, Any, Optional

from src.tokenizer import tokenize_text
from src.phrase_ranker import analyze_and_rank_phrases


class StreamingPhraseExtractor:
    """
    Online phrase statistics accumulator and extractor.
    """

    def __init__(self):
        self._accumulated_tokens: List[str] = []
        self._unigram_counts: Counter = Counter()
        self._bigram_counts: Counter = Counter()
        self._trigram_counts: Counter = Counter()
        self._incremental_latencies: List[float] = []
        self._segments_processed = 0

    def add_segment(self, segment_text: str) -> None:
        """
        Process an emitted subtitle segment incrementally.
        
        Args:
            segment_text: Clean subtitle segment string.
        """
        start_time = time.perf_counter()
        tokens = tokenize_text(segment_text)
        if not tokens:
            return

        # Update accumulated continuous token sequence
        prev_tail = self._accumulated_tokens[-2:] if len(self._accumulated_tokens) >= 2 else self._accumulated_tokens[:]
        self._accumulated_tokens.extend(tokens)
        self._segments_processed += 1

        # Online unigram counts
        self._unigram_counts.update(tokens)

        # Online intra-segment bigrams & trigrams
        for i in range(len(tokens) - 1):
            self._bigram_counts[tuple(tokens[i : i + 2])] += 1
        for i in range(len(tokens) - 2):
            self._trigram_counts[tuple(tokens[i : i + 3])] += 1

        # Cross-segment bigram/trigram transition linking
        if prev_tail:
            boundary_tokens = prev_tail + tokens[:2]
            if len(prev_tail) >= 1 and len(tokens) >= 1:
                # Cross-segment bigram
                self._bigram_counts[(prev_tail[-1], tokens[0])] += 1
            if len(prev_tail) >= 2 and len(tokens) >= 1:
                self._trigram_counts[(prev_tail[-2], prev_tail[-1], tokens[0])] += 1
            if len(prev_tail) >= 1 and len(tokens) >= 2:
                self._trigram_counts[(prev_tail[-1], tokens[0], tokens[1])] += 1

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        self._incremental_latencies.append(latency_ms)

    def get_incremental_stats(self) -> Dict[str, Any]:
        """Return summary of online streaming statistics."""
        avg_lat = (
            round(sum(self._incremental_latencies) / len(self._incremental_latencies), 3)
            if self._incremental_latencies
            else 0.0
        )
        return {
            "segments_processed": self._segments_processed,
            "total_tokens_accumulated": len(self._accumulated_tokens),
            "online_unique_unigrams": len(self._unigram_counts),
            "online_unique_bigrams": len(self._bigram_counts),
            "online_unique_trigrams": len(self._trigram_counts),
            "avg_incremental_update_ms": avg_lat,
        }

    def finalize_ranking(
        self,
        min_frequency: int = 2,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute final deterministic phrase ranking over the continuous token stream.
        
        Args:
            min_frequency: Minimum frequency threshold.
            top_k: Optional limit on top ranked phrases.
            
        Returns:
            Dict containing complete analysis and ranked phrases.
        """
        ranking_start = time.perf_counter()
        results = analyze_and_rank_phrases(
            tokens=self._accumulated_tokens,
            min_frequency=min_frequency,
            top_k=top_k,
        )
        ranking_duration_ms = round((time.perf_counter() - ranking_start) * 1000, 3)

        results["incremental_stats"] = self.get_incremental_stats()
        results["final_ranking_duration_ms"] = ranking_duration_ms
        return results
