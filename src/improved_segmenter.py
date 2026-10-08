"""
Improved Streaming Segmenter Module for Phase 3.

Implements an explainable, boundary-scoring subtitle segmentation engine
with phrase protection, clause-structure awareness, and strict hard-limit enforcement.
"""

import time
from typing import List, Dict, Any, Optional, Tuple

from src.streaming_buffer import StreamingBuffer
from src.streaming_segmenter import StreamingConfig
from src.phrase_protection import PhraseProtector
from src.boundary_scorer import BoundaryScorer, BoundaryEvaluation


class ImprovedStreamingSegmenter:
    """
    Incremental streaming segmenter utilizing deterministic boundary scoring
    and multi-word phrase protection.
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        phrase_protector: Optional[PhraseProtector] = None,
        emission_threshold: float = 40.0,
    ):
        self.config = config or StreamingConfig()
        self.phrase_protector = phrase_protector or PhraseProtector()
        self.scorer = BoundaryScorer(
            min_words=self.config.min_segment_words,
            max_words=self.config.max_segment_words,
            max_characters=self.config.max_segment_characters,
            target_words=self.config.target_words,
            emission_threshold=emission_threshold,
            phrase_protector=self.phrase_protector,
        )
        self.buffer = StreamingBuffer()
        self.events: List[Dict[str, Any]] = []
        self.emitted_segments: List[Dict[str, Any]] = []
        self.boundary_decisions: List[Dict[str, Any]] = []
        self._next_segment_id = 1

    def add_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an incoming chunk, update the rolling buffer, and evaluate
        whether a completed subtitle boundary has been reached.
        
        Args:
            chunk: Chunk dict containing "text", "chunk_index", "timestamp", etc.
            
        Returns:
            List of finalized subtitle segments emitted in this step.
        """
        start_time = time.perf_counter()
        chunk_text = chunk.get("text", "")
        chunk_index = chunk.get("chunk_index", 0)

        # 1. Append tokens to rolling buffer
        self.buffer.append(chunk_text)

        # 2. Record chunk_received event
        self.events.append({
            "event_type": "chunk_received",
            "chunk_index": chunk_index,
            "text": chunk_text,
            "timestamp": chunk.get("timestamp", time.time()),
            "buffer_word_count": self.buffer.word_count(),
        })

        newly_emitted: List[Dict[str, Any]] = []

        # 3. Evaluate boundaries incrementally
        while True:
            decision = self._find_best_boundary()
            if decision is None:
                break

            split_idx, eval_result = decision
            segment_text = self.buffer.finalize_segment(split_idx)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

            segment_event = {
                "event_type": "subtitle_emitted",
                "segment_id": self._next_segment_id,
                "text": segment_text,
                "word_count": len(segment_text.split()),
                "character_count": len(segment_text),
                "boundary_reason": eval_result.reason,
                "boundary_score": eval_result.score,
                "trigger_word": eval_result.trigger_word,
                "chunk_index": chunk_index,
                "timestamp": time.time(),
                "simulated_latency_ms": latency_ms,
            }

            self._next_segment_id += 1
            self.events.append(segment_event)
            self.emitted_segments.append(segment_event)
            self.boundary_decisions.append({
                "segment_id": segment_event["segment_id"],
                "text": segment_text,
                "word_count": segment_event["word_count"],
                "character_count": segment_event["character_count"],
                "boundary_reason": eval_result.reason,
                "boundary_score": eval_result.score,
                "trigger_word": eval_result.trigger_word,
                "chunk_index": chunk_index,
                "latency_ms": latency_ms,
            })
            newly_emitted.append(segment_event)

        return newly_emitted

    def flush(self, last_chunk_index: int = 0) -> List[Dict[str, Any]]:
        """
        Flush remaining buffer tokens at end-of-stream while strictly
        enforcing maximum length and character limits.
        
        Args:
            last_chunk_index: Index of the final chunk.
            
        Returns:
            List containing any remaining emitted segments.
        """
        flushed_segments: List[Dict[str, Any]] = []

        # First, ensure any content exceeding limits is properly partitioned
        while True:
            decision = self._find_best_boundary()
            if decision is None:
                break

            split_idx, eval_result = decision
            segment_text = self.buffer.finalize_segment(split_idx)
            segment_event = {
                "event_type": "subtitle_emitted",
                "segment_id": self._next_segment_id,
                "text": segment_text,
                "word_count": len(segment_text.split()),
                "character_count": len(segment_text),
                "boundary_reason": eval_result.reason,
                "boundary_score": eval_result.score,
                "trigger_word": eval_result.trigger_word,
                "chunk_index": last_chunk_index,
                "timestamp": time.time(),
                "simulated_latency_ms": 0.05,
            }
            self._next_segment_id += 1
            self.events.append(segment_event)
            self.emitted_segments.append(segment_event)
            flushed_segments.append(segment_event)

        # Emit whatever remains as the final concluding segment
        if not self.buffer.is_empty():
            remaining_text = self.buffer.flush()
            if remaining_text.strip():
                segment_event = {
                    "event_type": "subtitle_emitted",
                    "segment_id": self._next_segment_id,
                    "text": remaining_text,
                    "word_count": len(remaining_text.split()),
                    "character_count": len(remaining_text),
                    "boundary_reason": "end_of_stream",
                    "boundary_score": 100.0,
                    "trigger_word": "eos",
                    "chunk_index": last_chunk_index,
                    "timestamp": time.time(),
                    "simulated_latency_ms": 0.05,
                    "flushed_at_eos": True,
                }
                self._next_segment_id += 1
                self.events.append(segment_event)
                self.emitted_segments.append(segment_event)
                flushed_segments.append(segment_event)

        return flushed_segments

    def _find_best_boundary(self) -> Optional[Tuple[int, BoundaryEvaluation]]:
        """
        Scan candidate split positions and return the optimal boundary if
        a boundary condition or hard limit is met.
        
        Returns:
            Tuple of (split_idx, BoundaryEvaluation) or None.
        """
        tokens = self.buffer.get_tokens()
        num_tokens = len(tokens)
        
        if num_tokens < self.config.min_segment_words:
            return None

        # 1. Evaluate natural boundaries that satisfy hard limits and don't break protected phrases
        candidates: List[BoundaryEvaluation] = []
        for i in range(self.config.min_segment_words, min(num_tokens, self.config.max_segment_words + 1)):
            prefix_char_len = len(" ".join(tokens[:i]))
            if prefix_char_len <= self.config.max_segment_characters:
                eval_res = self.scorer.evaluate_boundary(tokens, i)
                if eval_res.is_valid:
                    candidates.append(eval_res)

        if candidates:
            # Pick candidate with highest score
            best = max(candidates, key=lambda c: c.score)
            return (best.split_idx, best)

        # 2. Check if a hard limit (words or characters) is being approached or exceeded
        current_char_len = len(" ".join(tokens))
        needs_forced_split = (
            num_tokens >= self.config.max_segment_words
            or current_char_len >= self.config.max_segment_characters
        )

        if needs_forced_split:
            valid_forced: List[BoundaryEvaluation] = []
            
            # First pass: try candidates >= min_segment_words that do NOT break protected phrases
            for i in range(min(num_tokens, self.config.max_segment_words), max(1, self.config.min_segment_words - 1), -1):
                prefix_char_len = len(" ".join(tokens[:i]))
                if prefix_char_len <= self.config.max_segment_characters:
                    eval_res = self.scorer.evaluate_boundary(tokens, i)
                    if not self.phrase_protector.is_split_inside_phrase(tokens, i):
                        valid_forced.append(eval_res)
                        
            if valid_forced:
                # Pick the highest-scoring candidate among those that don't break phrases
                best = max(valid_forced, key=lambda c: c.score)
                reason = "character_limit" if current_char_len >= self.config.max_segment_characters else "length_limit"
                best.reason = reason
                return (best.split_idx, best)

            # Second pass: if all candidates break a phrase, pick the largest i with char_len <= max_chars
            for i in range(min(num_tokens, self.config.max_segment_words), 0, -1):
                prefix_char_len = len(" ".join(tokens[:i]))
                if prefix_char_len <= self.config.max_segment_characters:
                    eval_res = self.scorer.evaluate_boundary(tokens, i)
                    reason = "character_limit" if current_char_len >= self.config.max_segment_characters else "length_limit"
                    eval_res.reason = reason
                    return (i, eval_res)

            # If even a single word exceeds max_chars (extreme edge case):
            single_eval = self.scorer.evaluate_boundary(tokens, 1)
            single_eval.reason = "character_limit"
            return (1, single_eval)

        return None
