"""
Streaming Segmenter Module for Phase 2.

Implements an incremental, deterministic subtitle segmenter on top of a rolling
StreamingBuffer. Receives sequential text chunks and evaluates linguistic and
length constraints to emit completed subtitle segments in real time.

Reuses Phase 1 linguistic markers (discourse cues, conjunctions, verb presence)
while operating strictly as an online, streaming state machine.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.streaming_buffer import StreamingBuffer
from src.segmenter import (
    DISCOURSE_MARKERS,
    _has_verb,
    _matches_multi_word_marker,
    CONJUNCTIONS,
)


@dataclass
class StreamingConfig:
    """Configuration parameters for the streaming simulation pipeline."""
    chunk_size: int = 3
    max_segment_words: int = 14
    max_segment_characters: int = 80
    min_segment_words: int = 3
    target_words: int = 9
    simulated_delay: float = 0.0


class StreamingSegmenter:
    """
    Incremental streaming segmenter that processes incoming text chunks
    and emits completed subtitle segments.
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.buffer = StreamingBuffer()
        self.events: List[Dict[str, Any]] = []
        self.emitted_segments: List[Dict[str, Any]] = []
        self._next_segment_id = 1

    def add_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an incoming chunk event, append to buffer, and evaluate
        whether one or more subtitle segments can be finalized.
        
        Args:
            chunk: Chunk dict containing "text", "chunk_index", "timestamp", etc.
            
        Returns:
            List of newly finalized subtitle segments (may be empty).
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

        # 3. Check for finalized segments
        newly_emitted: List[Dict[str, Any]] = []
        
        while True:
            boundary_idx = self._find_segment_boundary()
            if boundary_idx is None:
                break
                
            # Finalize segment from buffer
            segment_text = self.buffer.finalize_segment(boundary_idx)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

            segment_event = {
                "event_type": "subtitle_emitted",
                "segment_id": self._next_segment_id,
                "text": segment_text,
                "word_count": len(segment_text.split()),
                "character_count": len(segment_text),
                "chunk_index": chunk_index,
                "timestamp": time.time(),
                "simulated_latency_ms": latency_ms,
            }

            self._next_segment_id += 1
            self.events.append(segment_event)
            self.emitted_segments.append(segment_event)
            newly_emitted.append(segment_event)

        return newly_emitted

    def flush(self, last_chunk_index: int = 0) -> List[Dict[str, Any]]:
        """
        Flush any remaining text in the buffer at the end of the stream.
        
        Args:
            last_chunk_index: The index of the final stream chunk.
            
        Returns:
            List containing the final emitted segment (if buffer had content).
        """
        start_time = time.perf_counter()
        if self.buffer.is_empty():
            return []

        remaining_text = self.buffer.flush()
        if not remaining_text.strip():
            return []

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        segment_event = {
            "event_type": "subtitle_emitted",
            "segment_id": self._next_segment_id,
            "text": remaining_text,
            "word_count": len(remaining_text.split()),
            "character_count": len(remaining_text),
            "chunk_index": last_chunk_index,
            "timestamp": time.time(),
            "simulated_latency_ms": latency_ms,
            "flushed_at_eos": True,
        }

        self._next_segment_id += 1
        self.events.append(segment_event)
        self.emitted_segments.append(segment_event)
        return [segment_event]

    def _find_segment_boundary(self) -> Optional[int]:
        """
        Inspect current buffer tokens and determine if a completed segment
        boundary exists.
        
        Returns:
            Number of tokens to finalize as a segment, or None if not ready.
        """
        tokens = self.buffer.get_tokens()
        num_tokens = len(tokens)
        
        if num_tokens < self.config.min_segment_words:
            return None

        # Check natural linguistic boundaries between min_segment_words and max_segment_words
        for i in range(self.config.min_segment_words, num_tokens):
            # Check 1: Multi-word discourse marker starting at index i
            if _matches_multi_word_marker(tokens, i):
                return i

            # Check 2: Single-word discourse marker starting at index i
            tok_lower = tokens[i].lower()
            if tok_lower in DISCOURSE_MARKERS:
                return i

            # Check 3: Conjunction boundary with verb presence
            if (
                i >= self.config.target_words
                and tok_lower in CONJUNCTIONS
                and _has_verb(tokens[:i])
            ):
                return i

            # Check 4: Character limit exceeded for prefix
            prefix_char_len = len(" ".join(tokens[:i]))
            if prefix_char_len >= self.config.max_segment_characters:
                return i

        # Check 5: Hard maximum word limit reached
        if num_tokens >= self.config.max_segment_words:
            # Force boundary at max_segment_words
            return self.config.max_segment_words

        # Check 6: Hard maximum character limit reached
        current_char_len = len(" ".join(tokens))
        if current_char_len >= self.config.max_segment_characters:
            return num_tokens

        return None
