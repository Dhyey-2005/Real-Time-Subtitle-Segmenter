"""
Stream Simulator Module for Phase 2.

Simulates an incoming ASR text stream from an unpunctuated text transcript.
Supports word-by-word streaming, configurable word-chunk sizes, and optional
simulated real-time inter-chunk delays.
"""

import time
from typing import Generator, Dict, Any, List
from src.text_processor import normalize_text
from src.tokenizer import tokenize_text


class StreamSimulator:
    """
    Simulates an incoming stream of ASR text chunks from a source transcript.
    """

    def __init__(
        self,
        transcript_text: str,
        chunk_size: int = 3,
        delay: float = 0.0,
    ):
        """
        Initialize the stream simulator.
        
        Args:
            transcript_text: Raw or normalized input transcript.
            chunk_size: Number of words per emitted chunk (default: 3).
            delay: Simulated delay in seconds between emitted chunks (default: 0.0).
            
        Raises:
            ValueError: If chunk_size < 1 or delay < 0.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")

        self.chunk_size = chunk_size
        self.delay = delay
        
        # Normalize and tokenize the entire input to create pristine chunk units
        normalized = normalize_text(transcript_text)
        self.tokens: List[str] = tokenize_text(normalized)

    def total_words(self) -> int:
        """Return total word count of the transcript stream."""
        return len(self.tokens)

    def total_chunks(self) -> int:
        """Return total number of chunks that will be generated."""
        if not self.tokens:
            return 0
        return (len(self.tokens) + self.chunk_size - 1) // self.chunk_size

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator yielding simulated stream chunk events.
        
        Yields:
            Dict containing:
            - "event_type": "chunk_received"
            - "chunk_index": 1-based sequential index
            - "text": chunk text string
            - "tokens": list of word tokens
            - "word_count": number of words in this chunk
            - "timestamp": epoch timestamp when chunk was generated
        """
        if not self.tokens:
            return

        chunk_idx = 1
        for i in range(0, len(self.tokens), self.chunk_size):
            chunk_tokens = self.tokens[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            # Apply simulated delay if configured (non-zero)
            if self.delay > 0 and chunk_idx > 1:
                time.sleep(self.delay)

            yield {
                "event_type": "chunk_received",
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "tokens": chunk_tokens,
                "word_count": len(chunk_tokens),
                "timestamp": time.time(),
            }
            chunk_idx += 1
