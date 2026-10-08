"""
Streaming Buffer Module for Phase 2.

Maintains an incremental rolling text buffer for simulated real-time ASR text.
Preserves word order, prevents word loss or duplication, and supports
inspecting, finalizing, and flushing buffered content.
"""

from typing import List
from src.tokenizer import tokenize_text


class StreamingBuffer:
    """
    A rolling FIFO token buffer for incremental stream processing.
    """

    def __init__(self):
        self._tokens: List[str] = []

    def append(self, text: str) -> None:
        """
        Append text to the buffer.
        
        Args:
            text: Incoming text chunk from the stream.
        """
        if not text:
            return
        new_tokens = tokenize_text(text)
        self._tokens.extend(new_tokens)

    def append_tokens(self, tokens: List[str]) -> None:
        """
        Append pre-tokenized words directly to the buffer.
        
        Args:
            tokens: List of word tokens.
        """
        if tokens:
            self._tokens.extend(tokens)

    def get_tokens(self) -> List[str]:
        """Return a copy of the current tokens in the buffer."""
        return list(self._tokens)

    def get_text(self) -> str:
        """Return the current buffered tokens joined into a single string."""
        return " ".join(self._tokens)

    def word_count(self) -> int:
        """Return the number of words currently in the buffer."""
        return len(self._tokens)

    def char_count(self) -> int:
        """Return the character count of the buffered text."""
        return len(self.get_text())

    def finalize_segment(self, num_tokens: int) -> str:
        """
        Finalize and remove the first `num_tokens` tokens from the buffer.
        The remaining tokens stay in the buffer for the next subtitle segment.
        
        Args:
            num_tokens: Number of tokens to finalize.
            
        Returns:
            The finalized text segment.
            
        Raises:
            ValueError: If num_tokens <= 0 or num_tokens > current word count.
        """
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if num_tokens > len(self._tokens):
            raise ValueError(
                f"Cannot finalize {num_tokens} tokens when buffer only has {len(self._tokens)}"
            )

        finalized = self._tokens[:num_tokens]
        self._tokens = self._tokens[num_tokens:]
        return " ".join(finalized)

    def flush(self) -> str:
        """
        Finalize and clear all remaining tokens in the buffer.
        
        Returns:
            Remaining text string, or empty string if buffer was empty.
        """
        if not self._tokens:
            return ""
        text = " ".join(self._tokens)
        self._tokens.clear()
        return text

    def clear(self) -> None:
        """Completely clear the buffer."""
        self._tokens.clear()

    def is_empty(self) -> bool:
        """Check whether the buffer is empty."""
        return len(self._tokens) == 0

    def __len__(self) -> int:
        """Return the word count of the buffer."""
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"<StreamingBuffer(tokens={len(self._tokens)}, text='{self.get_text()}')>"
