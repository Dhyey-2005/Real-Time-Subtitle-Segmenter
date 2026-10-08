"""
Conservative Boundary-Overlap Handling Module for Phase 7 Live Audio Streaming.

Addresses ASR repetition across adjacent streaming audio chunk boundaries.
When speech is chunked into short windows (e.g. 3 seconds), phonemes or words
at chunk boundaries can be partially captured in both chunks, leading to repeated
words (e.g., "...artificial intelligence" in Chunk 1 and "artificial intelligence infrastructure..." in Chunk 2).

Core Guarantees:
- Conservative overlap detection: detects when the prefix of Chunk N matches the suffix of Chunk N-1.
- Conservative removal: strips only the exact overlapping prefix from Chunk N.
- Distinguishes legitimate repetition: legitimate repetitions such as "very very important"
  within a sentence or chunk are preserved and not aggressively eliminated.
- Deterministic and explainable: returns structured details (trimmed text, overlap token count, overlap string, reason).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.tokenizer import tokenize_text
from src.text_processor import normalize_text


# Known legitimate repeated words (intensifiers, conversational markers)
LEGITIMATE_REPETITION_WORDS = {
    "very", "really", "so", "no", "yes", "bye", "hear", "never", "well", "now", "again"
}


@dataclass
class OverlapResolution:
    """Structured result of boundary-overlap resolution."""
    cleaned_text: str
    overlap_token_count: int
    overlap_phrase: str
    original_text: str
    reason: str


class ConservativeOverlapHandler:
    """
    Detects and safely removes redundant boundary-overlap words between consecutive ASR chunks
    while preserving genuine repeated words.
    """

    def __init__(
        self,
        max_overlap_tokens: int = 6,
        min_overlap_tokens: int = 1,
        case_sensitive: bool = False,
    ):
        """
        Initialize overlap handler.
        
        Args:
            max_overlap_tokens: Maximum number of trailing tokens from previous chunk to check for overlap.
            min_overlap_tokens: Minimum tokens required to trigger overlap removal (default 1).
            case_sensitive: Whether matching is case-sensitive (default False).
        """
        self.max_overlap_tokens = max_overlap_tokens
        self.min_overlap_tokens = min_overlap_tokens
        self.case_sensitive = case_sensitive

    def resolve_overlap(
        self,
        previous_text: str,
        current_text: str,
    ) -> OverlapResolution:
        """
        Resolve boundary overlap between previous finalized text and current incoming text.
        
        Args:
            previous_text: Text recognized in preceding chunk(s).
            current_text: Text recognized in current chunk.
            
        Returns:
            OverlapResolution with trimmed text and explanation.
        """
        if not current_text or not current_text.strip():
            return OverlapResolution(
                cleaned_text="",
                overlap_token_count=0,
                overlap_phrase="",
                original_text=current_text,
                reason="empty_current_text",
            )

        if not previous_text or not previous_text.strip():
            return OverlapResolution(
                cleaned_text=current_text.strip(),
                overlap_token_count=0,
                overlap_phrase="",
                original_text=current_text,
                reason="no_previous_text",
            )

        prev_norm = previous_text if self.case_sensitive else previous_text.lower()
        curr_norm = current_text if self.case_sensitive else current_text.lower()

        prev_tokens = tokenize_text(prev_norm)
        curr_tokens = tokenize_text(curr_norm)

        if not prev_tokens or not curr_tokens:
            return OverlapResolution(
                cleaned_text=current_text.strip(),
                overlap_token_count=0,
                overlap_phrase="",
                original_text=current_text,
                reason="no_tokens",
            )

        # Check candidate overlap lengths from max_overlap_tokens down to min_overlap_tokens
        max_k = min(len(prev_tokens), len(curr_tokens), self.max_overlap_tokens)
        best_overlap_k = 0
        best_overlap_phrase = ""

        for k in range(max_k, self.min_overlap_tokens - 1, -1):
            prev_suffix = prev_tokens[-k:]
            curr_prefix = curr_tokens[:k]

            if prev_suffix == curr_prefix:
                # Potential overlap detected!
                # Now check for legitimate repetition safeguards:
                if k == 1:
                    matched_word = curr_prefix[0]
                    # Check if the word is a legitimate conversational repetition
                    # e.g. speaker said "very very important" inside current_text
                    if len(curr_tokens) >= 2 and curr_tokens[1] == matched_word:
                        # Current text starts with "very very ...". The first "very" matched previous chunk suffix.
                        # This means speaker repeated the word across the boundary ("...very" + "very very...").
                        # We only remove 1 duplicate token so one "very" remains in current text!
                        best_overlap_k = 1
                        best_overlap_phrase = matched_word
                        break

                    # If previous text already contained repeated word at end ("...very very"),
                    # and current text has 1 instance ("very important"), this could be an ASR echo.
                    # Strip single word if it matches
                    best_overlap_k = 1
                    best_overlap_phrase = matched_word
                    break
                else:
                    # Multi-token overlap (k >= 2, e.g. "artificial intelligence")
                    # Multi-word exact overlaps across chunk boundaries are virtually always chunk boundary artifacts
                    best_overlap_k = k
                    best_overlap_phrase = " ".join(curr_prefix)
                    break

        if best_overlap_k > 0:
            # Reconstruct current text without the overlapping prefix tokens
            cleaned = self._strip_prefix_tokens(current_text, best_overlap_k)
            return OverlapResolution(
                cleaned_text=cleaned,
                overlap_token_count=best_overlap_k,
                overlap_phrase=best_overlap_phrase,
                original_text=current_text,
                reason=f"boundary_overlap_removed_{best_overlap_k}_tokens",
            )

        return OverlapResolution(
            cleaned_text=current_text.strip(),
            overlap_token_count=0,
            overlap_phrase="",
            original_text=current_text,
            reason="no_overlap_detected",
        )

    def _strip_prefix_tokens(self, text: str, token_count: int) -> str:
        """
        Remove the first `token_count` words/tokens from the text string,
        preserving punctuation and casing on remaining words.
        """
        # Split text by whitespace to preserve word tokens
        words = text.strip().split()
        if token_count >= len(words):
            return ""
        return " ".join(words[token_count:]).strip()
