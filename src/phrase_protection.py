"""
Phrase Protection Module for Phase 3.

Provides deterministic detection and protection of meaningful multi-word phrases
(such as bigrams and trigrams) to prevent awkward boundaries that cut through
cohesive concepts.
"""

from typing import List, Set, Tuple, Optional, Dict
from src.tokenizer import tokenize_text


# Standard multi-word domain phrases known in the subtitle domain
DEFAULT_PROTECTED_PHRASES: List[str] = [
    "new economic package",
    "economic package",
    "renewable energy transition",
    "renewable energy",
    "energy transition",
    "artificial intelligence infrastructure",
    "artificial intelligence",
    "intelligence infrastructure",
    "public transit expansion",
    "public transit",
    "transit expansion",
    "job creation initiatives",
    "job creation",
    "creation initiatives",
    "local communities",
    "infrastructure modernization",
    "sustainable growth",
    "clean energy",
    "digital transformation",
    "regional development",
    "long term economic",
    "economic stability",
    "long term",
]


class PhraseProtector:
    """
    Tracks cohesive multi-word phrases and checks whether candidate segmentation
    boundaries would fracture them.
    """

    def __init__(self, protected_phrases: Optional[List[str]] = None):
        """
        Initialize the phrase protector.
        
        Args:
            protected_phrases: Optional list of multi-word phrase strings.
        """
        phrase_list = protected_phrases if protected_phrases is not None else DEFAULT_PROTECTED_PHRASES
        self._phrases: Dict[Tuple[str, ...], str] = {}
        for p in phrase_list:
            tokens = tuple(tokenize_text(p.lower()))
            if len(tokens) >= 2:
                self._phrases[tokens] = p.lower()

    def add_phrase(self, phrase: str) -> None:
        """Register a new multi-word phrase to protect."""
        tokens = tuple(tokenize_text(phrase.lower()))
        if len(tokens) >= 2:
            self._phrases[tokens] = phrase.lower()

    def get_protected_phrases(self) -> List[str]:
        """Return all registered protected phrase strings."""
        return sorted(list(self._phrases.values()))

    def is_split_inside_phrase(
        self,
        tokens: List[str],
        split_idx: int,
    ) -> Optional[Tuple[str, int, int]]:
        """
        Check if splitting `tokens` at `split_idx` cuts through an active protected phrase.
        
        A split at `split_idx` cuts a phrase if the phrase starts at `start_idx < split_idx`
        and ends at `end_idx > split_idx`.
        
        Args:
            tokens: Current tokens in buffer.
            split_idx: Proposed boundary index.
            
        Returns:
            Tuple of (phrase_string, start_idx, end_idx) if split fractures a phrase,
            or None if boundary is safe.
        """
        tokens_lower = [t.lower() for t in tokens]
        num_tokens = len(tokens_lower)

        for phrase_tuple, phrase_str in self._phrases.items():
            phrase_len = len(phrase_tuple)
            # Scan all possible start positions for this phrase in tokens
            for start_idx in range(num_tokens - phrase_len + 1):
                end_idx = start_idx + phrase_len
                # Check if phrase matches at this position
                if tuple(tokens_lower[start_idx:end_idx]) == phrase_tuple:
                    # Does split_idx strictly cut the phrase?
                    if start_idx < split_idx < end_idx:
                        return (phrase_str, start_idx, end_idx)

        return None

    def is_immediately_after_phrase(
        self,
        tokens: List[str],
        split_idx: int,
    ) -> Optional[str]:
        """
        Check if `split_idx` directly follows the completion of a protected phrase.
        
        Args:
            tokens: Current tokens in buffer.
            split_idx: Proposed boundary index.
            
        Returns:
            Phrase string if split is immediately after a completed phrase, or None.
        """
        tokens_lower = [t.lower() for t in tokens]
        
        for phrase_tuple, phrase_str in self._phrases.items():
            phrase_len = len(phrase_tuple)
            start_idx = split_idx - phrase_len
            if start_idx >= 0:
                if tuple(tokens_lower[start_idx:split_idx]) == phrase_tuple:
                    return phrase_str
                    
        return None

    def count_phrase_splits(self, segments: List[str]) -> int:
        """
        Count how many protected phrases were severed across adjacent segments.
        
        For example, if segment K ends with 'artificial intelligence' and
        segment K+1 starts with 'infrastructure', that counts as 1 split.
        
        Args:
            segments: Ordered list of subtitle segment strings.
            
        Returns:
            Number of severed protected phrases.
        """
        if len(segments) < 2:
            return 0

        split_count = 0
        for i in range(len(segments) - 1):
            tail_tokens = tokenize_text(segments[i])[-3:]
            head_tokens = tokenize_text(segments[i + 1])[:3]
            boundary_tokens = [t.lower() for t in tail_tokens + head_tokens]
            boundary_len = len(boundary_tokens)

            # Check if any protected phrase spans across the boundary
            for phrase_tuple in self._phrases:
                plen = len(phrase_tuple)
                for start in range(boundary_len - plen + 1):
                    end = start + plen
                    if tuple(boundary_tokens[start:end]) == phrase_tuple:
                        # Ensure the phrase actually crosses the boundary (between tail and head)
                        split_pos = len(tail_tokens)
                        if start < split_pos < end:
                            split_count += 1
                            break

        return split_count
