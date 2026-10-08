"""
Boundary Scorer Module for Phase 3.

Evaluates candidate subtitle boundaries deterministically by combining:
1. Discourse transition cues
2. Clause structure & verb presence
3. Multi-word phrase completeness and protection
4. Preposition/determiner boundary penalties
5. Segment length balancing around a target word count
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import nltk

from src.segmenter import (
    DISCOURSE_MARKERS,
    MULTI_WORD_MARKERS,
    CONJUNCTIONS,
    _has_verb,
    _matches_multi_word_marker,
)
from src.phrase_protection import PhraseProtector
from src.phrase_filter import BOUNDARY_STOPWORDS


# Additional strong discourse transitions for Phase 3
EXTENDED_DISCOURSE_MARKERS = {
    "however",
    "therefore",
    "meanwhile",
    "moreover",
    "furthermore",
    "consequently",
    "otherwise",
    "nevertheless",
    "finally",
}

EXTENDED_MULTI_WORD_MARKERS: List[List[str]] = [
    ["in", "addition"],
    ["on", "the", "other", "hand"],
    ["as", "a", "result"],
    ["for", "example"],
    ["in", "conclusion"],
    ["in", "summary"],
    ["officials", "confirmed"],
    ["officials", "said"],
    ["prime", "minister", "announced"],
    ["prime", "minister", "highlighted"],
    ["citizens", "demanded"],
    ["transport", "department", "discussed"],
    ["welcome", "to"],
    ["thank", "you"],
]


@dataclass
class BoundaryEvaluation:
    """Detailed result of evaluating a candidate boundary position."""
    split_idx: int
    score: float
    reason: str
    trigger_word: str
    is_valid: bool
    phrase_broken: Optional[str] = None
    character_length: int = 0
    word_count: int = 0


class BoundaryScorer:
    """
    Computes explainable boundary scores for candidate split positions.
    """

    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 14,
        max_characters: int = 80,
        target_words: int = 9,
        emission_threshold: float = 40.0,
        phrase_protector: Optional[PhraseProtector] = None,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.max_characters = max_characters
        self.target_words = target_words
        self.emission_threshold = emission_threshold
        self.phrase_protector = phrase_protector or PhraseProtector()

    def _matches_extended_multi_marker(self, tokens: List[str], start_idx: int) -> Optional[str]:
        """Check if tokens starting at start_idx match any multi-word marker."""
        tokens_lower = [t.lower() for t in tokens]
        all_markers = MULTI_WORD_MARKERS + EXTENDED_MULTI_WORD_MARKERS
        for marker in all_markers:
            mlen = len(marker)
            if start_idx + mlen <= len(tokens):
                if tokens_lower[start_idx:start_idx + mlen] == marker:
                    return " ".join(marker)
        return None

    def evaluate_boundary(self, tokens: List[str], split_idx: int) -> BoundaryEvaluation:
        """
        Evaluate candidate boundary at `split_idx` (where tokens[:split_idx] is prefix).
        
        Args:
            tokens: Buffer token list.
            split_idx: Candidate split index.
            
        Returns:
            BoundaryEvaluation with score, reason, trigger, and validity.
        """
        prefix_tokens = tokens[:split_idx]
        word_count = len(prefix_tokens)
        char_length = len(" ".join(prefix_tokens))

        # Check Hard Constraints
        if word_count < self.min_words:
            return BoundaryEvaluation(
                split_idx=split_idx,
                score=-999.0,
                reason="below_min_length",
                trigger_word="",
                is_valid=False,
                character_length=char_length,
                word_count=word_count,
            )

        if char_length > self.max_characters:
            return BoundaryEvaluation(
                split_idx=split_idx,
                score=-999.0,
                reason="character_limit",
                trigger_word="",
                is_valid=False,
                character_length=char_length,
                word_count=word_count,
            )

        if word_count > self.max_words:
            return BoundaryEvaluation(
                split_idx=split_idx,
                score=-999.0,
                reason="length_limit",
                trigger_word="",
                is_valid=False,
                character_length=char_length,
                word_count=word_count,
            )

        score = 0.0
        primary_reason = "balanced_boundary"
        trigger_word = tokens[split_idx].lower() if split_idx < len(tokens) else "eos"

        # Signal 1: Phrase Protection
        phrase_broken = self.phrase_protector.is_split_inside_phrase(tokens, split_idx)
        if phrase_broken:
            broken_text, _, _ = phrase_broken
            score -= 85.0
            phrase_broken_name = broken_text
        else:
            phrase_broken_name = None

        completed_phrase = self.phrase_protector.is_immediately_after_phrase(tokens, split_idx)
        if completed_phrase:
            score += 35.0
            primary_reason = "phrase_complete"
            trigger_word = completed_phrase

        # Signal 2: Discourse Transitions
        multi_marker = self._matches_extended_multi_marker(tokens, split_idx)
        if multi_marker:
            score += 65.0
            primary_reason = "discourse_transition"
            trigger_word = multi_marker
        elif split_idx < len(tokens):
            tok_lower = tokens[split_idx].lower()
            if tok_lower in EXTENDED_DISCOURSE_MARKERS or tok_lower in DISCOURSE_MARKERS:
                score += 55.0
                primary_reason = "discourse_transition"
                trigger_word = tok_lower

            # Signal 3: Conjunctions and Clause Structure
            elif tok_lower in CONJUNCTIONS:
                if _has_verb(prefix_tokens):
                    if word_count >= self.target_words - 2:
                        score += 35.0
                        primary_reason = "clause_boundary"
                        trigger_word = tok_lower
                    else:
                        # Short prefix with conjunction: mild signal, but not alone
                        score += 15.0
                        trigger_word = tok_lower
                else:
                    # Conjunction without preceding verb usually joins compound nouns
                    score -= 25.0

        # Signal 4: Tail Token Penalties (Avoid ending on prepositions/determiners)
        tail_tok = tokens[split_idx - 1].lower()
        if tail_tok in BOUNDARY_STOPWORDS:
            score -= 30.0

        # Signal 5: Length Balancing around target_words
        dist = abs(word_count - self.target_words)
        if dist == 0:
            score += 15.0
        elif dist <= 2:
            score += 10.0
        elif dist <= 4:
            score += 5.0

        if word_count <= 4:
            score -= 10.0

        is_valid = (score >= self.emission_threshold) and (phrase_broken_name is None)

        return BoundaryEvaluation(
            split_idx=split_idx,
            score=round(score, 2),
            reason=primary_reason,
            trigger_word=trigger_word,
            is_valid=is_valid,
            phrase_broken=phrase_broken_name,
            character_length=char_length,
            word_count=word_count,
        )
