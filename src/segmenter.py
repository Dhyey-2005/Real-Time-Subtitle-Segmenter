"""
Subtitle / Sentence Segmenter Module for Phase 1.

Provides a deterministic baseline segmenter for unpunctuated ASR streams.
Uses linguistic heuristics, length constraints, clause boundary cues,
and discourse markers to partition continuous speech text into readable
subtitle chunks.

LIMITATIONS OF PHASE 1 BASELINE:
1. Purely deterministic & rule-based: Does not utilize acoustic features (pitch, pause duration)
   present in real speech streams.
2. Approximate clause detection: Conjunctions (e.g., 'and') can either link clauses
   or coordinate noun phrases ('wind and solar'). The baseline relies on length thresholds
   and verb presence to disambiguate.
3. Fixed length bounds: Subtitle length heuristics (5 to 14 words) prioritize readability
   on screen rather than perfect semantic closure.
Future phases will introduce streaming buffers, acoustic alignment, and enhanced semantic segmentation.
"""

import os
from typing import List, Set
import nltk

# Ensure project-local NLTK data directory is registered
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

from src.tokenizer import tokenize_text

# Strong discourse markers and transition cues that signal new clauses/utterances
DISCOURSE_MARKERS: Set[str] = {
    "moreover",
    "furthermore",
    "meanwhile",
    "however",
    "therefore",
    "consequently",
    "otherwise",
    "nevertheless",
    "finally",
    "firstly",
    "secondly",
}

MULTI_WORD_MARKERS: List[List[str]] = [
    ["in", "conclusion"],
    ["in", "summary"],
    ["in", "addition"],
    ["on", "the", "other", "hand"],
    ["we", "also", "observed"],
    ["we", "also"],
    ["officials", "said"],
    ["officials", "confirmed"],
    ["projections", "indicate"],
    ["citizens", "have", "demanded"],
    ["prime", "minister", "highlighted"],
    ["good", "morning"],
    ["good", "afternoon"],
    ["good", "evening"],
    ["welcome", "to"],
    ["thank", "you"],
]

CONJUNCTIONS: Set[str] = {
    "and",
    "but",
    "or",
    "because",
    "although",
    "though",
    "while",
    "whereas",
    "since",
}


def _has_verb(tokens: List[str]) -> bool:
    """Check if token list contains at least one verb using POS tagging."""
    if not tokens:
        return False
    try:
        tagged = nltk.pos_tag(tokens)
        return any(pos.startswith("VB") for _, pos in tagged)
    except Exception:
        # Fallback heuristic: check for common auxiliary or primary verbs
        common_verbs = {"is", "are", "was", "were", "will", "would", "has", "have", "had", "can", "could", "said"}
        return any(tok.lower() in common_verbs for tok in tokens)


def _matches_multi_word_marker(tokens: List[str], start_idx: int) -> bool:
    """Check if a sequence of tokens starting at start_idx matches any known multi-word discourse marker."""
    for marker in MULTI_WORD_MARKERS:
        marker_len = len(marker)
        if start_idx + marker_len <= len(tokens):
            if [t.lower() for t in tokens[start_idx:start_idx + marker_len]] == marker:
                return True
    return False


def segment_text(
    text: str,
    min_words: int = 5,
    max_words: int = 14,
    target_words: int = 9,
) -> List[str]:
    """
    Deterministically segment unpunctuated text into readable subtitle/sentence units.
    
    Args:
        text: Raw or normalized unpunctuated speech text.
        min_words: Minimum words allowed in a segment before creating a boundary (default: 5).
        max_words: Maximum words allowed before forcing a boundary (default: 14).
        target_words: Target word length for ideal subtitle readability (default: 9).
        
    Returns:
        List of segmented subtitle text strings.
    """
    tokens = tokenize_text(text)
    if not tokens:
        return []
    
    # If the total text is short, return as a single segment
    if len(tokens) <= max_words:
        return [" ".join(tokens)]
    
    segments: List[List[str]] = []
    current_chunk: List[str] = []
    
    i = 0
    total_tokens = len(tokens)
    
    while i < total_tokens:
        tok = tokens[i]
        tok_lower = tok.lower()
        chunk_len = len(current_chunk)
        
        # Check boundary condition 1: Multi-word discourse marker
        if chunk_len >= min_words and _matches_multi_word_marker(tokens, i):
            segments.append(current_chunk)
            current_chunk = []
            chunk_len = 0
            
        # Check boundary condition 2: Single-word discourse marker
        elif chunk_len >= min_words and tok_lower in DISCOURSE_MARKERS:
            segments.append(current_chunk)
            current_chunk = []
            chunk_len = 0
            
        # Check boundary condition 3: Conjunction boundary
        # If current chunk has reached target_words and contains a verb, break before conjunction
        elif chunk_len >= target_words and tok_lower in CONJUNCTIONS and _has_verb(current_chunk):
            segments.append(current_chunk)
            current_chunk = []
            chunk_len = 0
            
        # Check boundary condition 4: Maximum word limit reached
        elif chunk_len >= max_words:
            # Force boundary at maximum length
            segments.append(current_chunk)
            current_chunk = []
            chunk_len = 0
            
        current_chunk.append(tok)
        i += 1
        
    if current_chunk:
        # If the trailing segment is too short, merge it with previous segment if possible
        if len(current_chunk) < min_words and segments:
            # Check if merging exceeds comfortable limits
            if len(segments[-1]) + len(current_chunk) <= max_words + 4:
                segments[-1].extend(current_chunk)
            else:
                segments.append(current_chunk)
        else:
            segments.append(current_chunk)
            
    # Format tokens back into strings
    return [" ".join(seg) for seg in segments if seg]
