"""
Phrase Filtering Module for Phase 1.

Filters frequent raw n-grams to isolate potentially meaningful recurring phrases
by applying:
1. Minimum frequency thresholding
2. Stopword filtering (all-stopword removal and boundary stopword pruning)
3. Grammatical / POS pattern filtering (identifying content-rich noun phrases and concepts)
4. Repetition and trivial noise elimination
"""

import os
from collections import Counter
from typing import List, Tuple, Dict, Set
import nltk
from nltk.corpus import stopwords

# Ensure project-local NLTK data directory is registered
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Load stopwords safely from local NLTK data
try:
    ENGLISH_STOPWORDS: Set[str] = set(stopwords.words("english"))
except Exception:
    ENGLISH_STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "this", "that", "these", "those", "it",
        "its", "we", "our", "you", "your", "they", "their", "he", "she", "him",
        "her", "i", "me", "my",
    }

# Boundary words that should not begin or end a clean stand-alone recurring phrase
BOUNDARY_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "this", "that", "these", "those",
    "is", "are", "was", "were", "will", "would", "shall", "be", "has",
    "have", "had", "our", "we", "you", "they", "its", "their", "into",
    "also", "moreover", "meanwhile", "furthermore", "which",
}


def is_meaningful_phrase(
    ngram_tokens: Tuple[str, ...],
    use_pos: bool = True,
) -> bool:
    """
    Evaluate whether an n-gram token tuple qualifies as a potentially meaningful phrase.
    
    Criteria:
    - Must not have all stopwords.
    - Must not have repeated identical words (e.g. 'the the').
    - Must not start or end with functional boundary words (determiners, prepositions, conjunctions).
    - Must contain at least one content noun (if POS tagging is enabled).
    - Must not consist purely of auxiliary/modal verbs, pronouns, or conjunctions.
    
    Args:
        ngram_tokens: Tuple of tokens representing the n-gram.
        use_pos: Whether to apply part-of-speech tagging heuristics.
        
    Returns:
        True if the n-gram is potentially meaningful, False otherwise.
    """
    tokens_lower = [t.lower() for t in ngram_tokens]
    
    # 1. Reject identical repeated words ('the the', 'package package')
    if len(set(tokens_lower)) == 1:
        return False
        
    # 2. Reject if ALL tokens are stopwords ('of the', 'in the', 'it is', 'and the')
    if all(t in ENGLISH_STOPWORDS for t in tokens_lower):
        return False
        
    # 3. Reject if starts or ends with functional boundary words ('the new', 'package will')
    if tokens_lower[0] in BOUNDARY_STOPWORDS:
        return False
    if tokens_lower[-1] in BOUNDARY_STOPWORDS:
        return False
        
    # 4. Linguistic POS check
    if use_pos:
        try:
            tagged = nltk.pos_tag(ngram_tokens)
            tags = [tag for _, tag in tagged]
            
            # Phrase must contain at least one noun (NN, NNS, NNP, NNPS) or key adjective
            has_noun = any(tag.startswith("NN") for tag in tags)
            if not has_noun:
                return False
                
            # Discard if any token is a coordinating conjunction or pronoun
            if any(tag in {"CC", "PRP", "PRP$", "IN", "DT", "MD"} for tag in [tags[0], tags[-1]]):
                return False
                
        except Exception:
            # If POS tagging fails for any reason, rely on stopword heuristics
            pass
            
    return True


def filter_phrases(
    ngram_counter: Counter,
    min_freq: int = 2,
    use_pos: bool = True,
) -> List[Dict[str, any]]:
    """
    Filter raw n-grams from a Counter into meaningful recurring phrases.
    
    Args:
        ngram_counter: Counter mapping n-gram tuples to frequency counts.
        min_freq: Minimum recurrence frequency required (default: 2).
        use_pos: Whether to use POS validation (default: True).
        
    Returns:
        List of dicts: [
            {"phrase": str, "frequency": int, "n": int, "pos_pattern": str},
            ...
        ] sorted by frequency descending.
    """
    filtered_results = []
    
    for ngram_tuple, freq in ngram_counter.most_common():
        # Check minimum frequency threshold
        if freq < min_freq:
            continue
            
        if is_meaningful_phrase(ngram_tuple, use_pos=use_pos):
            pos_pattern = ""
            try:
                tagged = nltk.pos_tag(ngram_tuple)
                pos_pattern = "-".join(tag for _, tag in tagged)
            except Exception:
                pos_pattern = "UNKNOWN"
                
            filtered_results.append({
                "phrase": " ".join(ngram_tuple),
                "frequency": freq,
                "n": len(ngram_tuple),
                "pos_pattern": pos_pattern,
            })
            
    # Sort primarily by frequency descending, secondarily by phrase length descending
    filtered_results.sort(key=lambda x: (x["frequency"], x["n"]), reverse=True)
    return filtered_results
