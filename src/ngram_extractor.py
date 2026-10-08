"""
N-gram Extractor Module for Phase 1.

Responsible for generating consecutive bigrams (2-grams) and trigrams (3-grams)
from word token sequences and maintaining accurate frequency counts using Counter.
"""

from collections import Counter
from typing import List, Tuple, Dict


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generate all consecutive n-grams of size n from a list of tokens.
    
    Args:
        tokens: Ordered list of word tokens.
        n: Size of n-gram (e.g. 2 for bigrams, 3 for trigrams).
        
    Returns:
        List of n-gram tuples.
    """
    if not tokens or n <= 0 or len(tokens) < n:
        return []
    
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_ngram_counts(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams of size n from tokens and calculate frequency counts.
    
    Args:
        tokens: Ordered list of word tokens.
        n: Size of the n-gram.
        
    Returns:
        Counter mapping n-gram tuples to frequency counts.
    """
    ngrams = generate_ngrams(tokens, n)
    return Counter(ngrams)


def extract_bigrams(tokens: List[str]) -> Counter:
    """
    Extract bigram (2-gram) frequency counts.
    
    Args:
        tokens: Ordered list of word tokens.
        
    Returns:
        Counter of bigrams.
    """
    return extract_ngram_counts(tokens, 2)


def extract_trigrams(tokens: List[str]) -> Counter:
    """
    Extract trigram (3-gram) frequency counts.
    
    Args:
        tokens: Ordered list of word tokens.
        
    Returns:
        Counter of trigrams.
    """
    return extract_ngram_counts(tokens, 3)


def format_ngram_counter(counter: Counter) -> List[Dict[str, any]]:
    """
    Format n-gram Counter into a structured list of dictionaries sorted by descending frequency.
    
    Args:
        counter: Counter of n-gram tuples.
        
    Returns:
        List of dicts: [{"phrase": str, "frequency": int, "n": int}, ...]
    """
    results = []
    for ngram, freq in counter.most_common():
        results.append({
            "phrase": " ".join(ngram),
            "frequency": freq,
            "n": len(ngram),
        })
    return results
