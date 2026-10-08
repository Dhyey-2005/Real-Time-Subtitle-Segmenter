"""
Collocation & Association Scorer Module for Phase 4.

Computes mathematically sound, numerically safe statistical association metrics
including Pointwise Mutual Information (PMI) for bigrams and generalized Tri-PMI
for trigrams.

STATISTICAL FORMULATION & DOCUMENTED LIMITATIONS:
1. Bigram PMI:
   PMI(x, y) = log2( P(x, y) / (P(x) * P(y)) )
   Measures how much more often words x and y occur together than would be expected
   if they were statistically independent.

2. Trigram Symmetric PMI (Tri-PMI):
   PMI3(x, y, z) = log2( P(x, y, z) / (P(x) * P(y) * P(z)) )
   Measures joint 3-word binding relative to unigram independence.

3. Small-Corpus Limitation:
   PMI naturally inflates for rare events (low unigram frequencies). In a small
   corpus, an n-gram occurring only once can achieve an artificially high PMI.
   Therefore, PMI is treated as an association signal and MUST be coupled with
   minimum frequency filtering and content-word density scoring.
"""

import math
from collections import Counter
from typing import Tuple, Dict, Any


def calculate_pmi_bigram(
    bigram: Tuple[str, str],
    bigram_counts: Counter,
    unigram_counts: Counter,
    total_bigrams: int,
    total_unigrams: int,
) -> float:
    """
    Calculate Pointwise Mutual Information (PMI) for a bigram tuple (x, y).
    
    Args:
        bigram: Tuple of two word tokens (x, y).
        bigram_counts: Counter of observed bigrams.
        unigram_counts: Counter of observed unigrams.
        total_bigrams: Total number of bigram tokens in corpus.
        total_unigrams: Total number of unigram tokens in corpus.
        
    Returns:
        PMI score as a float rounded to 4 decimals, or 0.0 if zero probability.
    """
    if total_bigrams <= 0 or total_unigrams <= 0:
        return 0.0

    count_xy = bigram_counts.get(bigram, 0)
    count_x = unigram_counts.get(bigram[0], 0)
    count_y = unigram_counts.get(bigram[1], 0)

    # Safe guard against zero occurrences
    if count_xy <= 0 or count_x <= 0 or count_y <= 0:
        return 0.0

    p_xy = count_xy / total_bigrams
    p_x = count_x / total_unigrams
    p_y = count_y / total_unigrams

    expected = p_x * p_y
    if expected <= 0:
        return 0.0

    ratio = p_xy / expected
    if ratio <= 0:
        return 0.0

    return round(math.log2(ratio), 4)


def calculate_pmi_trigram(
    trigram: Tuple[str, str, str],
    trigram_counts: Counter,
    unigram_counts: Counter,
    total_trigrams: int,
    total_unigrams: int,
) -> float:
    """
    Calculate Symmetric Tri-PMI for a trigram tuple (x, y, z).
    
    Args:
        trigram: Tuple of three word tokens (x, y, z).
        trigram_counts: Counter of observed trigrams.
        unigram_counts: Counter of observed unigrams.
        total_trigrams: Total number of trigram tokens in corpus.
        total_unigrams: Total number of unigram tokens in corpus.
        
    Returns:
        Tri-PMI score as a float rounded to 4 decimals, or 0.0 if zero probability.
    """
    if total_trigrams <= 0 or total_unigrams <= 0:
        return 0.0

    count_xyz = trigram_counts.get(trigram, 0)
    count_x = unigram_counts.get(trigram[0], 0)
    count_y = unigram_counts.get(trigram[1], 0)
    count_z = unigram_counts.get(trigram[2], 0)

    # Safe guard against zero occurrences
    if count_xyz <= 0 or count_x <= 0 or count_y <= 0 or count_z <= 0:
        return 0.0

    p_xyz = count_xyz / total_trigrams
    p_x = count_x / total_unigrams
    p_y = count_y / total_unigrams
    p_z = count_z / total_unigrams

    expected = p_x * p_y * p_z
    if expected <= 0:
        return 0.0

    ratio = p_xyz / expected
    if ratio <= 0:
        return 0.0

    return round(math.log2(ratio), 4)
