"""
Tests for N-gram Extractor Module.
"""

import os
import sys
from collections import Counter
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ngram_extractor import (
    generate_ngrams,
    extract_bigrams,
    extract_trigrams,
    format_ngram_counter,
)


class TestNgramExtraction:
    """Tests for consecutive bigram and trigram extraction and frequency counting."""

    def test_correct_bigram_generation(self):
        tokens = ["the", "government", "announced", "a", "new", "policy"]
        bigrams = generate_ngrams(tokens, 2)
        expected = [
            ("the", "government"),
            ("government", "announced"),
            ("announced", "a"),
            ("a", "new"),
            ("new", "policy"),
        ]
        assert bigrams == expected

    def test_correct_trigram_generation(self):
        tokens = ["the", "government", "announced", "a", "new", "policy"]
        trigrams = generate_ngrams(tokens, 3)
        expected = [
            ("the", "government", "announced"),
            ("government", "announced", "a"),
            ("announced", "a", "new"),
            ("a", "new", "policy"),
        ]
        assert trigrams == expected

    def test_frequency_counting(self):
        tokens = [
            "economic", "package", "and", "another", "economic", "package",
            "and", "a", "third", "economic", "package"
        ]
        bigram_counts = extract_bigrams(tokens)
        assert bigram_counts[("economic", "package")] == 3
        assert bigram_counts[("package", "and")] == 2

    def test_insufficient_tokens(self):
        assert generate_ngrams(["word"], 2) == []
        assert generate_ngrams(["word1", "word2"], 3) == []
        assert extract_bigrams([]) == Counter()
        assert extract_trigrams([]) == Counter()

    def test_format_ngram_counter(self):
        counter = Counter({("economic", "package"): 5, ("new", "policy"): 2})
        formatted = format_ngram_counter(counter)
        assert len(formatted) == 2
        assert formatted[0]["phrase"] == "economic package"
        assert formatted[0]["frequency"] == 5
        assert formatted[0]["n"] == 2
        assert formatted[1]["phrase"] == "new policy"
        assert formatted[1]["frequency"] == 2
