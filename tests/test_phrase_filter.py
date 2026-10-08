"""
Tests for Phrase Filtering Module and Gemini Safety.
"""

import os
import sys
from collections import Counter
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phrase_filter import is_meaningful_phrase, filter_phrases
from src.gemini_client import GeminiSemanticClient


class TestPhraseFilter:
    """Tests for distinguishing meaningful recurring phrases from stopword/noise n-grams."""

    def test_removal_of_stopword_heavy_phrases(self):
        # All stopword bigrams/trigrams must be discarded
        assert not is_meaningful_phrase(("of", "the"))
        assert not is_meaningful_phrase(("in", "the"))
        assert not is_meaningful_phrase(("to", "the"))
        assert not is_meaningful_phrase(("and", "the"))
        assert not is_meaningful_phrase(("it", "is"))
        assert not is_meaningful_phrase(("that", "the"))
        assert not is_meaningful_phrase(("and", "in", "the"))

    def test_removal_of_boundary_stopwords(self):
        # Phrases starting or ending with functional determiners/prepositions
        assert not is_meaningful_phrase(("the", "new"))
        assert not is_meaningful_phrase(("package", "will"))
        assert not is_meaningful_phrase(("package", "is"))
        assert not is_meaningful_phrase(("and", "economic"))

    def test_retention_of_meaningful_phrases(self):
        # Meaningful noun phrases should pass
        assert is_meaningful_phrase(("economic", "package"))
        assert is_meaningful_phrase(("renewable", "energy"))
        assert is_meaningful_phrase(("artificial", "intelligence"))
        assert is_meaningful_phrase(("renewable", "energy", "transition"))
        assert is_meaningful_phrase(("job", "creation", "initiatives"))

    def test_identical_repeated_word_removal(self):
        assert not is_meaningful_phrase(("the", "the"))
        assert not is_meaningful_phrase(("word", "word"))

    def test_minimum_frequency_threshold(self):
        counter = Counter({
            ("economic", "package"): 4,
            ("renewable", "energy"): 3,
            ("isolated", "phrase"): 1,  # below threshold
            ("of", "the"): 10,           # stopword phrase
        })
        
        filtered = filter_phrases(counter, min_freq=2)
        phrases = [item["phrase"] for item in filtered]
        
        assert "economic package" in phrases
        assert "renewable energy" in phrases
        assert "isolated phrase" not in phrases
        assert "of the" not in phrases


class TestGeminiSafetyAndFallbacks:
    """Tests for Gemini client robustness and graceful degradation."""

    def test_missing_api_key_initialization(self, monkeypatch):
        # Providing None or empty key with environment cleared should not raise exceptions
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiSemanticClient(api_key=None, use_env=False)
        assert not client.is_available()
        
        # Test categorizing phrases with unavailable client
        result = client.categorize_phrases(["economic package", "clean energy"])
        assert "economic package" in result
        assert result["economic package"] == "Local Deterministic Extraction"

    def test_invalid_api_key_does_not_crash(self):
        # An invalid key should gracefully fallback rather than raising unhandled errors
        client = GeminiSemanticClient(api_key="INVALID_DUMMY_KEY_FOR_TESTING")
        result = client.categorize_phrases(["economic package"])
        assert isinstance(result, dict)
        assert "economic package" in result

    def test_empty_phrase_list(self):
        client = GeminiSemanticClient(api_key=None)
        assert client.categorize_phrases([]) == {}
