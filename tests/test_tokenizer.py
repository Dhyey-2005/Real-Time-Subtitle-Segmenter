"""
Tests for Tokenizer and Text Normalization Module.
"""

import os
import sys
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.text_processor import normalize_text
from src.tokenizer import tokenize_text


class TestNormalization:
    """Tests for text normalization handling spaces, line breaks, casing, and empty inputs."""

    def test_normal_text(self):
        text = "the government announced a new economic package"
        assert normalize_text(text) == "the government announced a new economic package"

    def test_multiple_spaces(self):
        text = "the   government    announced     a new package"
        assert normalize_text(text) == "the government announced a new package"

    def test_line_breaks_and_blank_lines(self):
        text = "the government\n\nannounced a new\r\neconomic package\n"
        assert normalize_text(text) == "the government announced a new economic package"

    def test_empty_and_whitespace_input(self):
        assert normalize_text("") == ""
        assert normalize_text("    ") == ""
        assert normalize_text("\n\n\t  \r\n") == ""

    def test_casing_normalization(self):
        text = "THE GOVERNMENT announced A New Package"
        assert normalize_text(text, lower=True) == "the government announced a new package"
        assert normalize_text(text, lower=False) == "THE GOVERNMENT announced A New Package"

    def test_noise_removal(self):
        text = "the government announced *** special $$$ economic package!!!"
        assert normalize_text(text) == "the government announced special economic package"


class TestTokenizer:
    """Tests for tokenization covering normal text, edge cases, punctuation, and contractions."""

    def test_normal_text(self):
        text = "the government announced a new economic package"
        tokens = tokenize_text(text)
        assert tokens == ["the", "government", "announced", "a", "new", "economic", "package"]

    def test_empty_input(self):
        assert tokenize_text("") == []
        assert tokenize_text("   ") == []
        assert tokenize_text(None) == []

    def test_contractions_and_hyphens(self):
        text = "today's long-term economic package"
        tokens = tokenize_text(text)
        assert "today's" in tokens or ("today" in tokens and "'s" in tokens)
        assert any("long" in t for t in tokens)

    def test_punctuation_stripping(self):
        text = "hello, world! this is a test."
        tokens = tokenize_text(text)
        # Verify that standalone punctuation tokens are excluded
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens
        assert "hello" in tokens
        assert "world" in tokens
