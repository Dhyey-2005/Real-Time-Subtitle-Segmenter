"""
Tokenizer Module for Phase 1.

Provides clean word tokenization using NLTK (configured locally on the project drive)
with safe fallbacks. Handles punctuation, casing, and empty inputs robustly.
"""

import os
import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize

# Ensure project-local NLTK data directory is registered
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize an input text into a list of clean word tokens.
    
    Uses NLTK's word_tokenize backed by the local NLTK data directory.
    Filters out standalone punctuation symbols while preserving legitimate words
    (including contractions with apostrophes or hyphenated words).
    
    Args:
        text: Normalized or raw text string.
        
    Returns:
        List of clean word tokens.
    """
    if not text or not text.strip():
        return []
    
    try:
        raw_tokens = word_tokenize(text)
    except Exception:
        # Robust fallback using regex word boundary tokenization if NLTK tokenizer fails
        raw_tokens = re.findall(r"\b[\w'-]+\b", text)
    
    # Filter out pure punctuation tokens and keep alphanumeric words
    tokens = [
        tok.strip()
        for tok in raw_tokens
        if tok.strip() and re.search(r"\w", tok)
    ]
    
    return tokens
