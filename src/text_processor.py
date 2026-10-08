"""
Text Processor Module for Phase 1.

Responsible for loading raw unpunctuated ASR text and applying deterministic normalization
while preserving raw content for debugging and downstream comparison.
"""

import os
import re
from typing import Tuple


def load_text(file_path: str) -> str:
    """
    Load raw text from a local file path.
    
    Args:
        file_path: Absolute or relative path to the text file.
        
    Returns:
        Raw text content as a string.
        
    Raises:
        FileNotFoundError: If the input file cannot be found.
        IOError: If an error occurs during file reading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(raw_text: str, lower: bool = True) -> str:
    """
    Deterministically normalize input text:
    - Normalizes line breaks and whitespace
    - Strips leading and trailing padding
    - Removes non-alphanumeric noise while retaining apostrophes and hyphens
    - Converts to lowercase if requested for statistical processing
    
    Args:
        raw_text: Raw input string from ASR or transcript file.
        lower: Whether to lowercase the text. Default is True.
        
    Returns:
        Normalized text string.
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    # Normalize unicode spaces to standard ASCII space
    text = re.sub(r"[\u00A0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]", " ", raw_text)
    
    # Replace carriage returns and newlines with space
    text = re.sub(r"[\r\n]+", " ", text)
    
    # Remove unwanted punctuation and control noise, keeping word characters, apostrophes, and hyphens
    text = re.sub(r"[^\w\s\'-]", " ", text)
    
    # Collapse multiple consecutive whitespaces into a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    if lower:
        text = text.lower()
        
    return text


def process_input(file_path: str) -> Tuple[str, str]:
    """
    Convenience function that loads and normalizes text, keeping raw text distinct.
    
    Args:
        file_path: Path to the input file.
        
    Returns:
        Tuple of (raw_text, normalized_text).
    """
    raw_text = load_text(file_path)
    normalized_text = normalize_text(raw_text)
    return raw_text, normalized_text
