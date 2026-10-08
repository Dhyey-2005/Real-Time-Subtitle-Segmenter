"""
Real-Time Subtitle Segmenter and N-gram Phrase Extractor - Phase 1 Package.

This package configures project-local paths (including NLTK data) strictly on the
project drive to ensure zero usage of the C: drive for caches or resources.
"""

import os
import nltk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")

# Enforce project-local NLTK data directory
if os.path.exists(NLTK_DATA_DIR):
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
