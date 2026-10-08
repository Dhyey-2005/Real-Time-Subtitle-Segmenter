#!/usr/bin/env python3
"""
CLI Entrypoint for Real-Time Subtitle Segmenter and N-gram Phrase Extractor (Phase 1).

Usage:
    python run_phase1.py --input data/input/sample_transcript.txt
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce project-local NLTK data directory
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR):
    import nltk
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

from src.main import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Offline NLP Prototype for Subtitle Segmentation and N-gram Phrase Extraction"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join("data", "input", "sample_transcript.txt"),
        help="Path to input unpunctuated ASR transcript file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join("data", "output"),
        help="Path to directory for output artifacts",
    )
    parser.add_argument(
        "--min-freq",
        "-m",
        type=int,
        default=2,
        help="Minimum recurrence frequency for phrase filtering (default: 2)",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini semantic enrichment and run strictly local",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading transcript...")
    print("Normalizing text...")
    print("Tokenizing...")
    print("Segmenting...")
    print("Extracting bigrams...")
    print("Extracting trigrams...")
    print("Filtering phrases...")
    print("Writing results...")

    results = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        min_phrase_freq=args.min_freq,
        enable_gemini=not args.no_gemini,
    )

    print("\n" + "=" * 50)
    print("PHASE 1 EXECUTION SUMMARY:")
    print(f"  • Input Words:         {results['input_word_count']}")
    print(f"  • Normalized Tokens:   {results['token_count']}")
    print(f"  • Subtitle Segments:   {results['segment_count']}")
    print(f"  • Avg Segment Length:  {results['avg_segment_length']} words")
    print(f"  • Unique Bigrams:      {results['unique_bigrams']}")
    print(f"  • Unique Trigrams:     {results['unique_trigrams']}")
    print(f"  • Filtered Phrases:    {results['filtered_phrase_count']}")
    print(f"  • Output Directory:    {output_dir}")
    print(f"  • Gemini Status:       {results['gemini_status']}")
    print("=" * 50)
    print("Phase 1 completed.")


if __name__ == "__main__":
    main()
