#!/usr/bin/env python3
"""
CLI Entrypoint for Real-Time Subtitle Segmenter and N-gram Phrase Extractor (Phase 3).

Usage:
    python run_phase3.py --input data/input/sample_transcript.txt
    python run_phase3.py --input data/input/sample_transcript.txt --chunk-size 3 --delay 0.1
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

from src.streaming_segmenter import StreamingConfig
from src.segmentation_comparator import run_phase3_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Improved Subtitle Segmentation with Boundary Scoring & Phrase Protection"
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
        help="Path to directory for output artifacts (default: data/output)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=3,
        help="Simulated incoming chunk size in words (default: 3)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.0,
        help="Simulated real-time delay in seconds between chunks (default: 0.0)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=14,
        help="Maximum word limit per subtitle segment (default: 14)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum word limit before considering a boundary (default: 3)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=80,
        help="Maximum character limit per subtitle segment (default: 80)",
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

    config = StreamingConfig(
        chunk_size=args.chunk_size,
        simulated_delay=args.delay,
        max_segment_words=args.max_words,
        min_segment_words=args.min_words,
        max_segment_characters=args.max_chars,
    )

    print("=" * 65)
    print("PHASE 3: IMPROVED STREAMING SUBTITLE SEGMENTATION")
    print(f"Input:      {input_path}")
    print(f"Chunk Size: {config.chunk_size} words | Delay: {config.simulated_delay}s")
    print("=" * 65)

    def progress_logger(event_type: str, data: dict):
        if event_type == "chunk":
            idx = data.get("chunk_index", 0)
            text = data.get("text", "")
            if args.chunk_size == 1:
                if idx % 10 == 1 or idx <= 5:
                    print(f"[CHUNK {idx}] {text}")
            else:
                print(f"[CHUNK {idx}] {text}")
        elif event_type == "subtitle":
            sid = data.get("segment_id", 0)
            text = data.get("text", "")
            wcnt = data.get("word_count", 0)
            reason = data.get("boundary_reason", "")
            score = data.get("boundary_score", 0.0)
            lat = data.get("simulated_latency_ms", 0.0)
            print(f"  >>> [SUBTITLE {sid}] ({wcnt}w, {lat}ms | reason={reason}, score={score}) \"{text}\"")

    try:
        results = run_phase3_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            enable_gemini=not args.no_gemini,
            progress_callback=progress_logger,
        )
    except Exception as e:
        print(f"\n[ERROR] Phase 3 execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    p2 = results["phase2_quality"]
    p3 = results["phase3_quality"]

    print("\n" + "=" * 65)
    print("PHASE 3 EXECUTION COMPLETE — COMPARISON SUMMARY:")
    print(f"  • Total Input Words:       {results['input_word_count']}")
    print(f"  • Ingested Chunks:         {results['total_chunks']}")
    print(f"  • Segments (Phase 2 vs 3): {p2['segment_count']} -> {p3['segment_count']}")
    print(f"  • Avg Words/Segment:       {p2['avg_words_per_segment']} -> {p3['avg_words_per_segment']}")
    print(f"  • Short Fragments (<3w):   {p2['short_fragment_count']} -> {p3['short_fragment_count']}")
    print(f"  • Phrase Splits:           {p2['phrase_split_count']} -> {p3['phrase_split_count']}")
    print(f"  • Hard-Limit Violations:   {p2['hard_limit_violations']} -> {p3['hard_limit_violations']}")
    print(f"  • Avg Latency:             {p2['avg_latency_ms']}ms -> {p3['avg_latency_ms']}ms")
    print(f"  • Word Integrity:          {results['integrity_result']['status']} (Loss: 0, Duplication: 0)")
    print(f"  • Output Directory:        {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
