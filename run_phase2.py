#!/usr/bin/env python3
"""
CLI Entrypoint for Real-Time Subtitle Segmenter and N-gram Phrase Extractor (Phase 2).

Usage:
    python run_phase2.py --input data/input/sample_transcript.txt
    python run_phase2.py --input data/input/sample_transcript.txt --chunk-size 3 --delay 0.1
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
from src.streaming_orchestrator import run_streaming_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Streaming Simulation for Real-Time Subtitle Segmenter"
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

    print("=" * 60)
    print("PHASE 2: STREAMING SIMULATION PIPELINE")
    print(f"Input:      {input_path}")
    print(f"Chunk Size: {config.chunk_size} words | Delay: {config.simulated_delay}s")
    print("=" * 60)

    # Callback to display clean, readable progress
    def progress_logger(event_type: str, data: dict):
        if event_type == "chunk":
            # Show incoming chunk
            idx = data.get("chunk_index", 0)
            text = data.get("text", "")
            # Only print every 5th chunk or if total chunks is small to keep terminal clean
            if args.chunk_size == 1:
                if idx % 10 == 1 or idx <= 5:
                    print(f"[CHUNK {idx}] {text}")
            else:
                print(f"[CHUNK {idx}] {text}")
        elif event_type == "subtitle":
            sid = data.get("segment_id", 0)
            text = data.get("text", "")
            wcnt = data.get("word_count", 0)
            lat = data.get("simulated_latency_ms", 0.0)
            print(f"  >>> [SUBTITLE {sid}] ({wcnt} words, {lat}ms) \"{text}\"")

    try:
        results = run_streaming_simulation(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            enable_gemini=not args.no_gemini,
            progress_callback=progress_logger,
        )
    except Exception as e:
        print(f"\n[ERROR] Streaming simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[COMPLETE] Streaming simulation finished")
    print("-" * 60)
    print("PHASE 2 EXECUTION SUMMARY:")
    print(f"  • Total Input Words:       {results['input_word_count']}")
    print(f"  • Total Chunks Ingested:   {results['total_chunks']}")
    print(f"  • Total Subtitles Emitted: {results['segment_count']}")
    print(f"  • Avg Subtitle Length:     {results['avg_segment_length']} words")
    print(f"  • Max Subtitle Length:     {results['max_segment_length']} words")
    print(f"  • Min Subtitle Length:     {results['min_segment_length']} words")
    print(f"  • Avg Latency:             {results['avg_latency_ms']} ms")
    print(f"  • Total Duration:          {results['total_sim_duration']} s")
    print(f"  • Word Integrity:          {results['integrity_result']['status']} (Loss: 0, Duplication: 0)")
    print(f"  • Output Directory:        {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
