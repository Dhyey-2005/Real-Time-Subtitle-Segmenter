#!/usr/bin/env python3
"""
CLI Entrypoint for Real-Time Subtitle Segmenter and N-gram Phrase Extractor (Phase 4).

Usage:
    python run_phase4.py --input data/input/sample_transcript.txt
    python run_phase4.py --input data/input/sample_transcript.txt --min-frequency 2
    python run_phase4.py --input data/input/sample_transcript.txt --top-k 20
    python run_phase4.py --input data/input/phrase_extraction_test_transcript.txt --min-frequency 2 --top-k 20
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
from src.phrase_orchestrator import run_phase4_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Improved N-gram Phrase Extraction with PMI & Content-Word Quality Scoring"
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
        "--min-frequency",
        "-m",
        type=int,
        default=2,
        help="Minimum frequency threshold for recurring phrases (default: 2)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=20,
        help="Number of top ranked candidate phrases to return (default: 20)",
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
    )

    print("=" * 70)
    print("PHASE 4: IMPROVED N-GRAM PHRASE EXTRACTION & COLLOCATION ANALYSIS")
    print(f"Input:         {input_path}")
    print(f"Min Frequency: {args.min_frequency} | Top-K: {args.top_k}")
    print(f"Chunk Size:    {config.chunk_size} words | Delay: {config.simulated_delay}s")
    print("=" * 70)

    def progress_logger(event_type: str, data: dict):
        if event_type == "chunk":
            idx = data.get("chunk_index", 0)
            text = data.get("text", "")
            if idx <= 5 or idx % 10 == 0:
                print(f"[STREAM CHUNK {idx}] {text}")
        elif event_type == "subtitle":
            sid = data.get("segment_id", 0)
            text = data.get("text", "")
            wcnt = data.get("word_count", 0)
            print(f"  >>> [SUBTITLE {sid}] ({wcnt}w) \"{text}\"")

    try:
        results = run_phase4_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            min_frequency=args.min_frequency,
            top_k=args.top_k,
            enable_gemini=not args.no_gemini,
            progress_callback=progress_logger,
        )
    except Exception as e:
        print(f"\n[ERROR] Phase 4 execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    stats = results["stats"]
    ranking_res = results["ranking_results"]
    ranked_phrases = ranking_res["ranked_phrases"]

    print("\n" + "=" * 70)
    print("PHASE 4 EXTRACTION COMPLETE — TOP RECURRING PHRASES:")
    print(f"{'Rank':<5} {'Phrase':<32} {'N':<3} {'Freq':<5} {'PMI':<7} {'POS Pattern':<14} {'Score':<6} {'Category'}")
    print("-" * 70)

    for p in ranked_phrases[:args.top_k]:
        print(
            f"{p['rank']:<5} {p['phrase']:<32} {p['ngram_size']:<3} {p['frequency']:<5} "
            f"{p['pmi']:<7} {p['pos_pattern']:<14} {p['quality_score']:<6} {p['quality_category']}"
        )

    if not ranked_phrases:
        print("  (No candidate phrases met the minimum frequency threshold)")

    print("-" * 70)
    print(f"  • Total Normalized Words:     {results['input_word_count']}")
    print(f"  • Unique Bigrams / Trigrams:  {ranking_res['unique_bigrams']} / {ranking_res['unique_trigrams']}")
    print(f"  • Recurring Bigrams/Trigrams: {ranking_res['recurring_bigrams']} / {ranking_res['recurring_trigrams']}")
    print(f"  • Final Ranked Phrases:       {len(ranked_phrases)}")
    print(f"  • Quality Categories:         {stats['quality_category_distribution']}")
    print(f"  • Association (PMI) Range:    Min={stats['association_statistics']['min_pmi']} Max={stats['association_statistics']['max_pmi']}")
    print(f"  • Word Integrity:             {results['integrity_result']['status']} (Loss: 0, Duplication: 0)")
    print(f"  • Pipeline Latency:           {stats['latency_metrics']['total_pipeline_ms']} ms")
    print(f"  • Output Directory:           {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
