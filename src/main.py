"""
Main Pipeline Orchestrator for Phase 1.

Executes the offline NLP prototype pipeline:
1. Text loading & normalization
2. Word tokenization
3. Subtitle / sentence segmentation
4. Bigram & trigram extraction with frequencies
5. Meaningful recurring phrase filtering
6. Optional graceful Gemini semantic annotation
7. Structured output generation (JSON, CSV, and TXT report)
"""

import os
import csv
import json
from typing import Dict, Any, List

from src.text_processor import load_text, normalize_text
from src.tokenizer import tokenize_text
from src.segmenter import segment_text
from src.ngram_extractor import extract_bigrams, extract_trigrams, format_ngram_counter
from src.phrase_filter import filter_phrases
from src.gemini_client import GeminiSemanticClient


def run_pipeline(
    input_path: str,
    output_dir: str,
    min_phrase_freq: int = 2,
    enable_gemini: bool = True,
) -> Dict[str, Any]:
    """
    Execute the Phase 1 NLP pipeline and write outputs.
    
    Args:
        input_path: Path to the input transcript file.
        output_dir: Path to directory where output files will be written.
        min_phrase_freq: Minimum frequency threshold for recurring phrases.
        enable_gemini: Whether to attempt Gemini semantic enrichment if available.
        
    Returns:
        Dictionary of summary pipeline metrics and results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and normalize
    raw_text = load_text(input_path)
    normalized_text = normalize_text(raw_text)
    
    # 2. Tokenize
    tokens = tokenize_text(normalized_text)
    input_word_count = len(raw_text.split())
    token_count = len(tokens)
    
    # 3. Segment into subtitles
    segments = segment_text(normalized_text)
    segment_counts = [len(seg.split()) for seg in segments] if segments else [0]
    avg_segment_len = round(sum(segment_counts) / len(segment_counts), 2) if segments else 0.0
    max_segment_len = max(segment_counts) if segments else 0
    
    # 4. Extract n-grams
    bigram_counter = extract_bigrams(tokens)
    trigram_counter = extract_trigrams(tokens)
    
    unique_bigrams = len(bigram_counter)
    unique_trigrams = len(trigram_counter)
    
    bigrams_formatted = format_ngram_counter(bigram_counter)
    trigrams_formatted = format_ngram_counter(trigram_counter)
    
    # 5. Filter recurring phrases
    # Combine bigram and trigram counters for filtering
    combined_counter = bigram_counter + trigram_counter
    filtered_phrases = filter_phrases(combined_counter, min_freq=min_phrase_freq)
    
    # 6. Optional Gemini Semantic Enrichment
    gemini_client = GeminiSemanticClient()
    gemini_status = "Available (Active)" if gemini_client.is_available() else "Offline / Not Configured (Graceful Fallback)"
    phrase_names = [p["phrase"] for p in filtered_phrases[:10]]
    domain_annotations = {}
    if enable_gemini:
        domain_annotations = gemini_client.categorize_phrases(phrase_names)
        
    for p in filtered_phrases:
        p["semantic_domain"] = domain_annotations.get(p["phrase"], "Local Deterministic Extraction")
        
    # 7. Write Structured Outputs
    
    # (a) segmented_sentences.json
    segmented_data = [
        {
            "segment_id": i + 1,
            "text": seg,
            "word_count": len(seg.split()),
            "char_count": len(seg),
        }
        for i, seg in enumerate(segments)
    ]
    segmented_file = os.path.join(output_dir, "segmented_sentences.json")
    with open(segmented_file, "w", encoding="utf-8") as f:
        json.dump(segmented_data, f, indent=2)
        
    # (b) bigram_frequency.csv
    bigram_file = os.path.join(output_dir, "bigram_frequency.csv")
    with open(bigram_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "frequency", "n"])
        writer.writeheader()
        writer.writerows(bigrams_formatted)
        
    # (c) trigram_frequency.csv
    trigram_file = os.path.join(output_dir, "trigram_frequency.csv")
    with open(trigram_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "frequency", "n"])
        writer.writeheader()
        writer.writerows(trigrams_formatted)
        
    # (d) filtered_phrases.csv
    filtered_file = os.path.join(output_dir, "filtered_phrases.csv")
    with open(filtered_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "frequency", "n", "pos_pattern", "semantic_domain"])
        writer.writeheader()
        writer.writerows(filtered_phrases)
        
    # (e) phase1_report.txt
    report_file = os.path.join(output_dir, "phase1_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 1 OFFLINE NLP PROTOTYPE REPORT\n")
        f.write("Real-Time Subtitle Segmenter and N-gram Phrase Extractor\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("INPUT & TOKENIZATION METRICS:\n")
        f.write(f"- Input Source: {input_path}\n")
        f.write(f"- Number of input words (raw): {input_word_count}\n")
        f.write(f"- Number of tokens (normalized): {token_count}\n\n")
        
        f.write("SUBTITLE SEGMENTATION METRICS:\n")
        f.write(f"- Number of generated segments: {len(segments)}\n")
        f.write(f"- Average segment length: {avg_segment_len} words\n")
        f.write(f"- Maximum segment length: {max_segment_len} words\n\n")
        
        f.write("N-GRAM EXTRACTION METRICS:\n")
        f.write(f"- Total unique bigrams: {unique_bigrams}\n")
        f.write(f"- Total unique trigrams: {unique_trigrams}\n\n")
        
        f.write("TOP 10 BIGRAMS (RAW FREQUENCY):\n")
        for bg in bigrams_formatted[:10]:
            f.write(f"  • {bg['phrase']:<35} (freq: {bg['frequency']})\n")
        f.write("\n")
        
        f.write("TOP 10 TRIGRAMS (RAW FREQUENCY):\n")
        for tg in trigrams_formatted[:10]:
            f.write(f"  • {tg['phrase']:<35} (freq: {tg['frequency']})\n")
        f.write("\n")
        
        f.write("FILTERED MEANINGFUL RECURRING PHRASES:\n")
        f.write(f"- Total filtered recurring phrases (min_freq >= {min_phrase_freq}): {len(filtered_phrases)}\n")
        for fp in filtered_phrases:
            f.write(
                f"  • [{fp['n']}-gram] {fp['phrase']:<35} "
                f"freq: {fp['frequency']:<3} POS: {fp['pos_pattern']:<20} "
                f"domain: {fp['semantic_domain']}\n"
            )
        f.write("\n")
        
        f.write("SEMANTIC INTEGRATION STATUS:\n")
        f.write(f"- Gemini API: {gemini_status}\n")
        f.write("=" * 60 + "\n")
        
    return {
        "input_word_count": input_word_count,
        "token_count": token_count,
        "segment_count": len(segments),
        "avg_segment_length": avg_segment_len,
        "max_segment_length": max_segment_len,
        "unique_bigrams": unique_bigrams,
        "unique_trigrams": unique_trigrams,
        "filtered_phrase_count": len(filtered_phrases),
        "segments": segments,
        "bigrams": bigrams_formatted,
        "trigrams": trigrams_formatted,
        "filtered_phrases": filtered_phrases,
        "gemini_status": gemini_status,
        "output_files": [
            segmented_file,
            bigram_file,
            trigram_file,
            filtered_file,
            report_file,
        ],
    }
