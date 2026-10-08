"""
UI Session Management and Simulation Engine for Phase 5.

Coordinates the streaming simulation and incremental Phase 4 phrase extraction
for interactive UI presentation without duplicating any underlying NLP logic.
"""

import os
import csv
import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple

from src.text_processor import load_text, normalize_text
from src.tokenizer import tokenize_text
from src.stream_simulator import StreamSimulator
from src.streaming_segmenter import StreamingConfig
from src.improved_segmenter import ImprovedStreamingSegmenter
from src.streaming_phrase_extractor import StreamingPhraseExtractor
from src.phrase_protection import PhraseProtector
from src.gemini_client import GeminiSemanticClient


def highlight_phrases_in_text(
    text: str,
    ranked_phrases: List[Dict[str, Any]],
    max_phrases: int = 15,
) -> str:
    """
    Apply HTML styling to emphasize recurring phrases in subtitle text.
    
    Category styles:
    - HIGH: Bold with warm amber/gold background
    - MEDIUM: Bold with cool cyan/blue background
    - LOW: Subdued slate underline/border
    
    Args:
        text: Subtitle text string.
        ranked_phrases: List of ranked phrase dictionaries from Phase 4.
        max_phrases: Maximum number of top phrases to consider for highlighting.
        
    Returns:
        HTML formatted string with styled spans.
    """
    if not text or not ranked_phrases:
        return text

    # Filter to active phrases and sort by token length descending to match longest spans first
    candidates = sorted(
        ranked_phrases[:max_phrases],
        key=lambda p: len(p.get("phrase", "").split()),
        reverse=True,
    )

    highlighted = text
    # Styles mapped by quality category
    category_styles = {
        "HIGH": (
            "background: #fff3cd; color: #664d03; font-weight: 600; "
            "padding: 1px 6px; border-radius: 4px; border: 1px solid #ffe69c;"
        ),
        "MEDIUM": (
            "background: #cff4fc; color: #055160; font-weight: 600; "
            "padding: 1px 6px; border-radius: 4px; border: 1px solid #b6effb;"
        ),
        "LOW": (
            "background: #f8f9fa; color: #495057; text-decoration: underline; "
            "padding: 1px 4px; border-radius: 3px;"
        ),
    }

    # Replace case-insensitively using regex word boundaries
    for p in candidates:
        phrase_str = p.get("phrase", "")
        if not phrase_str:
            continue
        cat = p.get("quality_category", "LOW")
        style = category_styles.get(cat, category_styles["LOW"])
        pattern = re.compile(rf"\b({re.escape(phrase_str)})\b", re.IGNORECASE)
        replacement = rf'<span style="{style}">\1</span>'
        highlighted = pattern.sub(replacement, highlighted)

    return highlighted


class StreamingSimulationSession:
    """
    Manages the lifecycle of a real-time subtitle segmentation & phrase extraction session.
    """

    def __init__(
        self,
        transcript_text: str = "",
        config: Optional[StreamingConfig] = None,
        min_frequency: int = 2,
        top_k: int = 20,
    ):
        self.config = config or StreamingConfig()
        self.min_frequency = min_frequency
        self.top_k = top_k
        self.raw_text = transcript_text
        self.normalized_text = ""
        self.original_tokens: List[str] = []
        
        # State tracking
        self.status = "READY"  # READY, PROCESSING, PAUSED, COMPLETED
        self.chunk_index = 0
        self.total_chunks = 0
        self.all_chunks: List[Dict[str, Any]] = []
        self.emitted_segments: List[Dict[str, Any]] = []
        self.reconstructed_tokens: List[str] = []
        self.progressive_asr_text = ""
        
        # Backend components
        self.segmenter: Optional[ImprovedStreamingSegmenter] = None
        self.phrase_extractor: Optional[StreamingPhraseExtractor] = None
        self.current_rankings: Dict[str, Any] = {}
        self.pipeline_start_time = 0.0
        self.total_latency_ms = 0.0
        
        # Initialize components and chunks
        self.load_text(transcript_text)

    def load_text(self, text: str) -> None:
        """Load and normalize transcript text for a fresh simulation run."""
        self.raw_text = text or ""
        self.normalized_text = normalize_text(self.raw_text) if self.raw_text else ""
        self.original_tokens = tokenize_text(self.normalized_text) if self.normalized_text else []
        self.reset()

    def reset(self) -> None:
        """Reset session state to ready without losing loaded transcript."""
        self.status = "READY"
        self.chunk_index = 0
        self.emitted_segments = []
        self.reconstructed_tokens = []
        self.progressive_asr_text = ""
        self.current_rankings = {
            "ranked_phrases": [],
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "recurring_bigrams": 0,
            "recurring_trigrams": 0,
        }
        self.total_latency_ms = 0.0
        
        # Pre-generate simulated chunks from tokens
        self.all_chunks = []
        if self.original_tokens:
            step = self.config.chunk_size
            for idx, i in enumerate(range(0, len(self.original_tokens), step), start=1):
                chunk_slice = self.original_tokens[i : i + step]
                self.all_chunks.append({
                    "chunk_index": idx,
                    "text": " ".join(chunk_slice),
                    "tokens": chunk_slice,
                })
        self.total_chunks = len(self.all_chunks)

        # Initialize Phase 3 segmenter and Phase 4 phrase extractor
        self.segmenter = ImprovedStreamingSegmenter(config=self.config)
        self.phrase_extractor = StreamingPhraseExtractor()
        self.pipeline_start_time = time.perf_counter()

    def step(self) -> Dict[str, Any]:
        """
        Process the next single incoming ASR chunk.
        
        Returns:
            Dict containing step event details and updated state.
        """
        if self.chunk_index >= self.total_chunks:
            # End of stream reached
            return self._finalize_end_of_stream()

        start_step = time.perf_counter()
        chunk = self.all_chunks[self.chunk_index]
        self.chunk_index += 1
        self.status = "PROCESSING"

        # Update progressive ASR text
        if self.progressive_asr_text:
            self.progressive_asr_text += " " + chunk["text"]
        else:
            self.progressive_asr_text = chunk["text"]

        # 1. Feed chunk into Phase 3 Segmenter
        new_segments = self.segmenter.add_chunk(chunk)
        
        # 2. Feed emitted segments into Phase 4 StreamingPhraseExtractor
        for seg in new_segments:
            self.emitted_segments.append(seg)
            seg_tokens = tokenize_text(seg["text"])
            self.reconstructed_tokens.extend(seg_tokens)
            self.phrase_extractor.add_segment(seg["text"])

        # 3. Update current candidate rankings
        if new_segments or self.chunk_index % 3 == 0 or self.chunk_index == self.total_chunks:
            self.current_rankings = self.phrase_extractor.finalize_ranking(
                min_frequency=self.min_frequency,
                top_k=self.top_k,
            )

        step_lat = (time.perf_counter() - start_step) * 1000
        self.total_latency_ms += step_lat

        if self.chunk_index >= self.total_chunks:
            return self._finalize_end_of_stream()

        return {
            "chunk": chunk,
            "new_segments": new_segments,
            "is_complete": False,
            "progress_percent": round((self.chunk_index / max(1, self.total_chunks)) * 100, 1),
        }

    def _finalize_end_of_stream(self) -> Dict[str, Any]:
        """Flush final buffer from segmenter and finalize full session ranking."""
        final_segments = []
        if self.segmenter:
            final_segments = self.segmenter.flush(last_chunk_index=self.chunk_index)
            for seg in final_segments:
                self.emitted_segments.append(seg)
                seg_tokens = tokenize_text(seg["text"])
                self.reconstructed_tokens.extend(seg_tokens)
                self.phrase_extractor.add_segment(seg["text"])

        self.current_rankings = self.phrase_extractor.finalize_ranking(
            min_frequency=self.min_frequency,
            top_k=self.top_k,
        )
        self.status = "COMPLETED"

        return {
            "chunk": None,
            "new_segments": final_segments,
            "is_complete": True,
            "progress_percent": 100.0,
        }

    def run_all(self) -> Dict[str, Any]:
        """Fast-forward through entire simulation until completion."""
        while self.chunk_index < self.total_chunks:
            self.step()
        if self.status != "COMPLETED":
            self._finalize_end_of_stream()
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Generate structured summary of current session metrics."""
        inc_stats = self.phrase_extractor.get_incremental_stats() if self.phrase_extractor else {}
        ranked_list = self.current_rankings.get("ranked_phrases", [])
        
        return {
            "status": self.status,
            "total_tokens": len(self.original_tokens),
            "reconstructed_tokens": len(self.reconstructed_tokens),
            "word_loss": max(0, len(self.original_tokens) - len(self.reconstructed_tokens)),
            "total_chunks": self.total_chunks,
            "processed_chunks": self.chunk_index,
            "emitted_segments_count": len(self.emitted_segments),
            "unique_bigrams": self.current_rankings.get("unique_bigrams", 0),
            "unique_trigrams": self.current_rankings.get("unique_trigrams", 0),
            "recurring_bigrams": self.current_rankings.get("recurring_bigrams", 0),
            "recurring_trigrams": self.current_rankings.get("recurring_trigrams", 0),
            "ranked_phrases_count": len(ranked_list),
            "top_phrases": [p["phrase"] for p in ranked_list[:10]],
            "avg_incremental_update_ms": inc_stats.get("avg_incremental_update_ms", 0.0),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }

    def export_session(self, output_dir: str = os.path.join("data", "output", "phase5")) -> Dict[str, str]:
        """
        Export session artifacts to the designated output folder.
        
        Outputs:
        - session_segments.json
        - session_phrases.csv
        - session_summary.json
        
        Returns:
            Dict mapping artifact names to their absolute file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = self.get_summary()

        segments_path = os.path.join(output_dir, "session_segments.json")
        phrases_path = os.path.join(output_dir, "session_phrases.csv")
        summary_path = os.path.join(output_dir, "session_summary.json")

        # 1. Segments JSON
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(self.emitted_segments, f, indent=2)

        # 2. Phrases CSV
        ranked_phrases = self.current_rankings.get("ranked_phrases", [])
        with open(phrases_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "phrase", "ngram_size", "frequency",
                "association_score", "POS_pattern", "content_word_ratio",
                "quality_score", "quality_category", "parent_phrase"
            ])
            for p in ranked_phrases:
                writer.writerow([
                    p.get("rank", 0),
                    p.get("phrase", ""),
                    p.get("ngram_size", 0),
                    p.get("frequency", 0),
                    p.get("pmi", 0.0),
                    p.get("pos_pattern", ""),
                    p.get("content_word_ratio", 0.0),
                    p.get("quality_score", 0.0),
                    p.get("quality_category", ""),
                    p.get("parent_phrase", "none"),
                ])

        # 3. Summary JSON
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "segments_json": segments_path,
            "phrases_csv": phrases_path,
            "summary_json": summary_path,
        }
