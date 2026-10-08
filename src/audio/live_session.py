"""
Live Microphone Session Module for Phase 7.

Coordinates the end-to-end live microphone streaming session:
LiveMicrophoneCapture → LocalASR → ConservativeOverlapHandler →
ImprovedStreamingSegmenter → StreamingPhraseExtractor → Live Streamlit Display.

Guarantees:
- Model loaded once and reused across all live chunks
- Monotonic timestamps for audio chunks and emitted subtitles
- Conservative boundary overlap removal to eliminate duplicate words between adjacent ASR chunks
- Legitimate repeated words (e.g. 'very very important') preserved
- Reuses Phase 3 subtitle segmenter and Phase 4 phrase extractor without duplicating logic
- Live metrics: elapsed duration, ASR latency, NLP latency, E2E latency, and token throughput
- Local exports (TXT, JSON, CSV)
"""

import csv
import io
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

from src.audio.chunker import AudioChunk
from src.audio.asr import LocalASR, ASRChunkResult, clean_asr_text
from src.audio.microphone import LiveMicrophoneCapture, LiveMicrophoneConfig, MicrophoneDeviceInfo
from src.audio.overlap import ConservativeOverlapHandler, OverlapResolution
from src.improved_segmenter import ImprovedStreamingSegmenter
from src.streaming_segmenter import StreamingConfig
from src.streaming_phrase_extractor import StreamingPhraseExtractor
from src.tokenizer import tokenize_text
from src.text_processor import normalize_text


@dataclass
class LiveStepEvent:
    """Event emitted whenever a live microphone chunk is processed."""
    chunk_index: int
    start_time: float
    end_time: float
    raw_text: str
    cleaned_text: str
    overlap_info: Dict[str, Any]
    new_subtitles: List[Dict[str, Any]]
    all_subtitles: List[Dict[str, Any]]
    ranked_phrases: List[Dict[str, Any]]
    phrase_statistics: Dict[str, Any]
    asr_latency_ms: float
    nlp_latency_ms: float
    e2e_latency_ms: float
    is_stopped: bool


class LiveMicrophoneSession:
    """
    Session manager for live microphone audio captioning and phrase extraction.
    """

    def __init__(
        self,
        chunk_duration_seconds: float = 3.0,
        model_size: str = "base",
        device_index: Optional[int] = None,
        segmenter_config: Optional[StreamingConfig] = None,
        min_frequency: int = 2,
        top_k: int = 20,
        asr_instance: Optional[LocalASR] = None,
        capture_instance: Optional[LiveMicrophoneCapture] = None,
    ):
        self.chunk_duration = float(chunk_duration_seconds)
        self.model_size = model_size
        self.device_index = device_index
        self.min_frequency = min_frequency
        self.top_k = top_k
        self.segmenter_config = segmenter_config or StreamingConfig()

        # Audio capture & processing components
        self.capture_config = LiveMicrophoneConfig(
            chunk_duration_seconds=self.chunk_duration,
            device_index=self.device_index,
        )
        self.capture = capture_instance or LiveMicrophoneCapture(config=self.capture_config)
        self.asr = asr_instance or LocalASR(model_size=self.model_size)
        self.overlap_handler = ConservativeOverlapHandler()
        self.segmenter = ImprovedStreamingSegmenter(config=self.segmenter_config)
        self.phrase_extractor = StreamingPhraseExtractor()

        # Session state
        self.status = "IDLE"  # IDLE, LISTENING, PROCESSING, STOPPED, ERROR
        self.error_message: Optional[str] = None
        self.previous_raw_text = ""
        self.previous_cleaned_text = ""

        # Accumulated data
        self.chunk_history: List[Dict[str, Any]] = []
        self.emitted_subtitles: List[Dict[str, Any]] = []
        self.reconstructed_tokens: List[str] = []
        self.current_rankings: Dict[str, Any] = {
            "ranked_phrases": [],
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "recurring_bigrams": 0,
            "recurring_trigrams": 0,
        }

        # Performance measurements
        self.session_start_time = 0.0
        self.session_stop_time = 0.0
        self.asr_latencies_ms: List[float] = []
        self.nlp_latencies_ms: List[float] = []
        self.e2e_latencies_ms: List[float] = []
        self.total_audio_seconds = 0.0

    @property
    def elapsed_live_seconds(self) -> float:
        """Elapsed wall-clock time since session started."""
        if self.session_start_time == 0.0:
            return 0.0
        if self.session_stop_time > 0.0:
            return self.session_stop_time - self.session_start_time
        return time.perf_counter() - self.session_start_time

    def start(self) -> None:
        """
        Start the microphone recording stream and transition state to LISTENING.
        """
        self.reset()
        self.capture.start()
        self.status = "LISTENING"
        self.session_start_time = time.perf_counter()
        self.session_stop_time = 0.0

    def stop(self) -> LiveStepEvent:
        """
        Stop microphone capture, process remaining safe audio buffer, and flush segmenter.
        """
        self.session_stop_time = time.perf_counter()
        self.capture.stop()
        self.status = "STOPPED"

        # Check for remaining audio slice in buffer
        remaining_chunk = self.capture.get_remaining_chunk(min_duration_seconds=0.5)
        if remaining_chunk is not None:
            self._process_chunk(remaining_chunk)

        # Flush any trailing tokens in segmenter buffer
        trailing_segs = self.segmenter.flush()
        last_end_time = self.total_audio_seconds
        new_subtitles = []

        for seg in trailing_segs:
            seg_dict = dict(seg)
            seg_dict["start_time"] = seg_dict.get("timestamp", max(0.0, last_end_time - 1.0))
            seg_dict["end_time"] = last_end_time
            self.emitted_subtitles.append(seg_dict)
            new_subtitles.append(seg_dict)
            self.phrase_extractor.add_segment(seg_dict["text"])

        # Finalize phrase rankings
        self.current_rankings = self.phrase_extractor.finalize_ranking(
            min_frequency=self.min_frequency,
            top_k=self.top_k,
        )

        return LiveStepEvent(
            chunk_index=len(self.chunk_history),
            start_time=0.0,
            end_time=self.total_audio_seconds,
            raw_text="",
            cleaned_text="",
            overlap_info={"reason": "session_stopped"},
            new_subtitles=new_subtitles,
            all_subtitles=self.emitted_subtitles,
            ranked_phrases=self.current_rankings.get("ranked_phrases", []),
            phrase_statistics=self.phrase_extractor.get_incremental_stats(),
            asr_latency_ms=0.0,
            nlp_latency_ms=0.0,
            e2e_latency_ms=0.0,
            is_stopped=True,
        )

    def reset(self) -> None:
        """
        Reset live session state, buffers, counters, and timeline.
        Does NOT delete or reload the cached ASR model.
        """
        self.capture.reset()
        self.status = "IDLE"
        self.error_message = None
        self.previous_raw_text = ""
        self.previous_cleaned_text = ""
        self.chunk_history.clear()
        self.emitted_subtitles.clear()
        self.reconstructed_tokens.clear()
        self.current_rankings = {
            "ranked_phrases": [],
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "recurring_bigrams": 0,
            "recurring_trigrams": 0,
        }
        self.session_start_time = 0.0
        self.session_stop_time = 0.0
        self.asr_latencies_ms.clear()
        self.nlp_latencies_ms.clear()
        self.e2e_latencies_ms.clear()
        self.total_audio_seconds = 0.0

        # Re-initialize NLP state machines
        self.segmenter = ImprovedStreamingSegmenter(config=self.segmenter_config)
        self.phrase_extractor = StreamingPhraseExtractor()

    def step(self) -> Optional[LiveStepEvent]:
        """
        Check if an audio chunk is available from the capture stream and process it.
        
        Returns:
            LiveStepEvent if a chunk was ready and processed, else None.
        """
        if self.status not in ("LISTENING", "PROCESSING"):
            return None

        chunk = self.capture.get_chunk()
        if chunk is None:
            return None

        self.status = "PROCESSING"
        event = self._process_chunk(chunk)
        self.status = "LISTENING"
        return event

    def _process_chunk(self, chunk: AudioChunk) -> LiveStepEvent:
        """
        Core pipeline step for an individual live audio chunk.
        """
        t_capture_end = time.perf_counter()
        self.total_audio_seconds = chunk.end_time

        # 1. ASR Transcription
        t_asr_start = time.perf_counter()
        asr_res = self.asr.transcribe_chunk(chunk)
        asr_lat_ms = (time.perf_counter() - t_asr_start) * 1000.0
        self.asr_latencies_ms.append(asr_lat_ms)

        raw_text = asr_res.text.strip()
        cleaned_text = clean_asr_text(raw_text)

        # 2. Conservative Overlap Removal
        overlap_res = self.overlap_handler.resolve_overlap(
            previous_text=self.previous_cleaned_text,
            current_text=cleaned_text,
        )
        final_chunk_text = overlap_res.cleaned_text
        self.previous_cleaned_text = cleaned_text
        self.previous_raw_text = raw_text

        # 3. NLP Ingestion (Subtitle Segmentation + Phrase Extraction)
        t_nlp_start = time.perf_counter()
        new_subtitles: List[Dict[str, Any]] = []

        norm_text = normalize_text(final_chunk_text, lower=True) if final_chunk_text else ""
        tokens = tokenize_text(norm_text) if norm_text else []
        self.reconstructed_tokens.extend(tokens)

        if tokens:
            chunk_dict = {
                "chunk_index": chunk.chunk_index,
                "text": norm_text,
                "tokens": tokens,
                "timestamp": chunk.start_time,
            }
            raw_new_segs = self.segmenter.add_chunk(chunk_dict)
            for seg in raw_new_segs:
                seg_dict = dict(seg)
                seg_dict["start_time"] = chunk.start_time
                seg_dict["end_time"] = chunk.end_time
                self.emitted_subtitles.append(seg_dict)
                new_subtitles.append(seg_dict)
                self.phrase_extractor.add_segment(seg_dict["text"])

        # Update phrase rankings
        self.current_rankings = self.phrase_extractor.finalize_ranking(
            min_frequency=self.min_frequency,
            top_k=self.top_k,
        )

        nlp_lat_ms = (time.perf_counter() - t_nlp_start) * 1000.0
        self.nlp_latencies_ms.append(nlp_lat_ms)

        e2e_lat_ms = (time.perf_counter() - t_capture_end) * 1000.0
        self.e2e_latencies_ms.append(e2e_lat_ms)

        # Record chunk history
        chunk_record = {
            "chunk_index": chunk.chunk_index,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "duration": chunk.duration,
            "raw_text": raw_text,
            "cleaned_text": final_chunk_text,
            "overlap_tokens_removed": overlap_res.overlap_token_count,
            "overlap_phrase": overlap_res.overlap_phrase,
            "asr_latency_ms": round(asr_lat_ms, 2),
            "nlp_latency_ms": round(nlp_lat_ms, 2),
            "e2e_latency_ms": round(e2e_lat_ms, 2),
        }
        self.chunk_history.append(chunk_record)

        return LiveStepEvent(
            chunk_index=chunk.chunk_index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            raw_text=raw_text,
            cleaned_text=final_chunk_text,
            overlap_info={
                "tokens_removed": overlap_res.overlap_token_count,
                "overlap_phrase": overlap_res.overlap_phrase,
                "reason": overlap_res.reason,
            },
            new_subtitles=new_subtitles,
            all_subtitles=self.emitted_subtitles,
            ranked_phrases=self.current_rankings.get("ranked_phrases", []),
            phrase_statistics=self.phrase_extractor.get_incremental_stats(),
            asr_latency_ms=round(asr_lat_ms, 2),
            nlp_latency_ms=round(nlp_lat_ms, 2),
            e2e_latency_ms=round(e2e_lat_ms, 2),
            is_stopped=False,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Compute structured live session summary KPIs."""
        avg_asr_lat = float(np.mean(self.asr_latencies_ms)) if self.asr_latencies_ms else 0.0
        avg_nlp_lat = float(np.mean(self.nlp_latencies_ms)) if self.nlp_latencies_ms else 0.0
        avg_e2e_lat = float(np.mean(self.e2e_latencies_ms)) if self.e2e_latencies_ms else 0.0

        phrases = self.current_rankings.get("ranked_phrases", [])
        quality_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for p in phrases:
            q = p.get("quality_category", "LOW")
            if q in quality_counts:
                quality_counts[q] += 1

        full_transcript = " ".join(s.get("text", "") for s in self.emitted_subtitles).strip()

        return {
            "status": self.status,
            "live_duration_seconds": round(self.elapsed_live_seconds, 2),
            "audio_duration_seconds": round(self.total_audio_seconds, 2),
            "chunks_processed": len(self.chunk_history),
            "subtitles_generated": len(self.emitted_subtitles),
            "tokens_processed": len(self.reconstructed_tokens),
            "ranked_phrases_count": len(phrases),
            "quality_distribution": quality_counts,
            "unique_bigrams": self.current_rankings.get("unique_bigrams", 0),
            "unique_trigrams": self.current_rankings.get("unique_trigrams", 0),
            "recurring_bigrams": self.current_rankings.get("recurring_bigrams", 0),
            "recurring_trigrams": self.current_rankings.get("recurring_trigrams", 0),
            "average_asr_latency_ms": round(avg_asr_lat, 2),
            "average_nlp_latency_ms": round(avg_nlp_lat, 2),
            "average_e2e_latency_ms": round(avg_e2e_lat, 2),
            "asr_model_size": self.model_size,
            "full_transcript": full_transcript,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Alias for get_summary providing standardized API across audio sessions."""
        return self.get_summary()

    def export_txt(self) -> str:
        """Generate human-readable text report containing timestamped subtitles and transcript."""
        summary = self.get_summary()
        lines = [
            "=" * 60,
            "REAL-TIME SUBTITLE SEGMENTER — LIVE MICROPHONE SESSION",
            "=" * 60,
            f"Status: {summary['status']}",
            f"Live Duration: {summary['live_duration_seconds']}s",
            f"Audio Captured: {summary['audio_duration_seconds']}s",
            f"Chunks Processed: {summary['chunks_processed']}",
            f"Subtitles Emitted: {summary['subtitles_generated']}",
            f"Tokens Processed: {summary['tokens_processed']}",
            f"Average ASR Latency: {summary['average_asr_latency_ms']} ms",
            f"Average E2E Latency: {summary['average_e2e_latency_ms']} ms",
            f"ASR Model: {self.model_size}",
            "-" * 60,
            "TIMESTAMPED SUBTITLE SEGMENTS:",
            "-" * 60,
        ]

        for idx, seg in enumerate(self.emitted_subtitles, 1):
            st = seg.get("start_time", 0.0)
            et = seg.get("end_time", 0.0)
            sm, ss = int(st // 60), st % 60
            em, es = int(et // 60), et % 60
            lines.append(f"[{sm:02d}:{ss:04.1f} -> {em:02d}:{es:04.1f}] #{idx}: {seg.get('text', '')}")

        lines.extend([
            "-" * 60,
            "COMPLETE RECONSTRUCTED TRANSCRIPT:",
            "-" * 60,
            summary.get("full_transcript", ""),
            "-" * 60,
            "TOP RECURRING PHRASES:",
            "-" * 60,
        ])

        for p in self.current_rankings.get("ranked_phrases", [])[:20]:
            lines.append(
                f"- {p.get('phrase')} ({p.get('ngram_size')}-gram) | Freq: {p.get('frequency')} | "
                f"Score: {p.get('quality_score')} [{p.get('quality_category')}]"
            )

        return "\n".join(lines)

    def export_json(self) -> str:
        """Generate structured JSON session payload."""
        payload = {
            "session_type": "live_microphone",
            "summary": self.get_summary(),
            "config": {
                "chunk_duration_seconds": self.chunk_duration,
                "model_size": self.model_size,
                "device_index": self.device_index,
                "min_frequency": self.min_frequency,
                "top_k": self.top_k,
            },
            "chunks": self.chunk_history,
            "subtitles": self.emitted_subtitles,
            "ranked_phrases": self.current_rankings.get("ranked_phrases", []),
            "phrase_statistics": self.current_rankings,
        }
        return json.dumps(payload, indent=2)

    def export_csv(self) -> str:
        """Generate CSV string of ranked recurring phrases."""
        phrases = self.current_rankings.get("ranked_phrases", [])
        output = io.StringIO()
        fieldnames = [
            "rank", "phrase", "ngram_size", "frequency", "pmi_score",
            "quality_score", "quality_category", "pos_pattern", "parent_phrase"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(phrases, 1):
            writer.writerow({
                "rank": idx,
                "phrase": p.get("phrase", ""),
                "ngram_size": p.get("ngram_size", 2),
                "frequency": p.get("frequency", 0),
                "pmi_score": p.get("pmi_score", 0.0),
                "quality_score": p.get("quality_score", 0.0),
                "quality_category": p.get("quality_category", "LOW"),
                "pos_pattern": p.get("pos_pattern", ""),
                "parent_phrase": p.get("parent_phrase", "") or "",
            })
        return output.getvalue()
