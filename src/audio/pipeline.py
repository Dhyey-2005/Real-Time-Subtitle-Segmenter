"""
Audio Streaming Pipeline Module for Phase 6 & Phase 8.

Coordinates progressive audio streaming:
Audio Chunk → Local ASR → Speaker Diarization Alignment → Existing Subtitle Segmenter → Existing Phrase Extractor.

Strictly reuses the existing Phase 3 ImprovedStreamingSegmenter and Phase 4
StreamingPhraseExtractor without duplicating any underlying NLP algorithms.
Supports Phase 8 Speaker Diarization (automatic or known speaker count) with
temporal overlap alignment, speaker-change subtitle boundaries, and speaker-aware exports.
"""

import csv
import io
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, BinaryIO

import numpy as np

from src.audio.loader import AudioMetadata, load_audio, AudioLoadError
from src.audio.chunker import AudioChunk, AudioChunker
from src.audio.asr import LocalASR, ASRChunkResult
from src.audio.diarization import (
    BaseDiarizer,
    SpeakerSegment,
    get_diarizer,
    align_asr_with_diarization,
    compute_diarization_summary,
    DIARIZATION_MODELS_DIR,
)
from src.improved_segmenter import ImprovedStreamingSegmenter
from src.streaming_segmenter import StreamingConfig
from src.streaming_phrase_extractor import StreamingPhraseExtractor
from src.tokenizer import tokenize_text
from src.text_processor import normalize_text


@dataclass
class AudioStreamStepEvent:
    """Event payload emitted after processing each audio chunk."""
    chunk_index: int
    total_chunks: int
    start_time: float
    end_time: float
    chunk_text: str
    new_segments: List[Dict[str, Any]]
    all_segments: List[Dict[str, Any]]
    ranked_phrases: List[Dict[str, Any]]
    phrase_statistics: Dict[str, Any]
    progress_percent: float
    is_complete: bool
    processing_time_ms: float
    diarization_info: Optional[Dict[str, Any]] = None


class AudioStreamingSession:
    """
    Manages the progressive audio streaming simulation session with optional speaker diarization.
    """

    def __init__(
        self,
        chunk_duration_seconds: float = 5.0,
        model_size: str = "base",
        segmenter_config: Optional[StreamingConfig] = None,
        min_frequency: int = 2,
        top_k: int = 20,
        asr_instance: Optional[LocalASR] = None,
        enable_diarization: bool = False,
        num_speakers: Optional[int] = None,
        diarizer_instance: Optional[BaseDiarizer] = None,
    ):
        self.chunk_duration = float(chunk_duration_seconds)
        self.model_size = model_size
        self.min_frequency = min_frequency
        self.top_k = top_k
        self.segmenter_config = segmenter_config or StreamingConfig()

        # Diarization config
        self.enable_diarization = enable_diarization
        self.num_speakers = num_speakers
        self.diarizer = diarizer_instance
        if self.enable_diarization and self.diarizer is None:
            self.diarizer = get_diarizer(models_dir=DIARIZATION_MODELS_DIR)

        # Components
        self.asr = asr_instance or LocalASR(model_size=model_size)
        self.chunker = AudioChunker(chunk_duration_seconds=self.chunk_duration)
        self.segmenter = ImprovedStreamingSegmenter(config=self.segmenter_config)
        self.phrase_extractor = StreamingPhraseExtractor()

        # State
        self.status = "IDLE"  # IDLE, READY, PROCESSING, PAUSED, COMPLETED
        self.metadata: Optional[AudioMetadata] = None
        self.audio_samples: Optional[np.ndarray] = None
        self.chunks: List[AudioChunk] = []
        self.current_chunk_idx = 0
        self.total_chunks = 0
        self.speaker_segments: List[SpeakerSegment] = []
        self._current_speaker: str = "Speaker 1"

        # Accumulated outputs
        self.asr_chunk_results: List[ASRChunkResult] = []
        self.emitted_segments: List[Dict[str, Any]] = []
        self.reconstructed_tokens: List[str] = []
        self.current_rankings: Dict[str, Any] = {
            "ranked_phrases": [],
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "recurring_bigrams": 0,
            "recurring_trigrams": 0,
        }

        # Timing & Performance
        self.session_start_time = 0.0
        self.total_processing_time_s = 0.0
        self.diarization_time_s = 0.0
        self.asr_latencies_ms: List[float] = []

    def load_audio_source(
        self,
        file_source: Union[str, bytes, BinaryIO],
        filename: str = "audio.wav",
        chunk_duration_sec: Optional[float] = None,
    ) -> AudioMetadata:
        """
        Load and decode an audio file, generate chunks, perform diarization if enabled, and prepare session.
        """
        # Clear previous session state to prevent stale data retention
        self.audio_samples = None
        self.metadata = None
        self.chunks = []
        self.total_chunks = 0
        self.speaker_segments = []
        self.diarization_time_s = 0.0
        self.reset()

        if chunk_duration_sec is not None:
            self.chunk_duration = float(chunk_duration_sec)
            self.chunker = AudioChunker(chunk_duration_seconds=self.chunk_duration)

        samples, meta = load_audio(file_source=file_source, filename=filename)
        self.audio_samples = samples
        self.metadata = meta
        self.chunks = self.chunker.split(samples)
        self.total_chunks = len(self.chunks)

        # Run diarization once across the complete uploaded audio if enabled
        if self.enable_diarization and self.diarizer is not None:
            t_diar_start = time.perf_counter()
            self.speaker_segments = self.diarizer.diarize(
                samples=samples,
                sample_rate=16000,
                num_speakers=self.num_speakers,
            )
            self.diarization_time_s = time.perf_counter() - t_diar_start
        else:
            self.speaker_segments = []
            self.diarization_time_s = 0.0

        self.reset()
        return meta

    def reset(self) -> None:
        """Reset streaming state back to initial without discarding loaded audio."""
        self.status = "READY" if self.total_chunks > 0 else "IDLE"
        self.current_chunk_idx = 0
        self.asr_chunk_results = []
        self.emitted_segments = []
        self.reconstructed_tokens = []
        self._current_speaker = "Speaker 1"
        self.current_rankings = {
            "ranked_phrases": [],
            "unique_bigrams": 0,
            "unique_trigrams": 0,
            "recurring_bigrams": 0,
            "recurring_trigrams": 0,
        }
        self.total_processing_time_s = 0.0
        self.asr_latencies_ms = []

        # Re-initialize NLP state machines
        self.segmenter = ImprovedStreamingSegmenter(config=self.segmenter_config)
        self.phrase_extractor = StreamingPhraseExtractor()

    def step(self) -> AudioStreamStepEvent:
        """
        Process the next sequential audio chunk through ASR and the NLP pipeline.
        
        Returns:
            AudioStreamStepEvent describing the incremental progress.
        """
        if self.status == "COMPLETED":
            return None
        if self.current_chunk_idx >= self.total_chunks:
            return self._finalize_session()

        step_t0 = time.perf_counter()
        chunk = self.chunks[self.current_chunk_idx]
        self.current_chunk_idx += 1
        self.status = "PROCESSING"

        # 1. Transcribe audio chunk with local ASR
        asr_res = self.asr.transcribe_chunk(chunk)
        self.asr_chunk_results.append(asr_res)
        self.asr_latencies_ms.append(asr_res.processing_time_ms)

        # 2. Align ASR timestamps with speaker diarization intervals if enabled
        aligned_asr_segments = []
        chunk_speaker = self._current_speaker

        if self.enable_diarization and self.speaker_segments:
            if asr_res.segments:
                aligned_asr_segments = align_asr_with_diarization(
                    asr_segments=asr_res.segments,
                    speaker_segments=self.speaker_segments,
                    default_speaker=self._current_speaker,
                )
            else:
                # Align entire chunk
                c_seg = [{"start": chunk.start_time, "end": chunk.end_time, "text": asr_res.text}]
                aligned = align_asr_with_diarization(c_seg, self.speaker_segments, default_speaker=self._current_speaker)
                aligned_asr_segments = aligned
                chunk_speaker = aligned[0].get("speaker", self._current_speaker)
        else:
            chunk_speaker = "Speaker 1"

        # 3. Tokenize recognized text for deterministic NLP stream
        chunk_text = asr_res.text
        norm_text = normalize_text(chunk_text, lower=True) if chunk_text else ""
        tokens = tokenize_text(norm_text) if norm_text else []
        self.reconstructed_tokens.extend(tokens)

        # 4. Feed chunk into existing Phase 3 subtitle segmenter
        new_segments: List[Dict[str, Any]] = []

        if tokens:
            # Handle speaker changes as natural subtitle boundaries
            if self.enable_diarization and aligned_asr_segments:
                # Process segment by segment to respect speaker transitions
                for asr_sub in aligned_asr_segments:
                    sub_spk = asr_sub.get("speaker", self._current_speaker)
                    sub_text = normalize_text(asr_sub.get("text", ""), lower=True)
                    sub_tokens = tokenize_text(sub_text)

                    if not sub_tokens:
                        continue

                    # Trigger boundary flush if speaker changed and buffer has tokens
                    if sub_spk != self._current_speaker and self.segmenter.buffer.word_count() > 0:
                        flushed = self.segmenter.flush()
                        for f_seg in flushed:
                            f_dict = dict(f_seg)
                            f_dict["start_time"] = chunk.start_time
                            f_dict["end_time"] = round(asr_sub.get("start", chunk.start_time), 2)
                            f_dict["speaker"] = self._current_speaker
                            f_dict["reason"] = "speaker_change"
                            self.emitted_segments.append(f_dict)
                            new_segments.append(f_dict)
                            self.phrase_extractor.add_segment(f_dict["text"])

                    self._current_speaker = sub_spk
                    chunk_dict = {
                        "chunk_index": chunk.chunk_index,
                        "text": sub_text,
                        "tokens": sub_tokens,
                        "timestamp": asr_sub.get("start", chunk.start_time),
                    }
                    raw_new_segs = self.segmenter.add_chunk(chunk_dict)
                    for seg in raw_new_segs:
                        seg_dict = dict(seg)
                        seg_dict["start_time"] = asr_sub.get("start", chunk.start_time)
                        seg_dict["end_time"] = asr_sub.get("end", chunk.end_time)
                        seg_dict["speaker"] = sub_spk
                        self.emitted_segments.append(seg_dict)
                        new_segments.append(seg_dict)
                        self.phrase_extractor.add_segment(seg_dict["text"])
            else:
                self._current_speaker = chunk_speaker
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
                    seg_dict["speaker"] = chunk_speaker
                    self.emitted_segments.append(seg_dict)
                    new_segments.append(seg_dict)
                    self.phrase_extractor.add_segment(seg_dict["text"])

        # Update phrase rankings and statistics
        self.current_rankings = self.phrase_extractor.finalize_ranking(
            min_frequency=self.min_frequency,
            top_k=self.top_k,
        )

        step_elapsed_s = time.perf_counter() - step_t0
        self.total_processing_time_s += step_elapsed_s

        progress_pct = (self.current_chunk_idx / self.total_chunks) * 100.0 if self.total_chunks > 0 else 0.0
        is_complete = (self.current_chunk_idx >= self.total_chunks)

        if is_complete:
            return self._finalize_session()

        diar_info = None
        if self.enable_diarization:
            diar_info = compute_diarization_summary(self.emitted_segments, self.speaker_segments)

        return AudioStreamStepEvent(
            chunk_index=chunk.chunk_index,
            total_chunks=self.total_chunks,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            chunk_text=chunk_text,
            new_segments=new_segments,
            all_segments=self.emitted_segments,
            ranked_phrases=self.current_rankings.get("ranked_phrases", []),
            phrase_statistics=self.phrase_extractor.get_incremental_stats(),
            progress_percent=round(progress_pct, 1),
            is_complete=False,
            processing_time_ms=round(step_elapsed_s * 1000.0, 2),
            diarization_info=diar_info,
        )

    def _finalize_session(self) -> AudioStreamStepEvent:
        """Flush remaining buffered tokens and finalize session state."""
        self.status = "COMPLETED"
        new_segments = []

        # Flush any trailing buffered tokens in segmenter
        trailing_segs = self.segmenter.flush()
        last_chunk = self.chunks[-1] if self.chunks else None
        last_end_time = last_chunk.end_time if last_chunk else 0.0

        for seg in trailing_segs:
            seg_dict = dict(seg)
            seg_dict["start_time"] = round(last_chunk.start_time, 2) if last_chunk else 0.0
            seg_dict["end_time"] = last_end_time
            seg_dict["speaker"] = self._current_speaker
            self.emitted_segments.append(seg_dict)
            new_segments.append(seg_dict)
            self.phrase_extractor.add_segment(seg_dict["text"])

        # Compute final ranked phrases
        self.current_rankings = self.phrase_extractor.finalize_ranking(
            min_frequency=self.min_frequency,
            top_k=self.top_k,
        )

        diar_info = None
        if self.enable_diarization:
            diar_info = compute_diarization_summary(self.emitted_segments, self.speaker_segments)

        return AudioStreamStepEvent(
            chunk_index=self.total_chunks,
            total_chunks=self.total_chunks,
            start_time=0.0,
            end_time=last_end_time,
            chunk_text="",
            new_segments=new_segments,
            all_segments=self.emitted_segments,
            ranked_phrases=self.current_rankings.get("ranked_phrases", []),
            phrase_statistics=self.phrase_extractor.get_incremental_stats(),
            progress_percent=100.0,
            is_complete=True,
            processing_time_ms=0.0,
            diarization_info=diar_info,
        )

    @property
    def current_speaker(self) -> str:
        """Current speaker identifier if diarization is enabled, else empty."""
        return self._current_speaker if self.enable_diarization else ""

    @property
    def diarization_segments(self) -> List[SpeakerSegment]:
        """List of detected speaker intervals if diarization is enabled, else empty."""
        return self.speaker_segments if self.enable_diarization else []

    @property
    def asr_model(self) -> LocalASR:
        """Get the underlying ASR model instance."""
        return self.asr

    @asr_model.setter
    def asr_model(self, model: LocalASR) -> None:
        """Set the underlying ASR model instance."""
        self.asr = model

    def get_all_subtitles(self) -> List[Dict[str, Any]]:
        """Return all finalized emitted subtitle segments."""
        return list(self.emitted_segments)

    def get_session_summary(self) -> Dict[str, Any]:
        """Return structured session statistics and diarization summary."""
        diar_summary = None
        if self.enable_diarization:
            diar_summary = compute_diarization_summary(self.emitted_segments, self.speaker_segments)

        return {
            "status": self.status,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.current_chunk_idx,
            "total_words": len(self.reconstructed_tokens),
            "total_subtitles": len(self.emitted_segments),
            "processing_time_seconds": round(self.total_processing_time_s, 2),
            "diarization_time_seconds": round(self.diarization_time_s, 2),
            "real_time_factor": self.real_time_factor,
            "diarization": diar_summary,
        }

    @property
    def reconstructed_transcript(self) -> str:
        """Reconstruct complete finalized transcript text from emitted subtitle segments."""
        if self.emitted_segments:
            return " ".join(s["text"] for s in self.emitted_segments)
        return " ".join(c.text for c in self.asr_chunk_results if c.text)

    @property
    def real_time_factor(self) -> Optional[float]:
        """Real-time factor: processing_time / audio_duration (RTF < 1 is faster than real time)."""
        if self.metadata and self.metadata.duration_seconds > 0:
            total_time = self.total_processing_time_s + self.diarization_time_s
            return round(total_time / self.metadata.duration_seconds, 3)
        return None

    def export_txt(self) -> str:
        """Generate human-readable plain text export report with optional speaker tags."""
        meta_filename = self.metadata.filename if self.metadata else "audio"
        meta_duration = self.metadata.formatted_duration if self.metadata else "00:00"

        lines = [
            "=" * 70,
            "REAL-TIME SUBTITLE SEGMENTER & N-GRAM PHRASE EXTRACTOR",
            "SPEAKER-AWARE SUBTITLE TRANSCRIPT EXPORT",
            "=" * 70,
            f"Audio File:       {meta_filename}",
            f"Audio Duration:   {meta_duration}",
            f"ASR Model:        {self.model_size}",
            f"Diarization:      {'Enabled' if self.enable_diarization else 'Disabled'}",
            f"Total Segments:   {len(self.emitted_segments)}",
            f"Total Words:      {len(self.reconstructed_tokens)}",
            f"Processing Time:  {self.total_processing_time_s:.2f}s (Diarization: {self.diarization_time_s:.2f}s)",
        ]
        if self.real_time_factor is not None:
            lines.append(f"Real-Time Factor: {self.real_time_factor}x ({'Faster' if self.real_time_factor < 1.0 else 'Slower'} than playback)")

        if self.enable_diarization:
            summary = compute_diarization_summary(self.emitted_segments, self.speaker_segments)
            spk_list = summary.get("speaker_list", list(summary.get("speakers", {}).values()) if isinstance(summary.get("speakers"), dict) else summary.get("speakers", []))
            lines.append("-" * 70)
            lines.append(f"SPEAKERS DETECTED: {summary.get('speaker_count', len(spk_list))}")
            for spk in spk_list:
                lines.append(f"  - {spk.get('speaker', 'Speaker')}: {spk.get('duration_seconds', 0.0)}s ({spk.get('percentage', 0.0)}%) | Turns: {spk.get('turn_count', 0)}")

        lines.append("-" * 70)
        lines.append("SUBTITLE TIMELINE:")
        lines.append("-" * 70)

        for seg in self.emitted_segments:
            s_t = seg.get("start_time", 0.0)
            e_t = seg.get("end_time", 0.0)
            sm, ss = int(s_t // 60), s_t % 60
            em, es = int(e_t // 60), e_t % 60
            spk = seg.get("speaker", "Speaker 1")
            speaker_tag = f"{spk}: " if self.enable_diarization else ""
            lines.append(f"[{sm:02d}:{ss:04.1f} -> {em:02d}:{es:04.1f}] {speaker_tag}{seg.get('text', '')}")

        lines.append("-" * 70)
        lines.append("RECONSTRUCTED COMPLETE TRANSCRIPT:")
        lines.append("-" * 70)
        lines.append(self.reconstructed_transcript)
        lines.append("")

        lines.append("-" * 70)
        lines.append("TOP RANKED KEY PHRASES:")
        lines.append("-" * 70)
        for p in self.current_rankings.get("ranked_phrases", []):
            lines.append(
                f"#{p['rank']} [{p['quality_category']}] \"{p['phrase']}\" "
                f"(freq={p['frequency']}, score={p['quality_score']:.2f})"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def export_json(self) -> str:
        """Generate machine-readable JSON export with complete session and speaker data."""
        diar_summary = None
        if self.enable_diarization:
            diar_summary = compute_diarization_summary(self.emitted_segments, self.speaker_segments)

        formatted_subs = []
        for s in self.emitted_segments:
            st = s.get("start_time", s.get("start", 0.0))
            et = s.get("end_time", s.get("end", st + 1.0))
            formatted_subs.append({
                "start": round(st, 3),
                "end": round(et, 3),
                "speaker": s.get("speaker", "Speaker 1"),
                "text": s.get("text", ""),
                "reason": s.get("reason", "normal"),
            })

        payload = {
            "metadata": {
                "filename": self.metadata.filename if self.metadata else "",
                "format": self.metadata.format if self.metadata else "",
                "duration_seconds": self.metadata.duration_seconds if self.metadata else 0.0,
                "file_size_bytes": self.metadata.file_size_bytes if self.metadata else 0,
                "sample_rate": self.metadata.sample_rate if self.metadata else 16000,
                "channels": self.metadata.channels if self.metadata else 1,
            },
            "asr_config": {
                "engine": "faster-whisper",
                "model_size": self.model_size,
                "chunk_duration_seconds": self.chunk_duration,
                "diarization_enabled": self.enable_diarization,
                "num_speakers": self.num_speakers,
            },
            "performance": {
                "total_processing_time_seconds": round(self.total_processing_time_s, 2),
                "diarization_time_seconds": round(self.diarization_time_s, 2),
                "real_time_factor": self.real_time_factor,
                "average_asr_chunk_ms": (
                    round(sum(self.asr_latencies_ms) / len(self.asr_latencies_ms), 2)
                    if self.asr_latencies_ms
                    else 0.0
                ),
            },
            "diarization_enabled": self.enable_diarization,
            "diarization_summary": diar_summary,
            "diarization": diar_summary,
            "transcript": self.reconstructed_transcript,
            "subtitles": formatted_subs,
            "subtitle_segments": self.emitted_segments,
            "phrase_statistics": self.phrase_extractor.get_incremental_stats(),
            "ranked_phrases": self.current_rankings.get("ranked_phrases", []),
        }
        return json.dumps(payload, indent=2)

    def export_csv(self) -> str:
        """Generate CSV string of ranked phrases."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Rank", "Phrase", "NgramSize", "Frequency", "Score", "Quality", "POS"])
        for p in self.current_rankings.get("ranked_phrases", []):
            writer.writerow([
                p.get("rank", 0),
                p.get("phrase", ""),
                p.get("ngram_size", 0),
                p.get("frequency", 0),
                round(p.get("quality_score", 0.0), 3),
                p.get("quality_category", "LOW"),
                p.get("pos_pattern", ""),
            ])
        return buf.getvalue()

    def export_subtitles_csv(self) -> str:
        """Generate CSV string of finalized subtitles with timestamps and speaker attribution."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["start", "end", "speaker", "text"])
        for s in self.emitted_segments:
            st = s.get("start_time", 0.0)
            et = s.get("end_time", 0.0)
            spk = s.get("speaker", "Speaker 1")
            writer.writerow([round(st, 3), round(et, 3), spk, s.get("text", "")])
        return buf.getvalue()

    # Alias for export_csv
    export_phrases_csv = export_csv

