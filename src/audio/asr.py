"""
Local ASR Module for Phase 6.

Integrates local faster-whisper ASR model for offline/simulated speech recognition.
Guarantees:
- Model download & cache strictly isolated to F:\\Real-Time Subtitle Segmenter\\models
- Configurable model size (tiny, base, small, medium; default 'base' or 'small')
- CPU-first execution with INT8 quantization for fast, lightweight processing
- Single model load reused across all audio chunks
- Zero OpenAI, cloud APIs, or external network transcription services
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from faster_whisper import WhisperModel

from src.audio.chunker import AudioChunk


# Define and enforce project-local models directory on F: drive
AUDIO_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(AUDIO_MODULE_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Explicitly set cache environment variables to F: drive directory
os.environ["HF_HUB_CACHE"] = MODELS_DIR
os.environ["HF_HOME"] = MODELS_DIR

DEFAULT_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base")
SUPPORTED_MODELS = ["tiny", "base", "small", "medium"]

# Global cache for loaded model instances: key -> WhisperModel
_MODEL_CACHE: Dict[str, WhisperModel] = {}


@dataclass
class ASRChunkResult:
    """Structured transcription output for a single audio chunk."""
    chunk_index: int
    start_time: float
    end_time: float
    text: str
    raw_text: str
    processing_time_ms: float
    segments: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        """True if non-empty recognized text was generated."""
        return bool(self.text.strip())


def clean_asr_text(raw_text: str) -> str:
    """
    Perform minimal cleaning on raw ASR output before passing into text pipeline.
    
    Preserves raw recognized words without aggressive grammar alterations.
    Normalizes whitespace and strips null/control characters.
    """
    if not raw_text:
        return ""
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw_text)
    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_asr_model(
    model_size: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    """
    Retrieve or initialize a cached faster-whisper model on F: drive.
    
    Args:
        model_size: Size name ('tiny', 'base', 'small', 'medium'). Defaults to ASR_MODEL_SIZE env or 'base'.
        device: Execution device ('cpu' or 'auto').
        compute_type: Quantization type (default 'int8' for fast CPU inference).
        
    Returns:
        Loaded WhisperModel instance.
    """
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    if size not in SUPPORTED_MODELS:
        size = "base"

    cache_key = f"{size}_{device}_{compute_type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Initialize model using download_root explicitly set to project models dir on F:
    model = WhisperModel(
        model_size_or_path=size,
        device=device,
        compute_type=compute_type,
        download_root=MODELS_DIR,
    )
    _MODEL_CACHE[cache_key] = model
    return model


class LocalASR:
    """
    Local ASR transcriber wrapping faster-whisper.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = (model_size or DEFAULT_MODEL_SIZE).lower()
        self.device = device
        self.compute_type = compute_type
        self.models_dir = MODELS_DIR
        self._model: Optional[WhisperModel] = None

    def load_model(self) -> WhisperModel:
        """Ensure the local Whisper model is loaded and ready."""
        if self._model is None:
            self._model = get_asr_model(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe_chunk(
        self,
        chunk: AudioChunk,
        language: str = "en",
        beam_size: int = 5,
    ) -> ASRChunkResult:
        """
        Transcribe an individual audio chunk.
        
        Args:
            chunk: AudioChunk object containing 16kHz float32 audio samples.
            language: Target spoken language code (default 'en').
            beam_size: Beam search width.
            
        Returns:
            ASRChunkResult with recognized text and timing metadata.
        """
        model = self.load_model()
        start_time = time.perf_counter()

        # If chunk audio is empty or silence
        if len(chunk.samples) == 0:
            return ASRChunkResult(
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text="",
                raw_text="",
                processing_time_ms=0.0,
                segments=[],
            )

        # Transcribe directly from 1D float32 numpy samples
        # vad_filter avoids hallucinations during silent intervals
        segments_gen, info = model.transcribe(
            chunk.samples,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segment_texts = []
        raw_segments = []
        for s in segments_gen:
            segment_texts.append(s.text)
            raw_segments.append({
                "start": round(chunk.start_time + s.start, 2),
                "end": round(chunk.start_time + s.end, 2),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 3),
            })

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        raw_full_text = " ".join(segment_texts)
        cleaned_text = clean_asr_text(raw_full_text)

        return ASRChunkResult(
            chunk_index=chunk.chunk_index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=cleaned_text,
            raw_text=raw_full_text,
            processing_time_ms=round(duration_ms, 2),
            segments=raw_segments,
        )
