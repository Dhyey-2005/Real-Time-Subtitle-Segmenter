"""
Phase 6: Audio Processing & Local ASR Integration Package.
"""

from src.audio.loader import AudioMetadata, AudioLoadError, load_audio, validate_audio_extension
from src.audio.chunker import AudioChunk, AudioChunker
from src.audio.asr import LocalASR, ASRChunkResult, get_asr_model, clean_asr_text, MODELS_DIR
from src.audio.pipeline import AudioStreamingSession, AudioStreamStepEvent
from src.audio.overlap import ConservativeOverlapHandler, OverlapResolution
from src.audio.microphone import (
    LiveMicrophoneCapture,
    LiveMicrophoneConfig,
    MicrophoneDeviceInfo,
    MicrophoneError,
    get_microphone_devices,
)
from src.audio.live_session import LiveMicrophoneSession, LiveStepEvent
from src.audio.diarization import (
    SpeakerSegment,
    SpeakerLabelNormalizer,
    BaseDiarizer,
    AcousticEnergyDiarizer,
    MockDiarizer,
    get_diarizer,
    align_asr_with_diarization,
    compute_diarization_summary,
    DIARIZATION_MODELS_DIR,
)

__all__ = [
    "AudioMetadata",
    "AudioLoadError",
    "load_audio",
    "validate_audio_extension",
    "AudioChunk",
    "AudioChunker",
    "LocalASR",
    "ASRChunkResult",
    "get_asr_model",
    "clean_asr_text",
    "MODELS_DIR",
    "AudioStreamingSession",
    "AudioStreamStepEvent",
    "ConservativeOverlapHandler",
    "OverlapResolution",
    "LiveMicrophoneCapture",
    "LiveMicrophoneConfig",
    "MicrophoneDeviceInfo",
    "MicrophoneError",
    "get_microphone_devices",
    "LiveMicrophoneSession",
    "LiveStepEvent",
    "SpeakerSegment",
    "SpeakerLabelNormalizer",
    "BaseDiarizer",
    "AcousticEnergyDiarizer",
    "MockDiarizer",
    "get_diarizer",
    "align_asr_with_diarization",
    "compute_diarization_summary",
    "DIARIZATION_MODELS_DIR",
]

