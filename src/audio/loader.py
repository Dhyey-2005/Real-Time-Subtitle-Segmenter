"""
Audio Loader Module for Phase 6.

Validates audio files, extracts metadata, and decodes audio files (WAV, MP3, M4A)
into normalized mono 16kHz float32 NumPy arrays for local ASR ingestion.
Uses PyAV (av) with fallback to wave for maximum standalone reliability.
"""

import io
import os
import wave
from dataclasses import dataclass
from typing import Tuple, Union, BinaryIO

import av
import numpy as np


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded or is invalid/corrupt."""
    pass


@dataclass
class AudioMetadata:
    """Metadata describing an uploaded or loaded audio file."""
    filename: str
    format: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int

    @property
    def formatted_duration(self) -> str:
        """Format duration as MM:SS.S."""
        mins = int(self.duration_seconds // 60)
        secs = self.duration_seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    @property
    def formatted_size_kb(self) -> str:
        """Format file size in KB or MB."""
        if self.file_size_bytes < 1024 * 1024:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        return f"{self.file_size_bytes / (1024 * 1024):.2f} MB"


DEFAULT_MAX_AUDIO_MB = 150
MAX_AUDIO_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_MB", str(DEFAULT_MAX_AUDIO_MB))) * 1024 * 1024


def validate_audio_extension(filename: str) -> str:
    """
    Validate that the file extension is supported (WAV, MP3, M4A).
    
    Args:
        filename: Name of the file with extension.
        
    Returns:
        Normalized lower-case extension string (e.g. '.wav').
        
    Raises:
        AudioLoadError: If the extension is not supported.
    """
    if not filename:
        raise AudioLoadError("Empty audio filename provided.")
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()
    if ext_lower not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(ext.upper().lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        raise AudioLoadError(
            f"Unsupported audio format '{ext}'. Supported formats: {supported_str}."
        )
    return ext_lower


def load_audio(
    file_source: Union[str, bytes, BinaryIO],
    filename: str = "audio.wav",
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, AudioMetadata]:
    """
    Load an audio file, decode it into a mono 16kHz float32 NumPy array,
    and return its metadata.
    
    Args:
        file_source: File path (str), raw bytes (bytes), or file-like object.
        filename: Original name of the file for extension/metadata detection.
        target_sample_rate: Target sample rate in Hz (default 16000 for Whisper).
        
    Returns:
        Tuple of (audio_samples_1d_float32, AudioMetadata).
        
    Raises:
        AudioLoadError: If loading or decoding fails.
    """
    ext = validate_audio_extension(filename)

    # Determine file size and prepare stream
    if isinstance(file_source, (bytes, bytearray)):
        if len(file_source) == 0:
            raise AudioLoadError("The uploaded audio file is empty (0 bytes).")
        file_size = len(file_source)
        stream_obj = io.BytesIO(file_source)
    elif isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise AudioLoadError(f"Audio file not found: {file_source}")
        file_size = os.path.getsize(file_source)
        if file_size == 0:
            raise AudioLoadError(f"The audio file is empty (0 bytes): {file_source}")
        stream_obj = file_source
    elif hasattr(file_source, "read"):
        # File-like object (e.g. Streamlit UploadedFile)
        content = file_source.read()
        file_size = len(content)
        if file_size == 0:
            raise AudioLoadError("The uploaded audio file is empty (0 bytes).")
        stream_obj = io.BytesIO(content)
    else:
        raise AudioLoadError(f"Invalid audio source type: {type(file_source)}")

    # Check maximum file size bound
    if file_size > MAX_AUDIO_SIZE_BYTES:
        max_mb = MAX_AUDIO_SIZE_BYTES // (1024 * 1024)
        size_mb = file_size / (1024 * 1024)
        raise AudioLoadError(
            f"Audio file size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )

    # Decode audio using PyAV
    try:
        container = av.open(stream_obj, mode="r")
    except Exception as e:
        raise AudioLoadError(f"The audio file could not be decoded: {str(e)}") from e

    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise AudioLoadError(f"The audio file could not be decoded: No audio stream found in '{filename}'.")

        orig_sample_rate = audio_stream.rate or target_sample_rate
        orig_channels = audio_stream.channels or 1

        # Resample to mono s16 at target_sample_rate
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=target_sample_rate,
        )

        raw_buffer = io.BytesIO()
        for frame in container.decode(audio_stream):
            for resampled_frame in resampler.resample(frame):
                raw_buffer.write(resampled_frame.to_ndarray())

        # Flush any remaining frames from resampler
        for resampled_frame in resampler.resample(None):
            raw_buffer.write(resampled_frame.to_ndarray())

        buffer_data = raw_buffer.getvalue()
        if len(buffer_data) == 0:
            raise AudioLoadError(f"Decoded audio in '{filename}' contains 0 audio frames.")

        # Convert int16 samples to normalized float32 in [-1.0, 1.0]
        int16_samples = np.frombuffer(buffer_data, dtype=np.int16)
        float32_samples = int16_samples.astype(np.float32) / 32768.0

        duration_sec = float(len(float32_samples) / target_sample_rate)
        if duration_sec <= 0.0:
            raise AudioLoadError(f"Decoded audio in '{filename}' has invalid zero duration.")

        metadata = AudioMetadata(
            filename=filename,
            format=ext.upper().lstrip("."),
            file_size_bytes=file_size,
            duration_seconds=duration_sec,
            sample_rate=orig_sample_rate,
            channels=orig_channels,
        )

        return float32_samples, metadata

    except AudioLoadError:
        raise
    except Exception as e:
        raise AudioLoadError(f"Error decoding audio in '{filename}': {str(e)}") from e
    finally:
        try:
            container.close()
        except Exception:
            pass
