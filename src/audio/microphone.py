"""
Live Microphone Capture Module for Phase 7.

Captures real-time audio from local microphone devices via sounddevice.
Guarantees:
- Mono, 16kHz float32 audio normalization matching faster-whisper input requirements
- Device enumeration and configurable device selection with default fallback
- Non-blocking continuous streaming via background PortAudio callback
- Thread-safe FIFO ring buffering with configurable chunk duration (default: 3.0s)
- Synthetic / buffer injection support for automated offline test suites
- Zero cloud uploads, zero external telemetry, 100% local processing
"""

import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError, Exception):
    sounddevice = None
    sd = None
    SOUNDDEVICE_AVAILABLE = False

from src.audio.chunker import AudioChunk


class MicrophoneError(Exception):
    """Raised when microphone capture cannot be initialized or fails."""
    pass


@dataclass
class MicrophoneDeviceInfo:
    """Metadata describing a host audio input device."""
    index: int
    name: str
    hostapi: str
    channels: int
    default_samplerate: float
    is_default: bool = False

    @property
    def display_name(self) -> str:
        """Formatted display name for UI dropdowns."""
        def_tag = " (Default)" if self.is_default else ""
        return f"[{self.index}] {self.name}{def_tag}"


def get_microphone_devices() -> List[MicrophoneDeviceInfo]:
    """
    Enumerate all available audio input devices on the local system.
    
    Returns:
        List of MicrophoneDeviceInfo objects for devices with input channels.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return []

    devices: List[MicrophoneDeviceInfo] = []
    try:
        all_devs = sd.query_devices()
        default_input = sd.default.device[0] if sd.default.device else None
        hostapis = {i: h["name"] for i, h in enumerate(sd.query_hostapis())}

        for idx, dev in enumerate(all_devs):
            if dev.get("max_input_channels", 0) > 0:
                is_def = (idx == default_input)
                api_name = hostapis.get(dev.get("hostapi", -1), "Unknown")
                devices.append(
                    MicrophoneDeviceInfo(
                        index=idx,
                        name=dev.get("name", f"Device {idx}"),
                        hostapi=api_name,
                        channels=dev.get("max_input_channels", 1),
                        default_samplerate=dev.get("default_samplerate", 16000.0),
                        is_default=is_def,
                    )
                )
    except Exception as e:
        # Fallback empty list on enumeration failure
        return []

    return devices


@dataclass
class LiveMicrophoneConfig:
    """Configuration for live microphone streaming."""
    chunk_duration_seconds: float = 3.0
    sample_rate: int = 16000
    device_index: Optional[int] = None
    channels: int = 1
    dtype: str = "float32"

    def __post_init__(self):
        if self.chunk_duration_seconds < 1.0 or self.chunk_duration_seconds > 10.0:
            raise ValueError(
                f"Chunk duration must be between 1.0s and 10.0s, got {self.chunk_duration_seconds}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")


class LiveMicrophoneCapture:
    """
    Continuous audio capture manager for local microphone streams.
    """

    def __init__(self, config: Optional[LiveMicrophoneConfig] = None):
        self.config = config or LiveMicrophoneConfig()
        self.sample_rate = self.config.sample_rate
        self.chunk_duration = self.config.chunk_duration_seconds
        self.samples_per_chunk = int(round(self.chunk_duration * self.sample_rate))

        # Thread-safe buffer
        self._lock = threading.Lock()
        self._buffer: List[float] = []
        self._stream: Optional[Any] = None
        self._is_recording = False

        # Monotonic timeline tracking
        self._chunk_sequence = 0
        self._elapsed_audio_seconds = 0.0
        self._start_perf_time = 0.0

    @property
    def is_recording(self) -> bool:
        """True if the audio stream is currently capturing."""
        return self._is_recording

    @property
    def buffered_duration_seconds(self) -> float:
        """Duration in seconds of audio currently waiting in the buffer."""
        with self._lock:
            return len(self._buffer) / self.sample_rate

    @property
    def elapsed_time_seconds(self) -> float:
        """Elapsed live duration since capture started."""
        if not self._is_recording and self._start_perf_time == 0.0:
            return 0.0
        return time.perf_counter() - self._start_perf_time if self._start_perf_time > 0 else 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback running in PortAudio thread."""
        if status:
            pass  # Overflow / underflow status if any
        if not self._is_recording:
            return

        # Ensure 1D float32 mono
        if indata.ndim > 1:
            mono = indata[:, 0].copy()
        else:
            mono = indata.copy().flatten()

        with self._lock:
            self._buffer.extend(mono.astype(np.float32).tolist())

    def start(self) -> None:
        """
        Start capturing audio from the configured microphone device.
        
        Raises:
            MicrophoneError: If capture cannot be initialized.
        """
        if self._is_recording:
            return

        if not SOUNDDEVICE_AVAILABLE:
            raise MicrophoneError("sounddevice package is not installed or available.")

        try:
            device = self.config.device_index
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            self._start_perf_time = time.perf_counter()
        except Exception as e:
            self._is_recording = False
            raise MicrophoneError(f"Failed to start microphone stream: {e}") from e

    def stop(self) -> None:
        """Stop capturing audio and close the stream."""
        self._is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def reset(self) -> None:
        """Stop capture and clear all internal buffers and sequence counters."""
        self.stop()
        with self._lock:
            self._buffer.clear()
        self._chunk_sequence = 0
        self._elapsed_audio_seconds = 0.0
        self._start_perf_time = 0.0

    def feed_samples(self, samples: np.ndarray) -> None:
        """
        Inject synthetic or recorded audio samples directly into the buffer.
        Useful for automated unit testing without physical microphone input.
        """
        if samples is None or len(samples) == 0:
            return
        flat_samples = samples.flatten().astype(np.float32)
        with self._lock:
            self._buffer.extend(flat_samples.tolist())

    def get_chunk(self, duration_seconds: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Retrieve the next sequential AudioChunk if enough audio is buffered.
        
        Args:
            duration_seconds: Custom chunk duration in seconds. Defaults to config duration.
            
        Returns:
            AudioChunk if available, else None.
        """
        target_dur = duration_seconds if duration_seconds is not None else self.chunk_duration
        target_samples = int(round(target_dur * self.sample_rate))

        with self._lock:
            if len(self._buffer) < target_samples:
                return None

            chunk_slice = np.array(self._buffer[:target_samples], dtype=np.float32)
            del self._buffer[:target_samples]

        self._chunk_sequence += 1
        start_t = self._elapsed_audio_seconds
        end_t = start_t + target_dur
        self._elapsed_audio_seconds = end_t

        return AudioChunk(
            chunk_index=self._chunk_sequence,
            start_time=round(start_t, 3),
            end_time=round(end_t, 3),
            duration=round(target_dur, 3),
            samples=chunk_slice,
            is_final=False,
        )

    def get_remaining_chunk(self, min_duration_seconds: float = 0.4) -> Optional[AudioChunk]:
        """
        Flush any remaining partial audio from the buffer upon stop.
        
        Args:
            min_duration_seconds: Minimum duration threshold required to process partial chunk.
            
        Returns:
            Final partial AudioChunk if buffer exceeds threshold, else None.
        """
        with self._lock:
            if not self._buffer:
                return None
            remaining_samples = np.array(self._buffer, dtype=np.float32)
            self._buffer.clear()

        dur = len(remaining_samples) / self.sample_rate
        if dur < min_duration_seconds:
            # Below meaningful threshold, discard
            return None

        self._chunk_sequence += 1
        start_t = self._elapsed_audio_seconds
        end_t = start_t + dur
        self._elapsed_audio_seconds = end_t

        return AudioChunk(
            chunk_index=self._chunk_sequence,
            start_time=round(start_t, 3),
            end_time=round(end_t, 3),
            duration=round(dur, 3),
            samples=remaining_samples,
            is_final=True,
        )
