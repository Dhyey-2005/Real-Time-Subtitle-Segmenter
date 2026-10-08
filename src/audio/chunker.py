"""
Audio Chunker Module for Phase 6.

Divides decoded 16kHz audio sample arrays into chronological sequential chunks
for progressive local ASR processing.
Guarantees full coverage, zero skipped audio, zero duplicate audio,
and accurate final partial chunk handling.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class AudioChunk:
    """Represents a sequential audio slice for progressive ASR processing."""
    chunk_index: int          # 1-indexed (C1, C2, ...)
    start_time: float         # Start offset in seconds (e.g. 0.0)
    end_time: float           # End offset in seconds (e.g. 5.0)
    duration: float           # Actual chunk duration in seconds
    samples: np.ndarray       # 1D float32 audio samples at 16kHz
    is_final: bool            # True if this is the concluding chunk

    @property
    def formatted_interval(self) -> str:
        """Formatted time interval string [MM:SS.S → MM:SS.S]."""
        start_m, start_s = int(self.start_time // 60), self.start_time % 60
        end_m, end_s = int(self.end_time // 60), self.end_time % 60
        return f"[{start_m:02d}:{start_s:04.1f} -> {end_m:02d}:{end_s:04.1f}]"


class AudioChunker:
    """
    Chronological audio chunk generator with configurable duration.
    """

    def __init__(self, chunk_duration_seconds: float = 5.0, sample_rate: int = 16000):
        if chunk_duration_seconds <= 0.0:
            raise ValueError(f"Chunk duration must be positive, got {chunk_duration_seconds}")
        self.chunk_duration = float(chunk_duration_seconds)
        self.sample_rate = int(sample_rate)
        self.samples_per_chunk = int(round(self.chunk_duration * self.sample_rate))

    def split(self, audio_samples: np.ndarray) -> List[AudioChunk]:
        """
        Divide the audio samples into chronological AudioChunk objects.
        
        Args:
            audio_samples: 1D float32 NumPy array at self.sample_rate.
            
        Returns:
            List of AudioChunk objects in strict chronological order.
        """
        if audio_samples is None or len(audio_samples) == 0:
            return []

        total_samples = len(audio_samples)
        total_duration = total_samples / self.sample_rate
        chunks: List[AudioChunk] = []

        # Iterate in steps of samples_per_chunk
        chunk_idx = 1
        for start_idx in range(0, total_samples, self.samples_per_chunk):
            end_idx = min(start_idx + self.samples_per_chunk, total_samples)
            chunk_slice = audio_samples[start_idx:end_idx]

            start_t = start_idx / self.sample_rate
            end_t = end_idx / self.sample_rate
            duration = end_t - start_t
            is_final = (end_idx == total_samples)

            chunks.append(
                AudioChunk(
                    chunk_index=chunk_idx,
                    start_time=round(start_t, 3),
                    end_time=round(end_t, 3),
                    duration=round(duration, 3),
                    samples=chunk_slice,
                    is_final=is_final,
                )
            )
            chunk_idx += 1

        return chunks
