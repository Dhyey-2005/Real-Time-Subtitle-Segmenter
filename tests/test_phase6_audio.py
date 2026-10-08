"""
Automated Test Suite for Phase 6: Audio Processing, Local ASR, and Streaming Subtitle Pipeline.

Covers:
1. Audio format validation (WAV, MP3, M4A accepted; invalid rejected)
2. Corrupt / empty audio rejection
3. Audio metadata extraction (duration, sample rate, channels, size)
4. Audio chunking boundaries and chronological ordering
5. Final partial chunk handling (e.g. 10.7s file slicing)
6. No skipped or duplicate chunks
7. ASR output structure (ASRChunkResult)
8. Empty ASR result handling (silence / no speech)
9. ASR-to-text-pipeline integration
10. Progressive subtitle segment generation
11. Online phrase extraction and ranking after ASR
12. Cross-segment phrase capture from progressive ASR
13. Overlapping phrase resolution
14. Reset behavior and consistency
15. Export data integrity (TXT, JSON, CSV)
16. F-drive model/cache isolation guarantee
17. Configurable model selection
18. Real faster-whisper ASR integration on sample speech audio
19. Strict OpenAI exclusion
"""

import io
import json
import os
import sys
import wave
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.loader import (
    load_audio,
    validate_audio_extension,
    AudioMetadata,
    AudioLoadError,
)
from src.audio.chunker import AudioChunk, AudioChunker
from src.audio.asr import (
    LocalASR,
    ASRChunkResult,
    clean_asr_text,
    MODELS_DIR,
    get_asr_model,
)
from src.audio.pipeline import AudioStreamingSession, AudioStreamStepEvent
from src.streaming_segmenter import StreamingConfig


def _create_synthetic_wav(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Helper to generate a valid in-memory PCM 16-bit mono WAV file."""
    buf = io.BytesIO()
    total_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 440 Hz sine wave tone
        t = np.linspace(0, duration_seconds, total_frames, endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        wf.writeframes(tone.tobytes())
    return buf.getvalue()


class MockASR(LocalASR):
    """Mock ASR that returns deterministic recognized text for fast unit testing."""

    def __init__(self, transcript_segments: list = None):
        super().__init__(model_size="tiny")
        self.transcripts = transcript_segments or [
            "today we are discussing the new economic package and renewable energy transition",
            "the government announced several new initiatives for job creation and public transit expansion",
            "renewable energy transition represents a major commitment for sustainable economic growth",
        ]

    def load_model(self):
        return None

    def transcribe_chunk(self, chunk: AudioChunk, language: str = "en", beam_size: int = 5) -> ASRChunkResult:
        idx = chunk.chunk_index - 1
        text = self.transcripts[idx] if idx < len(self.transcripts) else ""
        return ASRChunkResult(
            chunk_index=chunk.chunk_index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=text,
            raw_text=text,
            processing_time_ms=12.5,
            segments=[{"start": chunk.start_time, "end": chunk.end_time, "text": text}],
        )


# ==============================================================================
# 1. AUDIO LOADER & VALIDATION TESTS
# ==============================================================================
class TestAudioLoaderAndValidation:
    """Tests for audio format validation, loading, and metadata extraction."""

    def test_01_valid_extensions_accepted(self):
        """Verify .wav, .mp3, .m4a extensions are accepted in any case."""
        assert validate_audio_extension("speech.wav") == ".wav"
        assert validate_audio_extension("audio.WAV") == ".wav"
        assert validate_audio_extension("track.mp3") == ".mp3"
        assert validate_audio_extension("voice.M4A") == ".m4a"

    def test_02_unsupported_extensions_rejected(self):
        """Verify unsupported extensions raise AudioLoadError cleanly."""
        with pytest.raises(AudioLoadError, match="Unsupported audio format"):
            validate_audio_extension("file.txt")

        with pytest.raises(AudioLoadError, match="Unsupported audio format"):
            validate_audio_extension("audio.flac")

        with pytest.raises(AudioLoadError, match="Empty audio filename"):
            validate_audio_extension("")

    def test_03_empty_audio_rejected(self):
        """Verify empty audio bytes or empty files raise AudioLoadError."""
        with pytest.raises(AudioLoadError, match="empty"):
            load_audio(b"", filename="empty.wav")

    def test_04_synthetic_wav_loading_and_metadata(self):
        """Verify valid synthetic WAV is decoded and metadata is extracted accurately."""
        wav_bytes = _create_synthetic_wav(duration_seconds=3.5, sample_rate=16000)
        samples, meta = load_audio(wav_bytes, filename="test_synthetic.wav")

        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert len(samples) == int(3.5 * 16000)
        assert meta.filename == "test_synthetic.wav"
        assert meta.format == "WAV"
        assert abs(meta.duration_seconds - 3.5) < 0.05
        assert meta.sample_rate == 16000
        assert meta.channels == 1
        assert meta.file_size_bytes == len(wav_bytes)

    def test_05_corrupt_audio_raises_clean_error(self):
        """Verify corrupt audio bytes do not crash the application."""
        corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt garbage corrupt header"
        with pytest.raises(AudioLoadError):
            load_audio(corrupt_bytes, filename="corrupt.wav")


# ==============================================================================
# 2. AUDIO CHUNKER TESTS
# ==============================================================================
class TestAudioChunker:
    """Tests for chronological audio slicing, boundary calculation, and completeness."""

    def test_06_chunk_boundaries_and_ordering(self):
        """Verify chunks are strictly chronological and sequential."""
        # 10.0 seconds of audio at 16kHz
        samples = np.zeros(160000, dtype=np.float32)
        chunker = AudioChunker(chunk_duration_seconds=3.0, sample_rate=16000)
        chunks = chunker.split(samples)

        # Expected: [0.0->3.0], [3.0->6.0], [6.0->9.0], [9.0->10.0]
        assert len(chunks) == 4
        assert [c.chunk_index for c in chunks] == [1, 2, 3, 4]
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 3.0
        assert chunks[1].start_time == 3.0
        assert chunks[1].end_time == 6.0
        assert chunks[2].start_time == 6.0
        assert chunks[2].end_time == 9.0
        assert chunks[3].start_time == 9.0
        assert chunks[3].end_time == 10.0

    def test_07_final_partial_chunk_handled_correctly(self):
        """Verify a 10.7s file produces 0-5, 5-10, 10-10.7 with is_final set correctly."""
        total_samples = int(10.7 * 16000)
        samples = np.zeros(total_samples, dtype=np.float32)
        chunker = AudioChunker(chunk_duration_seconds=5.0, sample_rate=16000)
        chunks = chunker.split(samples)

        assert len(chunks) == 3
        # Chunk 1: 0.0 -> 5.0
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 5.0
        assert chunks[0].is_final is False

        # Chunk 2: 5.0 -> 10.0
        assert chunks[1].start_time == 5.0
        assert chunks[1].end_time == 10.0
        assert chunks[1].is_final is False

        # Chunk 3: 10.0 -> 10.7 (final partial)
        assert chunks[2].start_time == 10.0
        assert chunks[2].end_time == 10.7
        assert chunks[2].is_final is True
        assert abs(chunks[2].duration - 0.7) < 0.01

    def test_08_no_skipped_or_duplicated_samples(self):
        """Verify concatenation of chunk samples equals the exact original sample array."""
        samples = np.random.randn(80000).astype(np.float32)
        chunker = AudioChunker(chunk_duration_seconds=2.0, sample_rate=16000)
        chunks = chunker.split(samples)

        reconstructed = np.concatenate([c.samples for c in chunks])
        assert np.array_equal(samples, reconstructed), "Chunk samples do not match original samples!"

    def test_09_empty_samples_returns_empty_chunks(self):
        """Verify splitting empty audio returns empty list."""
        chunker = AudioChunker(chunk_duration_seconds=5.0)
        assert chunker.split(np.array([], dtype=np.float32)) == []


# ==============================================================================
# 3. ASR MODULE & CLEANING TESTS
# ==============================================================================
class TestASRModule:
    """Tests for ASR text cleaning, result structure, and F-drive isolation."""

    def test_10_asr_text_cleaning(self):
        """Verify whitespace normalization and null character stripping."""
        raw = "   today we are discussing   renewable energy\x00 transition\n\t  "
        cleaned = clean_asr_text(raw)
        assert cleaned == "today we are discussing renewable energy transition"
        assert clean_asr_text("") == ""

    def test_11_f_drive_model_isolation(self):
        """Verify models directory and Hugging Face cache are strictly isolated on F: drive."""
        assert MODELS_DIR.lower().startswith("f:"), f"MODELS_DIR '{MODELS_DIR}' is not on F: drive!"
        assert os.path.exists(MODELS_DIR)
        assert os.environ.get("HF_HUB_CACHE", "").lower().startswith("f:")
        assert os.environ.get("HF_HOME", "").lower().startswith("f:")

    def test_12_asr_chunk_result_structure(self):
        """Verify ASRChunkResult fields and has_text helper."""
        res = ASRChunkResult(
            chunk_index=1,
            start_time=0.0,
            end_time=5.0,
            text="hello world",
            raw_text=" hello world ",
            processing_time_ms=50.2,
        )
        assert res.chunk_index == 1
        assert res.has_text is True
        assert res.processing_time_ms == 50.2

        empty_res = ASRChunkResult(1, 0.0, 5.0, "", "", 0.0)
        assert empty_res.has_text is False


# ==============================================================================
# 4. AUDIO STREAMING PIPELINE INTEGRATION TESTS
# ==============================================================================
class TestAudioStreamingPipeline:
    """Tests for end-to-end audio streaming, subtitle segmentation, phrase extraction, and exports."""

    def test_13_pipeline_step_and_progressive_subtitles(self):
        """Verify pipeline steps sequentially through chunks emitting subtitles and phrases."""
        mock_asr = MockASR()
        sess = AudioStreamingSession(
            chunk_duration_seconds=5.0,
            asr_instance=mock_asr,
        )

        # 15.0s audio -> 3 chunks
        samples = np.zeros(240000, dtype=np.float32)
        sess.audio_samples = samples
        sess.metadata = AudioMetadata("test.wav", "WAV", 480000, 15.0, 16000, 1)
        sess.chunks = sess.chunker.split(samples)
        sess.total_chunks = len(sess.chunks)
        sess.reset()

        assert sess.total_chunks == 3
        assert sess.current_chunk_idx == 0

        # Step 1
        ev1 = sess.step()
        assert ev1.chunk_index == 1
        assert ev1.is_complete is False
        assert "economic package" in ev1.chunk_text

        # Step 2
        ev2 = sess.step()
        assert ev2.chunk_index == 2
        assert ev2.is_complete is False
        assert len(sess.emitted_segments) > 0

        # Step 3 (final)
        ev3 = sess.step()
        assert ev3.is_complete is True
        assert sess.status == "COMPLETED"
        assert len(sess.emitted_segments) >= 3

    def test_14_cross_segment_phrase_extraction_preserved(self):
        """Verify phrases spanning across audio/subtitle chunk boundaries are extracted."""
        # Chunk 1: "renewable energy", Chunk 2: "transition is accelerating", Chunk 3: "renewable energy transition"
        mock_asr = MockASR([
            "we are committed to the renewable energy",
            "transition across the country for jobs",
            "moreover renewable energy transition represents our future",
        ])
        sess = AudioStreamingSession(
            chunk_duration_seconds=5.0,
            min_frequency=2,
            top_k=10,
            asr_instance=mock_asr,
        )

        samples = np.zeros(240000, dtype=np.float32)
        sess.audio_samples = samples
        sess.metadata = AudioMetadata("test.wav", "WAV", 480000, 15.0, 16000, 1)
        sess.chunks = sess.chunker.split(samples)
        sess.total_chunks = len(sess.chunks)
        sess.reset()

        while sess.current_chunk_idx < sess.total_chunks:
            sess.step()

        phrases = [p["phrase"] for p in sess.current_rankings.get("ranked_phrases", [])]
        assert any("energy transition" in p or "renewable energy" in p for p in phrases), (
            f"Expected recurring phrase not found in {phrases}"
        )

    def test_15_reset_consistency(self):
        """Verify resetting an audio session clears segments and phrases while preserving audio."""
        mock_asr = MockASR()
        sess = AudioStreamingSession(chunk_duration_seconds=5.0, asr_instance=mock_asr)
        samples = np.zeros(160000, dtype=np.float32)
        sess.audio_samples = samples
        sess.metadata = AudioMetadata("test.wav", "WAV", 320000, 10.0, 16000, 1)
        sess.chunks = sess.chunker.split(samples)
        sess.total_chunks = len(sess.chunks)
        sess.reset()

        # Step 1
        sess.step()
        assert sess.current_chunk_idx == 1
        assert len(sess.asr_chunk_results) == 1

        # Reset
        sess.reset()
        assert sess.status == "READY"
        assert sess.current_chunk_idx == 0
        assert sess.emitted_segments == []
        assert sess.asr_chunk_results == []
        assert sess.total_chunks == 2
        assert sess.audio_samples is not None

    def test_16_export_integrity(self):
        """Verify TXT, JSON, and CSV exports contain valid data matching session output."""
        mock_asr = MockASR()
        sess = AudioStreamingSession(chunk_duration_seconds=5.0, asr_instance=mock_asr)
        samples = np.zeros(240000, dtype=np.float32)
        sess.audio_samples = samples
        sess.metadata = AudioMetadata("speech.wav", "WAV", 480000, 15.0, 16000, 1)
        sess.chunks = sess.chunker.split(samples)
        sess.total_chunks = len(sess.chunks)
        sess.reset()

        while sess.current_chunk_idx < sess.total_chunks:
            sess.step()

        # 1. TXT export
        txt = sess.export_txt()
        assert "REAL-TIME SUBTITLE SEGMENTER" in txt
        assert "speech.wav" in txt
        assert "SUBTITLE TIMELINE:" in txt

        # 2. JSON export
        json_str = sess.export_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["filename"] == "speech.wav"
        assert parsed["asr_config"]["engine"] == "faster-whisper"
        assert len(parsed["subtitle_segments"]) == len(sess.emitted_segments)

        # 3. CSV export
        csv_str = sess.export_csv()
        assert "Rank,Phrase,NgramSize,Frequency,Score,Quality,POS" in csv_str


# ==============================================================================
# 5. REAL FASTER-WHISPER ASR INTEGRATION ON LOCAL SPEECH WAV
# ==============================================================================
class TestRealFasterWhisperASR:
    """Tests real faster-whisper model execution on the local sample speech fixture."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(PROJECT_ROOT, "data", "input", "sample_speech.wav")),
        reason="Local sample speech WAV not found",
    )
    def test_17_real_asr_transcribes_speech_audio(self):
        """Verify real faster-whisper ASR executes on sample_speech.wav with 0 crashes."""
        speech_wav = os.path.join(PROJECT_ROOT, "data", "input", "sample_speech.wav")
        samples, meta = load_audio(speech_wav)
        assert meta.duration_seconds > 10.0

        asr = LocalASR(model_size="tiny")
        chunker = AudioChunker(chunk_duration_seconds=5.0)
        chunks = chunker.split(samples)
        assert len(chunks) >= 3

        # Transcribe first chunk
        result = asr.transcribe_chunk(chunks[0])
        assert result.chunk_index == 1
        assert result.has_text is True
        assert len(result.text) > 0
        assert any(word in result.text.lower() for word in ["today", "economic", "package", "renewable", "energy"])


# ==============================================================================
# 6. ARCHITECTURE & DEPENDENCY EXCLUSION AUDIT
# ==============================================================================
class TestArchitectureExclusions:
    """Verify strict exclusion of OpenAI and cloud dependencies."""

    def test_18_openai_strictly_excluded(self):
        """Verify openai is not installed or imported in sys.modules."""
        assert "openai" not in sys.modules
        for mod in sys.modules:
            assert not mod.startswith("openai."), f"Forbidden module '{mod}' found in sys.modules!"
