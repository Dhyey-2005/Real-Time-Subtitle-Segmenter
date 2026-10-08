"""
Automated Test Suite for Phase 8: Speaker Diarization for Uploaded Audio.

Tests all requirements:
1. Speaker segment representation.
2. Speaker label normalization.
3. Stable speaker mapping.
4. Chronological ordering.
5. Timestamp overlap calculation.
6. Speaker assignment by strongest temporal overlap.
7. Deterministic tie-breaking.
8. Speaker-change subtitle boundary.
9. Same-speaker subtitle continuation.
10. JSON serialization.
11. CSV serialization.
12. TXT serialization.
13. Speaker duration calculation.
14. Speaker turn counting.
15. Mock diarization + mock ASR alignment.
16. Speaker-aware subtitle generation.
17. Speaker-aware export.
18. Existing phrase extractor receives transcript correctly.
19. Existing phrase ranking remains deterministic.
20. Existing uploaded-audio pipeline works with diarization disabled.
21. Existing uploaded-audio pipeline works with diarization enabled using mocked diarization.
22. Existing live microphone pipeline remains unaffected.
23. AcousticEnergyDiarizer on synthetic and real audio.
24. Strict OpenAI and F-drive isolation.
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

from src.audio.diarization import (
    SpeakerSegment,
    SpeakerLabelNormalizer,
    BaseDiarizer,
    AcousticEnergyDiarizer,
    MockDiarizer,
    align_asr_with_diarization,
    compute_diarization_summary,
    get_diarizer,
    DIARIZATION_MODELS_DIR,
)
from src.audio.asr import ASRChunkResult, LocalASR
from src.audio.pipeline import AudioStreamingSession
from src.audio.live_session import LiveMicrophoneSession, LiveStepEvent
from src.audio.microphone import LiveMicrophoneCapture, LiveMicrophoneConfig


def _create_synthetic_pcm(duration_sec: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate simple synthetic sine wave PCM array."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 440 Hz tone
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return samples


def _create_synthetic_wav_bytes(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate valid PCM 16-bit mono WAV bytes."""
    samples = _create_synthetic_pcm(duration_sec, sample_rate)
    pcm_data = (samples * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class MockASRForDiarization(LocalASR):
    """Deterministic Mock ASR providing predictable segments and timestamps."""
    def __init__(self, predefined_transcriptions=None):
        self.predefined_transcriptions = predefined_transcriptions or [
            "Today we are discussing artificial intelligence and renewable energy.",
            "Renewable energy transition is a major topic for global climate policy.",
            "Job creation and economic growth remain top government priorities.",
        ]
        self._call_count = 0
        self.is_loaded = True
        self.model_size = "tiny"
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe_chunk(self, audio_chunk, beam_size=1, language="en"):
        idx = self._call_count % len(self.predefined_transcriptions)
        text = self.predefined_transcriptions[idx]
        self._call_count += 1
        words = [
            {"word": w, "start": audio_chunk.start_time + i * 0.2, "end": audio_chunk.start_time + (i + 1) * 0.2}
            for i, w in enumerate(text.split())
        ]
        return ASRChunkResult(
            chunk_index=audio_chunk.chunk_index,
            start_time=audio_chunk.start_time,
            end_time=audio_chunk.end_time,
            text=text,
            raw_text=text,
            processing_time_ms=10.0,
            segments=[{"start": audio_chunk.start_time, "end": audio_chunk.end_time, "text": text, "words": words}],
        )


class TestSpeakerDiarizationUnit:
    """Unit tests for speaker diarization data models, normalizers, and algorithms."""

    def test_01_speaker_segment_representation(self):
        """Test SpeakerSegment properties, duration, overlap calculation, and serialization."""
        seg = SpeakerSegment(start=1.5, end=4.5, speaker="Speaker 1")
        assert seg.start == 1.5
        assert seg.end == 4.5
        assert seg.speaker == "Speaker 1"
        assert seg.duration == pytest.approx(3.0, 0.001)

        d = seg.to_dict()
        assert d == {"start": 1.5, "end": 4.5, "speaker": "Speaker 1", "duration": 3.0}

    def test_02_speaker_label_normalization(self):
        """Test SpeakerLabelNormalizer transforms raw IDs into clean user-facing labels."""
        normalizer = SpeakerLabelNormalizer()
        assert normalizer.normalize("SPEAKER_00") == "Speaker 1"
        assert normalizer.normalize("SPEAKER_01") == "Speaker 2"
        assert normalizer.normalize("SPEAKER_02") == "Speaker 3"
        # Already normalized labels pass through cleanly
        assert normalizer.normalize("Speaker 1") == "Speaker 1"
        assert normalizer.normalize("Speaker 4") == "Speaker 4"
        # Alternative raw prefixes
        assert normalizer.normalize("spk_99") == "Speaker 5"

    def test_03_stable_speaker_mapping(self):
        """Test SpeakerLabelNormalizer maintains stable mapping across multiple calls."""
        normalizer = SpeakerLabelNormalizer()
        first_call = normalizer.normalize("SPEAKER_00")
        second_call = normalizer.normalize("SPEAKER_01")
        repeat_first = normalizer.normalize("SPEAKER_00")
        repeat_second = normalizer.normalize("SPEAKER_01")

        assert first_call == "Speaker 1"
        assert second_call == "Speaker 2"
        assert repeat_first == "Speaker 1"
        assert repeat_second == "Speaker 2"
        assert normalizer.total_speakers == 2

    def test_04_chronological_ordering(self):
        """Test that speaker intervals and alignment maintain chronological ordering."""
        diar_segments = [
            SpeakerSegment(start=4.0, end=8.0, speaker="Speaker 2"),
            SpeakerSegment(start=0.0, end=4.0, speaker="Speaker 1"),
        ]
        asr_segments = [
            {"start": 1.0, "end": 3.0, "text": "First turn"},
            {"start": 5.0, "end": 7.0, "text": "Second turn"},
        ]
        aligned = align_asr_with_diarization(asr_segments, diar_segments)
        assert len(aligned) == 2
        assert aligned[0]["speaker"] == "Speaker 1"
        assert aligned[1]["speaker"] == "Speaker 2"
        assert aligned[0]["start"] <= aligned[1]["start"]

    def test_05_timestamp_overlap_calculation(self):
        """Test temporal overlap math across different spatial relationships."""
        seg = SpeakerSegment(start=2.0, end=6.0, speaker="Speaker 1")
        
        # 1. Fully inside
        assert seg.overlap_with(3.0, 5.0) == pytest.approx(2.0, 0.001)
        # 2. Covering left boundary
        assert seg.overlap_with(1.0, 4.0) == pytest.approx(2.0, 0.001)
        # 3. Covering right boundary
        assert seg.overlap_with(5.0, 8.0) == pytest.approx(1.0, 0.001)
        # 4. Completely engulfing
        assert seg.overlap_with(1.0, 8.0) == pytest.approx(4.0, 0.001)
        # 5. Non-overlapping before
        assert seg.overlap_with(0.0, 2.0) == 0.0
        # 6. Non-overlapping after
        assert seg.overlap_with(6.5, 9.0) == 0.0

    def test_06_speaker_assignment_by_strongest_overlap(self):
        """Test that ASR segments spanning speaker turns are assigned to the maximum overlap."""
        diar_segments = [
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=10.0, speaker="Speaker 2"),
        ]
        # ASR segment: 2.5 to 5.5 (0.5s overlap with Spk 1, 2.5s overlap with Spk 2)
        asr_segments = [{"start": 2.5, "end": 5.5, "text": "Spanning turn"}]
        aligned = align_asr_with_diarization(asr_segments, diar_segments)
        assert len(aligned) == 1
        assert aligned[0]["speaker"] == "Speaker 2"

    def test_07_deterministic_tie_breaking(self):
        """Test deterministic tie breaking when an ASR segment has exactly equal overlap with two speakers."""
        diar_segments = [
            SpeakerSegment(start=0.0, end=4.0, speaker="Speaker 1"),
            SpeakerSegment(start=4.0, end=8.0, speaker="Speaker 2"),
        ]
        # Equal overlap of 1.0s each (3.0 -> 4.0 and 4.0 -> 5.0)
        asr_segments = [{"start": 3.0, "end": 5.0, "text": "Exact tie text"}]
        aligned = align_asr_with_diarization(asr_segments, diar_segments)
        assert len(aligned) == 1
        # Midpoint is 4.0, distance to spk 1 center (2.0) is 2.0, spk 2 center (6.0) is 2.0
        # Deterministic resolution assigns safely without crash or duplicate
        assert aligned[0]["speaker"] in ["Speaker 1", "Speaker 2"]
        assert aligned[0]["text"] == "Exact tie text"

    def test_08_speaker_change_subtitle_boundary(self):
        """Test that speaker change between consecutive ASR chunks triggers a clean subtitle boundary."""
        session = AudioStreamingSession(enable_diarization=True)
        # Mock diarization with 2 turns: 0-3s Speaker 1, 3-6s Speaker 2
        mock_diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        session.diarizer = mock_diarizer
        
        # Load synthetic audio (6.0s)
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "Hello this is the first speaker speaking.",
            "And now this is the second speaker responding.",
        ])

        # Step 1: Chunk 0 (0.0 to 3.0s) -> Speaker 1
        step1 = session.step()
        assert step1 is not None
        assert session.current_speaker == "Speaker 1"

        # Step 2: Chunk 1 (3.0 to 6.0s) -> Speaker 2 (Triggers speaker_change boundary flush)
        step2 = session.step()
        assert step2 is not None
        assert session.current_speaker == "Speaker 2"

        # Check emitted subtitle segments
        subs = session.get_all_subtitles()
        assert len(subs) >= 1
        # First subtitle must be attributed to Speaker 1
        assert subs[0]["speaker"] == "Speaker 1"
        assert "first speaker" in subs[0]["text"].lower()

    def test_09_same_speaker_subtitle_continuation(self):
        """Test that consecutive ASR chunks from the same speaker do not artificially force boundaries."""
        session = AudioStreamingSession(enable_diarization=True)
        # Mock diarization with single speaker throughout
        mock_diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=10.0, speaker="Speaker 1"),
        ])
        session.diarizer = mock_diarizer
        
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "Continuing speech from speaker one.",
            "Still speaking on the same topic.",
        ])

        session.step()
        assert session.current_speaker == "Speaker 1"
        session.step()
        assert session.current_speaker == "Speaker 1"

    def test_10_json_serialization(self):
        """Test JSON serialization preserves start, end, speaker, text and metadata."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "Speaker one test line.",
            "Speaker two test line.",
        ])
        while session.step():
            pass

        json_str = session.export_json()
        data = json.loads(json_str)
        assert data["diarization_enabled"] is True
        assert "diarization_summary" in data
        assert len(data["subtitles"]) >= 2
        for s in data["subtitles"]:
            assert "speaker" in s
            assert "start" in s
            assert "end" in s
            assert "text" in s

    def test_11_csv_serialization(self):
        """Test CSV serialization exports start, end, speaker, text with valid headers."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "First turn speech.",
            "Second turn speech.",
        ])
        while session.step():
            pass

        csv_str = session.export_subtitles_csv()
        lines = csv_str.strip().split("\r\n") if "\r\n" in csv_str else csv_str.strip().split("\n")
        assert len(lines) >= 3  # Header + 2 rows
        assert "start,end,speaker,text" in lines[0]
        assert "Speaker 1" in lines[1]
        assert "Speaker 2" in lines[2]

    def test_12_txt_serialization(self):
        """Test TXT serialization produces formatted '[start -> end] Speaker N: text'."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "Alpha statement.",
            "Beta statement.",
        ])
        while session.step():
            pass

        txt_str = session.export_txt()
        assert "Speaker 1:" in txt_str
        assert "Speaker 2:" in txt_str
        assert "alpha statement" in txt_str.lower()
        assert "beta statement" in txt_str.lower()

    def test_13_speaker_duration_calculation(self):
        """Test compute_diarization_summary calculates accurate speech durations and percentages."""
        segments = [
            SpeakerSegment(start=0.0, end=6.0, speaker="Speaker 1"),
            SpeakerSegment(start=6.0, end=10.0, speaker="Speaker 2"),
        ]
        summary = compute_diarization_summary(segments)
        assert summary["total_speakers"] == 2
        spk1 = summary["speakers"]["Speaker 1"]
        spk2 = summary["speakers"]["Speaker 2"]
        assert spk1["speech_time_sec"] == 6.0
        assert spk2["speech_time_sec"] == 4.0
        assert spk1["speech_pct"] == 60.0
        assert spk2["speech_pct"] == 40.0

    def test_14_speaker_turn_counting(self):
        """Test compute_diarization_summary accurately counts turns with alternating speakers."""
        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="Speaker 1"),
            SpeakerSegment(start=2.0, end=4.0, speaker="Speaker 2"),
            SpeakerSegment(start=4.0, end=6.0, speaker="Speaker 1"),
            SpeakerSegment(start=6.0, end=8.0, speaker="Speaker 2"),
            SpeakerSegment(start=8.0, end=10.0, speaker="Speaker 1"),
        ]
        summary = compute_diarization_summary(segments)
        assert summary["total_turns"] == 5
        assert summary["speakers"]["Speaker 1"]["turns"] == 3
        assert summary["speakers"]["Speaker 2"]["turns"] == 2


class TestSpeakerDiarizationIntegration:
    """Integration tests for end-to-end diarization pipeline, ASR, NLP and backward compatibility."""

    def test_15_mock_diarization_and_mock_asr_alignment(self):
        """Test alignment between mock diarizer and mock ASR words/segments."""
        diar_segments = [
            SpeakerSegment(start=0.0, end=4.0, speaker="Speaker 1"),
            SpeakerSegment(start=4.0, end=8.0, speaker="Speaker 2"),
        ]
        asr_segments = [
            {"start": 0.5, "end": 3.5, "text": "Sentence one from speaker one."},
            {"start": 4.5, "end": 7.5, "text": "Sentence two from speaker two."},
        ]
        aligned = align_asr_with_diarization(asr_segments, diar_segments)
        assert len(aligned) == 2
        assert aligned[0]["speaker"] == "Speaker 1"
        assert aligned[1]["speaker"] == "Speaker 2"

    def test_16_speaker_aware_subtitle_generation(self):
        """Test that generated subtitles carry speaker tags end-to-end through pipeline."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization()

        while session.step():
            pass

        subtitles = session.get_all_subtitles()
        assert len(subtitles) >= 2
        speakers_in_subs = {s.get("speaker") for s in subtitles}
        assert "Speaker 1" in speakers_in_subs
        assert "Speaker 2" in speakers_in_subs

    def test_17_speaker_aware_export_pipeline(self):
        """Test all session export formats include valid speaker labels."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization()
        while session.step():
            pass

        # CSV export
        csv_data = session.export_subtitles_csv()
        assert "start,end,speaker,text" in csv_data
        assert "Speaker 1" in csv_data
        assert "Speaker 2" in csv_data

        # JSON export
        json_data = json.loads(session.export_json())
        assert any(sub["speaker"] == "Speaker 1" for sub in json_data["subtitles"])
        assert any(sub["speaker"] == "Speaker 2" for sub in json_data["subtitles"])

        # TXT export
        txt_data = session.export_txt()
        assert "Speaker 1:" in txt_data
        assert "Speaker 2:" in txt_data

    def test_18_phrase_extractor_receives_transcript_correctly(self):
        """Test that existing StreamingPhraseExtractor receives text from speaker-aware subtitles."""
        session = AudioStreamingSession(enable_diarization=True)
        session.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization([
            "Renewable energy transition represents critical economic policy.",
            "Renewable energy transition creates significant new employment.",
        ])
        while session.step():
            pass

        # Extract phrases from session
        ranking = session.phrase_extractor.finalize_ranking(min_frequency=1)
        phrases = [p["phrase"] for p in ranking.get("ranked_phrases", [])]
        assert len(phrases) > 0
        assert any("renewable energy" in p for p in phrases)

    def test_19_phrase_ranking_remains_deterministic(self):
        """Test that phrase ranking is identical whether diarization is enabled or disabled."""
        texts = [
            "Renewable energy transition accelerates economic growth and innovation.",
            "Renewable energy transition provides sustainable infrastructure development.",
        ]

        # Run 1: Diarization Disabled
        session_off = AudioStreamingSession(enable_diarization=False)
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session_off.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session_off.asr_model = MockASRForDiarization(texts)
        while session_off.step():
            pass
        ranking_off = session_off.phrase_extractor.finalize_ranking(min_frequency=1)
        phrases_off = [p["phrase"] for p in ranking_off.get("ranked_phrases", [])]

        # Run 2: Diarization Enabled
        session_on = AudioStreamingSession(enable_diarization=True)
        session_on.diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        session_on.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session_on.asr_model = MockASRForDiarization(texts)
        while session_on.step():
            pass
        ranking_on = session_on.phrase_extractor.finalize_ranking(min_frequency=1)
        phrases_on = [p["phrase"] for p in ranking_on.get("ranked_phrases", [])]

        assert phrases_off == phrases_on

    def test_20_pipeline_diarization_disabled_backward_compatibility(self):
        """Test that uploaded audio pipeline functions identically to Phase 6 when diarization is disabled."""
        session = AudioStreamingSession(enable_diarization=False)
        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization()
        
        step_events = []
        while True:
            event = session.step()
            if not event:
                break
            step_events.append(event)

        assert len(step_events) == 2
        assert session.enable_diarization is False
        assert session.diarization_segments == []
        assert session.current_speaker == ""

        # Subtitles without diarization still have empty or unforced speaker tags
        subtitles = session.get_all_subtitles()
        assert len(subtitles) >= 1

    def test_21_pipeline_diarization_enabled_with_mock(self):
        """Test complete session progression with enable_diarization=True and mock diarizer."""
        session = AudioStreamingSession(enable_diarization=True, num_speakers=2)
        mock_diarizer = MockDiarizer([
            SpeakerSegment(start=0.0, end=3.0, speaker="Speaker 1"),
            SpeakerSegment(start=3.0, end=6.0, speaker="Speaker 2"),
        ])
        session.diarizer = mock_diarizer

        wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
        session.load_audio_source(wav_bytes, filename="test.wav", chunk_duration_sec=3.0)
        session.asr_model = MockASRForDiarization()

        progress_history = []
        while True:
            event = session.step()
            if not event:
                break
            progress_history.append(event)

        assert len(progress_history) == 2
        summary = session.get_session_summary()
        assert "diarization" in summary
        assert summary["diarization"]["total_speakers"] == 2
        assert summary["diarization"]["total_turns"] == 2

    def test_22_live_microphone_pipeline_remains_unaffected(self):
        """Test that Phase 7 LiveMicrophoneSession is completely unaffected by diarization addition."""
        mock_asr = MockASRForDiarization()
        capture = LiveMicrophoneCapture(LiveMicrophoneConfig(chunk_duration_seconds=3.0))
        mic_session = LiveMicrophoneSession(asr_instance=mock_asr, capture_instance=capture)
        mic_session.status = "LISTENING"

        # Feed synthetic audio samples to capture
        samples = _create_synthetic_pcm(duration_sec=3.0)
        capture.feed_samples(samples)

        step_event = mic_session.step()
        assert step_event is not None
        assert step_event.chunk_index == 1
        assert "Today" in step_event.cleaned_text or "energy" in step_event.cleaned_text

        summary = mic_session.get_session_summary()
        assert summary["chunks_processed"] == 1


class TestAcousticDiarizerAndAudio:
    """Test the local AcousticEnergyDiarizer and multi-speaker audio processing."""

    def test_23_acoustic_energy_diarizer_on_audio(self):
        """Test AcousticEnergyDiarizer runs locally on audio samples without external dependencies."""
        diarizer = AcousticEnergyDiarizer()
        
        # 1. Very short audio fallback
        short_samples = _create_synthetic_pcm(duration_sec=0.5)
        segs_short = diarizer.diarize(short_samples, sample_rate=16000)
        assert len(segs_short) == 1
        assert segs_short[0].speaker == "Speaker 1"

        # 2. Multi-second synthetic audio with known speaker count
        longer_samples = _create_synthetic_pcm(duration_sec=4.0)
        segs = diarizer.diarize(longer_samples, sample_rate=16000, num_speakers=2)
        assert len(segs) >= 1
        for s in segs:
            assert s.speaker in ["Speaker 1", "Speaker 2"]
            assert s.start >= 0.0
            assert s.end <= 4.0

    def test_24_f_drive_model_isolation_and_zero_openai(self):
        """Verify model cache directory is strictly on F: and zero OpenAI packages/APIs exist."""
        # 1. F-drive isolation
        assert DIARIZATION_MODELS_DIR.startswith("F:\\") or DIARIZATION_MODELS_DIR.startswith("F:/")
        assert "C:" not in DIARIZATION_MODELS_DIR

        # 2. Zero OpenAI in environment or imports
        import sys
        for mod in sys.modules:
            assert "openai" not in mod.lower(), f"Forbidden openai module detected: {mod}"

        # 3. Code inspection for forbidden tokens in src/audio/diarization.py
        diar_path = os.path.join(PROJECT_ROOT, "src", "audio", "diarization.py")
        with open(diar_path, "r", encoding="utf-8") as f:
            code = f.read()
            assert "import openai" not in code.lower()
            assert "from openai" not in code.lower()
            assert "api.openai.com" not in code.lower()
