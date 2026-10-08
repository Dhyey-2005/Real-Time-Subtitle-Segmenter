"""
Phase 9 Stability, UX Polish & Regression Test Suite.

Verifies the 18 Phase 9 stability and reliability requirements:
1. Invalid audio extension.
2. Empty upload.
3. Corrupted/undecodable audio handling.
4. Empty transcript handling.
5. Empty phrase result handling.
6. New upload resets previous session state.
7. Diarization OFF preserves previous behavior.
8. Diarization ON preserves speaker metadata.
9. Speaker summary remains correct.
10. TXT export remains valid.
11. JSON export remains valid.
12. CSV export remains valid.
13. Subtitle CSV remains valid.
14. Export controls only appear when appropriate.
15. Processing failure can recover to another upload.
16. No stale speaker labels between sessions.
17. No stale phrase results between sessions.
18. Existing deterministic phrase ranking remains unchanged.
"""

import io
import json
import os
import sys
import wave
import csv
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.loader import (
    load_audio,
    validate_audio_extension,
    AudioLoadError,
    AudioMetadata,
    SUPPORTED_EXTENSIONS,
)
from src.audio.pipeline import AudioStreamingSession
from src.audio.asr import LocalASR, ASRChunkResult
from src.audio.diarization import (
    SpeakerSegment,
    MockDiarizer,
    compute_diarization_summary,
)
from src.phrase_ranker import analyze_and_rank_phrases


def _create_synthetic_pcm(duration_sec: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate simple synthetic sine wave PCM array."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
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


class MockASR(LocalASR):
    """Deterministic Mock ASR providing predictable speech or silence."""
    def __init__(self, transcriptions=None):
        self.transcriptions = transcriptions or [
            "Today we are discussing artificial intelligence and renewable energy.",
            "Renewable energy transition is a major topic for global climate policy.",
            "Artificial intelligence infrastructure requires significant renewable energy investments.",
        ]
        self._call_count = 0
        self.is_loaded = True
        self.model_size = "tiny"
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe_chunk(self, audio_chunk, beam_size=1, language="en"):
        idx = self._call_count % len(self.transcriptions)
        text = self.transcriptions[idx]
        self._call_count += 1
        words = [{"word": w, "start": audio_chunk.start_time, "end": audio_chunk.end_time} for w in text.split()]
        return ASRChunkResult(
            chunk_index=audio_chunk.chunk_index,
            start_time=audio_chunk.start_time,
            end_time=audio_chunk.end_time,
            text=text,
            raw_text=text,
            segments=[{
                "id": 0,
                "start": audio_chunk.start_time,
                "end": audio_chunk.end_time,
                "text": text,
                "words": words,
            }],
            processing_time_ms=10.0,
        )


class MockSilentASR(LocalASR):
    """Deterministic Mock ASR that simulates non-speech audio (empty transcription)."""
    def __init__(self):
        self.is_loaded = True
        self.model_size = "tiny"
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe_chunk(self, audio_chunk, beam_size=1, language="en"):
        return ASRChunkResult(
            chunk_index=audio_chunk.chunk_index,
            start_time=audio_chunk.start_time,
            end_time=audio_chunk.end_time,
            text="",
            raw_text="",
            segments=[],
            processing_time_ms=5.0,
        )


# ==============================================================================
# TESTS 1 - 5: INPUT VALIDATION & EMPTY STATES
# ==============================================================================

def test_1_invalid_audio_extension():
    """Requirement 1: Unsupported extension raises clean AudioLoadError."""
    with pytest.raises(AudioLoadError) as exc_info:
        validate_audio_extension("audio.ogg")
    assert "Unsupported audio format" in str(exc_info.value)

    with pytest.raises(AudioLoadError) as exc_info2:
        validate_audio_extension("document.pdf")
    assert "Unsupported audio format" in str(exc_info2.value)


def test_2_empty_upload():
    """Requirement 2: Empty upload (0 bytes) raises clean user-facing error."""
    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(b"", filename="empty.wav")
    assert "empty (0 bytes)" in str(exc_info.value)


def test_3_corrupted_audio_handling():
    """Requirement 3: Corrupted or undecodable audio is handled gracefully."""
    corrupted_data = b"NOT_A_VALID_AUDIO_HEADER_CORRUPTED_STREAM_DATA_BYTES"
    with pytest.raises(AudioLoadError) as exc_info:
        load_audio(corrupted_data, filename="corrupt.wav")
    assert "could not be decoded" in str(exc_info.value) or "Error decoding audio" in str(exc_info.value)


def test_4_empty_transcript_handling():
    """Requirement 4: Audio with no detected speech handles empty transcript cleanly."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_silent_asr = MockSilentASR()

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_silent_asr,
    )
    session.load_audio_source(wav_bytes, filename="silent.wav")
    assert session.total_chunks == 2

    # Step through entire audio
    for _ in range(session.total_chunks):
        session.step()

    assert session.reconstructed_transcript == ""
    assert len(session.emitted_segments) == 0
    assert len(session.current_rankings["ranked_phrases"]) == 0
    assert session.export_txt() != ""  # Produces clean report without crashing


def test_5_empty_phrase_result_handling():
    """Requirement 5: Audio with single words / no recurring phrases produces empty rankings cleanly."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    # Each chunk has disjoint non-repeating words
    unique_words_asr = MockASR(transcriptions=[
        "apple banana cherry",
        "dog elephant fox",
    ])

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        min_frequency=2,
        asr_instance=unique_words_asr,
    )
    session.load_audio_source(wav_bytes, filename="unique.wav")
    while session.current_chunk_idx < session.total_chunks:
        session.step()

    # No phrases meet min_frequency of 2
    assert len(session.current_rankings["ranked_phrases"]) == 0
    csv_out = session.export_csv()
    assert "Rank,Phrase,NgramSize,Frequency,Score,Quality,POS" in csv_out


# ==============================================================================
# TESTS 6 - 9: SESSION RESET & DIARIZATION INTEGRITY
# ==============================================================================

def test_6_new_upload_resets_previous_session_state():
    """Requirement 6: Loading a new file completely resets all previous session state."""
    wav_bytes_1 = _create_synthetic_wav_bytes(duration_sec=6.0)
    wav_bytes_2 = _create_synthetic_wav_bytes(duration_sec=2.0)
    mock_asr = MockASR()

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_asr,
    )
    # Process file 1
    session.load_audio_source(wav_bytes_1, filename="session1.wav")
    assert session.total_chunks == 3
    session.step()
    session.step()
    assert len(session.emitted_segments) > 0
    assert len(session.reconstructed_tokens) > 0

    # Now load file 2
    session.load_audio_source(wav_bytes_2, filename="session2.wav")
    # All stale state must be wiped
    assert session.current_chunk_idx == 0
    assert session.total_chunks == 1
    assert len(session.emitted_segments) == 0
    assert len(session.reconstructed_tokens) == 0
    assert len(session.asr_chunk_results) == 0
    assert len(session.current_rankings["ranked_phrases"]) == 0
    assert session.metadata.filename == "session2.wav"


def test_7_diarization_off_preserves_previous_behavior():
    """Requirement 7: Diarization OFF preserves Phase 6 behavior without invoking diarizer."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_asr,
        enable_diarization=False,
    )
    session.load_audio_source(wav_bytes, filename="audio.wav")
    assert len(session.speaker_segments) == 0
    assert session.diarization_time_s == 0.0

    session.step()
    assert len(session.emitted_segments) > 0
    assert session.emitted_segments[0].get("speaker") == "Speaker 1"


def test_8_diarization_on_preserves_speaker_metadata():
    """Requirement 8: Diarization ON preserves speaker labels in emitted subtitle segments."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=6.0)
    mock_asr = MockASR(transcriptions=[
        "First speaker says hello.",
        "Second speaker responds warmly.",
        "First speaker concludes conversation.",
    ])
    mock_diar = MockDiarizer(predefined_segments=[
        SpeakerSegment(start=0.0, end=2.0, speaker="Speaker 1"),
        SpeakerSegment(start=2.0, end=4.0, speaker="Speaker 2"),
        SpeakerSegment(start=4.0, end=6.0, speaker="Speaker 1"),
    ])

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_asr,
        enable_diarization=True,
        diarizer_instance=mock_diar,
    )
    session.load_audio_source(wav_bytes, filename="multi.wav")
    assert len(session.speaker_segments) == 3

    while session.current_chunk_idx < session.total_chunks:
        session.step()

    assert len(session.emitted_segments) >= 2
    speakers = {s.get("speaker") for s in session.emitted_segments}
    assert "Speaker 1" in speakers
    assert "Speaker 2" in speakers


def test_9_speaker_summary_remains_correct():
    """Requirement 9: Speaker summary calculates turns, duration, and percentages correctly."""
    segments = [
        {"start_time": 0.0, "end_time": 5.0, "speaker": "Speaker 1", "text": "hello"},
        {"start_time": 5.0, "end_time": 10.0, "speaker": "Speaker 2", "text": "world"},
        {"start_time": 10.0, "end_time": 15.0, "speaker": "Speaker 1", "text": "again"},
    ]
    summary = compute_diarization_summary(segments)
    assert summary["speaker_count"] == 2
    spk1 = next(s for s in summary["speaker_list"] if s["speaker"] == "Speaker 1")
    spk2 = next(s for s in summary["speaker_list"] if s["speaker"] == "Speaker 2")
    assert spk1["turn_count"] == 2
    assert spk1["duration_seconds"] == 10.0
    assert spk2["turn_count"] == 1
    assert spk2["duration_seconds"] == 5.0
    assert spk1["percentage"] == 66.7
    assert spk2["percentage"] == 33.3


# ==============================================================================
# TESTS 10 - 14: EXPORT FORMAT & CONTENT VALIDATION
# ==============================================================================

def test_10_txt_export_remains_valid():
    """Requirement 10: TXT export contains timestamps, speaker labels, and transcript text."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_asr,
        enable_diarization=True,
        diarizer_instance=MockDiarizer(),
    )
    session.load_audio_source(wav_bytes, filename="test.wav")
    session.step()

    txt = session.export_txt()
    assert "REAL-TIME SUBTITLE SEGMENTER & N-GRAM PHRASE EXTRACTOR" in txt
    assert "Audio File:       test.wav" in txt
    assert "SUBTITLE TIMELINE:" in txt
    assert "Speaker 1" in txt
    assert "RECONSTRUCTED COMPLETE TRANSCRIPT:" in txt


def test_11_json_export_remains_valid():
    """Requirement 11: JSON export is valid, structured, and contains all session sections."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    session = AudioStreamingSession(chunk_duration_seconds=2.0, asr_instance=mock_asr)
    session.load_audio_source(wav_bytes, filename="test.wav")
    session.step()

    json_str = session.export_json()
    data = json.loads(json_str)
    assert "metadata" in data
    assert data["metadata"]["filename"] == "test.wav"
    assert "asr_config" in data
    assert "performance" in data
    assert "subtitles" in data
    assert "transcript" in data
    assert "ranked_phrases" in data


def test_12_csv_export_remains_valid():
    """Requirement 12: CSV export contains valid columns for phrases."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    session = AudioStreamingSession(chunk_duration_seconds=2.0, min_frequency=1, asr_instance=mock_asr)
    session.load_audio_source(wav_bytes, filename="test.wav")
    session.step()

    csv_data = session.export_csv()
    reader = list(csv.reader(io.StringIO(csv_data)))
    assert len(reader) >= 1
    assert reader[0] == ["Rank", "Phrase", "NgramSize", "Frequency", "Score", "Quality", "POS"]


def test_13_subtitle_csv_remains_valid():
    """Requirement 13: Subtitles CSV contains valid start, end, speaker, and text columns."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    session = AudioStreamingSession(chunk_duration_seconds=2.0, asr_instance=mock_asr)
    session.load_audio_source(wav_bytes, filename="test.wav")
    session.step()

    sub_csv = session.export_subtitles_csv()
    reader = list(csv.reader(io.StringIO(sub_csv)))
    assert len(reader) >= 2
    assert reader[0] == ["start", "end", "speaker", "text"]
    assert len(reader[1]) == 4


def test_14_export_controls_only_appear_when_appropriate():
    """Requirement 14: Emitted segments are empty prior to processing, populated after."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    session = AudioStreamingSession(chunk_duration_seconds=2.0, asr_instance=mock_asr)
    session.load_audio_source(wav_bytes, filename="test.wav")

    # Before stepping, no segments exist
    assert len(session.emitted_segments) == 0

    # After stepping, segments exist
    session.step()
    assert len(session.emitted_segments) > 0


# ==============================================================================
# TESTS 15 - 18: ERROR RECOVERY, STALE DATA & DETERMINISM
# ==============================================================================

def test_15_processing_failure_can_recover_to_another_upload():
    """Requirement 15: Failed upload recovers cleanly when a valid file is loaded next."""
    mock_asr = MockASR()
    session = AudioStreamingSession(chunk_duration_seconds=2.0, asr_instance=mock_asr)

    # 1. Attempt invalid load
    with pytest.raises(AudioLoadError):
        session.load_audio_source(b"CORRUPTED_BYTES", filename="bad.wav")

    # 2. Upload valid file immediately after
    valid_wav = _create_synthetic_wav_bytes(duration_sec=2.0)
    meta = session.load_audio_source(valid_wav, filename="good.wav")
    assert meta.filename == "good.wav"
    assert session.status == "READY"
    assert session.total_chunks == 1

    # Can step without issues
    event = session.step()
    assert event is not None
    assert len(session.emitted_segments) > 0


def test_16_no_stale_speaker_labels_between_sessions():
    """Requirement 16: Speaker labels from previous diarized file do not bleed into new session."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr = MockASR()
    mock_diar = MockDiarizer(predefined_segments=[
        SpeakerSegment(start=0.0, end=2.0, speaker="Speaker 9"),
        SpeakerSegment(start=2.0, end=4.0, speaker="Speaker 10"),
    ])

    session = AudioStreamingSession(
        chunk_duration_seconds=2.0,
        asr_instance=mock_asr,
        enable_diarization=True,
        diarizer_instance=mock_diar,
    )
    # Session 1: Diarization active with custom speakers
    session.load_audio_source(wav_bytes, filename="multi.wav")
    session.step()
    assert any("Speaker 9" in s.get("speaker", "") for s in session.emitted_segments)

    # Session 2: Load new audio with diarization disabled
    session.enable_diarization = False
    session.load_audio_source(wav_bytes, filename="mono.wav")
    assert len(session.speaker_segments) == 0
    session.step()
    assert all(s.get("speaker") == "Speaker 1" for s in session.emitted_segments)


def test_17_no_stale_phrase_results_between_sessions():
    """Requirement 17: Extracted phrases from session 1 do not persist into session 2."""
    wav_bytes = _create_synthetic_wav_bytes(duration_sec=4.0)
    mock_asr_1 = MockASR(transcriptions=[
        "machine learning machine learning",
        "machine learning models",
    ])
    session = AudioStreamingSession(chunk_duration_seconds=2.0, min_frequency=2, asr_instance=mock_asr_1)
    session.load_audio_source(wav_bytes, filename="ml.wav")
    while session.current_chunk_idx < session.total_chunks:
        session.step()

    ml_phrases = [p["phrase"] for p in session.current_rankings["ranked_phrases"]]
    assert "machine learning" in ml_phrases

    # Session 2 with completely different text
    mock_asr_2 = MockASR(transcriptions=[
        "solar panels solar panels",
        "solar panels clean",
    ])
    session.asr = mock_asr_2
    session.load_audio_source(wav_bytes, filename="solar.wav")
    while session.current_chunk_idx < session.total_chunks:
        session.step()

    solar_phrases = [p["phrase"] for p in session.current_rankings["ranked_phrases"]]
    assert "solar panels" in solar_phrases
    assert "machine learning" not in solar_phrases


def test_18_existing_deterministic_phrase_ranking_remains_unchanged():
    """Requirement 18: Deterministic phrase ranking retains exact scoring, category, and order."""
    corpus = [
        "artificial", "intelligence", "artificial", "intelligence",
        "infrastructure", "artificial", "intelligence", "systems"
    ]
    res_1 = analyze_and_rank_phrases(tokens=corpus, min_frequency=2, top_k=5)
    res_2 = analyze_and_rank_phrases(tokens=corpus, min_frequency=2, top_k=5)
    rankings_1 = res_1["ranked_phrases"]
    rankings_2 = res_2["ranked_phrases"]

    assert len(rankings_1) > 0
    assert len(rankings_1) == len(rankings_2)

    for p1, p2 in zip(rankings_1, rankings_2):
        assert p1["phrase"] == p2["phrase"]
        assert p1["frequency"] == p2["frequency"]
        assert p1["quality_score"] == p2["quality_score"]
        assert p1["quality_category"] == p2["quality_category"]

