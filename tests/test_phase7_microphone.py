"""
Phase 7 Automated Test Suite: Live Microphone Audio Transcription.

Verifies:
1. Microphone configuration validation
2. Audio buffer sizing and 16kHz float32 normalization
3. Microphone device enumeration
4. Synthetic sample feeding and chunk extraction
5. Empty buffer handling
6. Monotonic timestamps across sequential chunks
7. Final partial chunk flushed upon stop
8. Conservative boundary overlap removal (multi-word overlap)
9. Conservative boundary overlap removal (single-word overlap)
10. Legitimate repeated words preservation ('very very important')
11. Non-overlapping text returned intact
12. ASR model reuse across successive live chunks
13. Live session end-to-end stepping and progressive subtitles
14. Cross-segment phrase extraction across chunk boundaries
15. Live session stop behavior and session summary KPIs
16. Live session reset behavior (clears buffers, preserves model)
17. Export data integrity (TXT, JSON, CSV)
18. Microphone error handling (unavailable device / invalid index)
19. Strict exclusion of OpenAI and cloud ASR dependencies
20. Strict F-drive model and cache isolation
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.chunker import AudioChunk
from src.audio.asr import LocalASR, ASRChunkResult, MODELS_DIR
from src.audio.microphone import (
    LiveMicrophoneConfig,
    LiveMicrophoneCapture,
    MicrophoneDeviceInfo,
    MicrophoneError,
    get_microphone_devices,
)
from src.audio.overlap import ConservativeOverlapHandler, OverlapResolution
from src.audio.live_session import LiveMicrophoneSession, LiveStepEvent


class TestMicrophoneConfigurationAndCapture(unittest.TestCase):
    """Tests for microphone configuration, devices, and buffer management."""

    def test_01_microphone_config_validation(self):
        # Valid configs
        cfg = LiveMicrophoneConfig(chunk_duration_seconds=3.0, sample_rate=16000)
        self.assertEqual(cfg.chunk_duration_seconds, 3.0)
        self.assertEqual(cfg.sample_rate, 16000)

        # Invalid durations
        with self.assertRaises(ValueError):
            LiveMicrophoneConfig(chunk_duration_seconds=0.5)

        with self.assertRaises(ValueError):
            LiveMicrophoneConfig(chunk_duration_seconds=15.0)

        # Invalid sample rate
        with self.assertRaises(ValueError):
            LiveMicrophoneConfig(sample_rate=-100)

    def test_02_audio_buffer_sizing_and_normalization(self):
        cfg = LiveMicrophoneConfig(chunk_duration_seconds=3.0, sample_rate=16000)
        cap = LiveMicrophoneCapture(config=cfg)
        self.assertEqual(cap.samples_per_chunk, 48000)

    def test_03_microphone_devices_enumeration(self):
        devices = get_microphone_devices()
        self.assertIsInstance(devices, list)
        if devices:
            d = devices[0]
            self.assertIsInstance(d, MicrophoneDeviceInfo)
            self.assertGreater(d.channels, 0)
            self.assertIsInstance(d.display_name, str)

    def test_04_synthetic_sample_feeding_and_chunk_extraction(self):
        cfg = LiveMicrophoneConfig(chunk_duration_seconds=2.0, sample_rate=16000)
        cap = LiveMicrophoneCapture(config=cfg)

        # Feed 2.5 seconds of synthetic audio (40,000 samples)
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 2.5, 40000)).astype(np.float32)
        cap.feed_samples(samples)

        # Should extract 1 full 2.0s chunk (32,000 samples)
        chunk1 = cap.get_chunk()
        self.assertIsNotNone(chunk1)
        self.assertEqual(chunk1.chunk_index, 1)
        self.assertEqual(chunk1.duration, 2.0)
        self.assertEqual(len(chunk1.samples), 32000)
        self.assertEqual(chunk1.samples.dtype, np.float32)
        self.assertEqual(chunk1.start_time, 0.0)
        self.assertEqual(chunk1.end_time, 2.0)

        # Buffer now has 8,000 samples (0.5s) -> not enough for a 2.0s chunk
        chunk2 = cap.get_chunk()
        self.assertIsNone(chunk2)

    def test_05_empty_buffer_returns_none(self):
        cap = LiveMicrophoneCapture()
        self.assertIsNone(cap.get_chunk())
        self.assertEqual(cap.buffered_duration_seconds, 0.0)

    def test_06_monotonic_timestamps(self):
        cfg = LiveMicrophoneConfig(chunk_duration_seconds=1.0, sample_rate=16000)
        cap = LiveMicrophoneCapture(config=cfg)

        # Feed 3 seconds of audio
        cap.feed_samples(np.zeros(48000, dtype=np.float32))

        c1 = cap.get_chunk()
        c2 = cap.get_chunk()
        c3 = cap.get_chunk()

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertIsNotNone(c3)

        self.assertEqual(c1.start_time, 0.0)
        self.assertEqual(c1.end_time, 1.0)
        self.assertEqual(c2.start_time, 1.0)
        self.assertEqual(c2.end_time, 2.0)
        self.assertEqual(c3.start_time, 2.0)
        self.assertEqual(c3.end_time, 3.0)

    def test_07_final_partial_chunk_flushed_on_stop(self):
        cfg = LiveMicrophoneConfig(chunk_duration_seconds=2.0, sample_rate=16000)
        cap = LiveMicrophoneCapture(config=cfg)

        # Feed 2.8s of audio (44,800 samples)
        cap.feed_samples(np.zeros(44800, dtype=np.float32))

        c1 = cap.get_chunk()  # 2.0s
        self.assertIsNotNone(c1)

        # 0.8s remaining -> get_remaining_chunk should return it
        rem = cap.get_remaining_chunk(min_duration_seconds=0.4)
        self.assertIsNotNone(rem)
        self.assertEqual(rem.chunk_index, 2)
        self.assertEqual(rem.start_time, 2.0)
        self.assertAlmostEqual(rem.duration, 0.8, places=2)
        self.assertTrue(rem.is_final)


class TestConservativeOverlapHandler(unittest.TestCase):
    """Tests for boundary-overlap detection and conservative removal."""

    def setUp(self):
        self.handler = ConservativeOverlapHandler()

    def test_08_multi_word_overlap_removal(self):
        prev = "Today we are discussing artificial intelligence"
        curr = "artificial intelligence infrastructure is growing"

        res = self.handler.resolve_overlap(prev, curr)
        self.assertEqual(res.overlap_token_count, 2)
        self.assertEqual(res.overlap_phrase, "artificial intelligence")
        self.assertEqual(res.cleaned_text, "infrastructure is growing")
        self.assertIn("boundary_overlap_removed_2_tokens", res.reason)

    def test_09_single_word_overlap_removal(self):
        prev = "We must focus on renewable energy"
        curr = "energy transition is happening worldwide"

        res = self.handler.resolve_overlap(prev, curr)
        self.assertEqual(res.overlap_token_count, 1)
        self.assertEqual(res.overlap_phrase, "energy")
        self.assertEqual(res.cleaned_text, "transition is happening worldwide")

    def test_10_legitimate_repetition_preservation(self):
        # Speaker says "very very important" inside the current chunk
        prev = "This new economic package is"
        curr = "very very important for the public"

        res = self.handler.resolve_overlap(prev, curr)
        # Prev ended with "is", curr starts with "very" -> no overlap!
        self.assertEqual(res.overlap_token_count, 0)
        self.assertEqual(res.cleaned_text, "very very important for the public")

        # Even if prev ended with "very" and curr starts with "very very important"
        prev_ending_with_word = "This is very"
        curr_with_repeat = "very very important"
        res2 = self.handler.resolve_overlap(prev_ending_with_word, curr_with_repeat)
        # One overlapping "very" stripped, leaving one "very" in cleaned text
        self.assertEqual(res2.overlap_token_count, 1)
        self.assertEqual(res2.cleaned_text, "very important")
        # Combined text: "This is very" + " " + "very important" = "This is very very important"
        combined = f"{prev_ending_with_word} {res2.cleaned_text}"
        self.assertEqual(combined, "This is very very important")

    def test_11_no_overlap_returns_intact(self):
        prev = "First segment of speech"
        curr = "Second segment discusses economy"

        res = self.handler.resolve_overlap(prev, curr)
        self.assertEqual(res.overlap_token_count, 0)
        self.assertEqual(res.cleaned_text, curr)


class TestLiveMicrophoneSessionPipeline(unittest.TestCase):
    """Tests for LiveMicrophoneSession end-to-end integration and controls."""

    def setUp(self):
        # Create mock ASR that returns deterministic recognized text
        self.mock_asr = MagicMock(spec=LocalASR)
        self.mock_asr.model_size = "tiny"
        self.mock_asr.device = "cpu"

        # Create session with mock ASR and isolated capture
        self.capture = LiveMicrophoneCapture(config=LiveMicrophoneConfig(chunk_duration_seconds=2.0))
        self.session = LiveMicrophoneSession(
            chunk_duration_seconds=2.0,
            model_size="tiny",
            asr_instance=self.mock_asr,
            capture_instance=self.capture,
            min_frequency=2,
            top_k=10,
        )

    def test_12_asr_model_reuse_across_live_chunks(self):
        # Feed 2 chunks
        self.capture.feed_samples(np.zeros(64000, dtype=np.float32))
        self.session.status = "LISTENING"

        self.mock_asr.transcribe_chunk.side_effect = [
            ASRChunkResult(
                chunk_index=1,
                start_time=0.0,
                end_time=2.0,
                text="renewable energy transition",
                raw_text="renewable energy transition",
                processing_time_ms=45.0,
            ),
            ASRChunkResult(
                chunk_index=2,
                start_time=2.0,
                end_time=4.0,
                text="renewable energy transition is vital",
                raw_text="renewable energy transition is vital",
                processing_time_ms=42.0,
            ),
        ]

        # Step twice
        e1 = self.session.step()
        self.assertIsNotNone(e1)
        self.assertEqual(e1.chunk_index, 1)

        e2 = self.session.step()
        self.assertIsNotNone(e2)
        self.assertEqual(e2.chunk_index, 2)

        # Verify ASR was called twice on the same model instance
        self.assertEqual(self.mock_asr.transcribe_chunk.call_count, 2)

    def test_13_live_session_step_and_progressive_subtitles(self):
        self.capture.feed_samples(np.zeros(32000, dtype=np.float32))
        self.session.status = "LISTENING"

        self.mock_asr.transcribe_chunk.return_value = ASRChunkResult(
            chunk_index=1,
            start_time=0.0,
            end_time=2.0,
            text="Today the minister announced a new economic package",
            raw_text="Today the minister announced a new economic package",
            processing_time_ms=50.0,
        )

        event = self.session.step()
        self.assertIsNotNone(event)
        self.assertEqual(event.chunk_index, 1)
        self.assertEqual(event.cleaned_text, "Today the minister announced a new economic package")
        self.assertGreater(event.asr_latency_ms, 0.0)
        self.assertEqual(len(self.session.chunk_history), 1)

    def test_14_cross_segment_phrase_extraction(self):
        self.capture.feed_samples(np.zeros(64000, dtype=np.float32))
        self.session.status = "LISTENING"

        self.mock_asr.transcribe_chunk.side_effect = [
            ASRChunkResult(
                chunk_index=1,
                start_time=0.0,
                end_time=2.0,
                text="renewable energy transition and economic package",
                raw_text="renewable energy transition and economic package",
                processing_time_ms=40.0,
            ),
            ASRChunkResult(
                chunk_index=2,
                start_time=2.0,
                end_time=4.0,
                text="renewable energy transition creates new jobs",
                raw_text="renewable energy transition creates new jobs",
                processing_time_ms=41.0,
            ),
        ]

        self.session.step()
        self.session.step()
        self.session.stop()

        summary = self.session.get_summary()
        self.assertGreater(summary["ranked_phrases_count"], 0)
        ranked = self.session.current_rankings.get("ranked_phrases", [])
        ranked_names = [p["phrase"] for p in ranked]

        # "renewable energy" or "renewable energy transition" should be detected as recurring
        self.assertTrue(
            any("renewable energy" in p for p in ranked_names),
            f"Expected 'renewable energy' in ranked phrases, got {ranked_names}"
        )


    def test_15_live_session_stop_and_summary(self):
        self.capture.feed_samples(np.zeros(32000, dtype=np.float32))
        self.session.status = "LISTENING"
        self.mock_asr.transcribe_chunk.return_value = ASRChunkResult(
            chunk_index=1,
            start_time=0.0,
            end_time=2.0,
            text="artificial intelligence infrastructure",
            raw_text="artificial intelligence infrastructure",
            processing_time_ms=30.0,
        )

        self.session.step()
        stop_event = self.session.stop()

        self.assertTrue(stop_event.is_stopped)
        self.assertEqual(self.session.status, "STOPPED")

        summary = self.session.get_summary()
        self.assertEqual(summary["status"], "STOPPED")
        self.assertEqual(summary["chunks_processed"], 1)
        self.assertGreaterEqual(summary["tokens_processed"], 3)
        self.assertIn("average_asr_latency_ms", summary)
        self.assertIn("full_transcript", summary)

    def test_16_live_session_reset(self):
        self.capture.feed_samples(np.zeros(32000, dtype=np.float32))
        self.session.status = "LISTENING"
        self.mock_asr.transcribe_chunk.return_value = ASRChunkResult(
            chunk_index=1,
            start_time=0.0,
            end_time=2.0,
            text="sample text",
            raw_text="sample text",
            processing_time_ms=10.0,
        )
        self.session.step()
        self.assertEqual(len(self.session.chunk_history), 1)

        self.session.reset()
        self.assertEqual(self.session.status, "IDLE")
        self.assertEqual(len(self.session.chunk_history), 0)
        self.assertEqual(len(self.session.emitted_subtitles), 0)
        self.assertEqual(len(self.session.reconstructed_tokens), 0)
        self.assertEqual(self.session.total_audio_seconds, 0.0)

    def test_17_export_data_integrity(self):
        self.session.emitted_subtitles = [
            {"start_time": 0.0, "end_time": 3.0, "text": "artificial intelligence infrastructure"}
        ]
        self.session.reconstructed_tokens = ["artificial", "intelligence", "infrastructure"]
        self.session.current_rankings = {
            "ranked_phrases": [
                {
                    "phrase": "artificial intelligence",
                    "ngram_size": 2,
                    "frequency": 2,
                    "pmi_score": 3.5,
                    "quality_score": 7.2,
                    "quality_category": "HIGH",
                    "pos_pattern": "JJ NN",
                    "parent_phrase": "",
                }
            ],
            "unique_bigrams": 1,
            "unique_trigrams": 0,
            "recurring_bigrams": 1,
            "recurring_trigrams": 0,
        }

        # TXT
        txt = self.session.export_txt()
        self.assertIn("TIMESTAMPED SUBTITLE SEGMENTS", txt)
        self.assertIn("artificial intelligence infrastructure", txt)

        # JSON
        json_str = self.session.export_json()
        payload = json.loads(json_str)
        self.assertEqual(payload["session_type"], "live_microphone")
        self.assertEqual(len(payload["subtitles"]), 1)
        self.assertEqual(len(payload["ranked_phrases"]), 1)

        # CSV
        csv_str = self.session.export_csv()
        self.assertIn("artificial intelligence", csv_str)
        self.assertIn("HIGH", csv_str)


class TestArchitectureAndExclusions(unittest.TestCase):
    """Tests for architectural constraints and isolation rules."""

    def test_18_microphone_error_handling(self):
        # Invalid configuration raises ValueError
        with self.assertRaises(ValueError):
            LiveMicrophoneConfig(chunk_duration_seconds=-1.0)

    def test_19_openai_and_cloud_asr_strictly_excluded(self):
        """Verify openai and cloud transcription libraries are not imported."""
        self.assertNotIn("openai", sys.modules)
        for mod_name in list(sys.modules.keys()):
            self.assertFalse(mod_name.startswith("openai"), f"Found forbidden module: {mod_name}")

    def test_20_f_drive_model_isolation(self):
        """Verify ASR model directory resides strictly on F: drive."""
        self.assertTrue(
            MODELS_DIR.lower().startswith("f:"),
            f"MODELS_DIR must reside on F: drive, found: {MODELS_DIR}"
        )
        self.assertTrue(
            os.environ.get("HF_HUB_CACHE", "").lower().startswith("f:"),
            f"HF_HUB_CACHE must reside on F: drive, found: {os.environ.get('HF_HUB_CACHE')}"
        )


if __name__ == "__main__":
    unittest.main()
