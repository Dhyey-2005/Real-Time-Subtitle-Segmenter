"""
Streamlit Web UI Application for Phase 7.

Provides an interactive real-time dashboard for:
- Mode A: Progressive Uploaded Audio Streaming & Local ASR Transcription (WAV, MP3, M4A)
- Mode B: Live Local Microphone Continuous Audio Streaming & Real-Time ASR Transcription
- Mode C: Progressive Text Streaming Simulation (Phase 5 Regression / Demo Mode)
- Explainable deterministic subtitle segmentation display with timestamps
- Real-time n-gram collocation & phrase extraction ticker
- Contextual phrase highlighting in subtitles
- Filterable candidate ranking table with quality categories (HIGH / MEDIUM / LOW)
- Local client-side session exports (TXT, JSON, CSV)
"""

import os
import sys
import time
import json
import re
import base64
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce project-local NLTK data directory
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if os.path.exists(NLTK_DATA_DIR):
    import nltk
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

from src.text_processor import load_text
from src.streaming_segmenter import StreamingConfig
from src.ui_session import StreamingSimulationSession, highlight_phrases_in_text
from src.gemini_client import GeminiSemanticClient
from src.audio.loader import AudioMetadata, AudioLoadError, load_audio
from src.audio.pipeline import AudioStreamingSession, AudioStreamStepEvent
from src.audio.asr import LocalASR, MODELS_DIR
from src.audio.microphone import (
    LiveMicrophoneCapture,
    LiveMicrophoneConfig,
    MicrophoneDeviceInfo,
    MicrophoneError,
    get_microphone_devices,
)
from src.audio.live_session import LiveMicrophoneSession, LiveStepEvent
from src.audio.diarization import compute_diarization_summary, SpeakerSegment


def render_html(html_str: str):
    """Render HTML safely without markdown 4-space code block indentation artifacts."""
    cleaned = re.sub(r"^[ \t]+", "", html_str, flags=re.MULTILINE).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


# Configure page settings
st.set_page_config(
    page_title="Real-Time Subtitle Segmenter & ASR Phrase Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom layout and component polish
render_html(
    """
    <style>
    /* Global layout enhancements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    /* Metric Card Polish */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Clean button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }

    /* Download button styling */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    </style>
    """
)

SAMPLE_FILES = {
    "Growth of AI (~70 Words)": os.path.join(PROJECT_ROOT, "data", "input", "ai_growth_transcript.txt"),
    "Sample Transcript": os.path.join(PROJECT_ROOT, "data", "input", "sample_transcript.txt"),
    "Phrase Extraction Test Transcript": os.path.join(PROJECT_ROOT, "data", "input", "phrase_extraction_test_transcript.txt"),
    "Segmentation Test Transcript": os.path.join(PROJECT_ROOT, "data", "input", "segmentation_test_transcript.txt"),
    "Edge Case Transcript": os.path.join(PROJECT_ROOT, "data", "input", "edge_case_transcript.txt"),
}

SAMPLE_SPEECH_PATH = os.path.join(PROJECT_ROOT, "data", "input", "sample_speech.wav")
SAMPLE_MULTI_SPEAKER_PATH = os.path.join(PROJECT_ROOT, "data", "input", "multi_speaker_test.wav")


def get_default_transcript(key: str) -> str:
    path = SAMPLE_FILES.get(key, "")
    if path and os.path.exists(path):
        return load_text(path)
    return "the government announced a new economic package today to accelerate renewable energy transition"


@st.cache_resource(show_spinner="Initializing local faster-whisper ASR model on F: drive...")
def get_cached_asr(model_size: str = "base") -> LocalASR:
    """Load and cache the local faster-whisper ASR model on F: drive."""
    asr = LocalASR(model_size=model_size)
    asr.load_model()
    return asr


# Initialize Session State
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📁 Mode A — Uploaded Audio (Local ASR)"

# Text mode states
if "session" not in st.session_state:
    st.session_state.session = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Sample Transcript"
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = get_default_transcript("Sample Transcript")
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "delay_val" not in st.session_state:
    st.session_state.delay_val = 0.08
if "chunk_size_val" not in st.session_state:
    st.session_state.chunk_size_val = 3
if "min_freq_val" not in st.session_state:
    st.session_state.min_freq_val = 2
if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 20

# Audio mode states
if "audio_session" not in st.session_state:
    st.session_state.audio_session = None
if "audio_is_running" not in st.session_state:
    st.session_state.audio_is_running = False
if "audio_model_size" not in st.session_state:
    st.session_state.audio_model_size = "base"
if "audio_chunk_sec" not in st.session_state:
    st.session_state.audio_chunk_sec = 5.0
if "audio_source_type" not in st.session_state:
    st.session_state.audio_source_type = "Preloaded Speech Sample (15.1s WAV)"
if "uploaded_audio_bytes" not in st.session_state:
    st.session_state.uploaded_audio_bytes = None
if "uploaded_audio_name" not in st.session_state:
    st.session_state.uploaded_audio_name = ""
if "audio_enable_diarization" not in st.session_state:
    st.session_state.audio_enable_diarization = False
if "audio_spk_mode" not in st.session_state:
    st.session_state.audio_spk_mode = "Automatic"
if "audio_num_speakers" not in st.session_state:
    st.session_state.audio_num_speakers = 2

if "audio_error" not in st.session_state:
    st.session_state.audio_error = None
if "audio_processing_stage" not in st.session_state:
    st.session_state.audio_processing_stage = "Ready"
if "audio_autoplay" not in st.session_state:
    st.session_state.audio_autoplay = False

# Live Microphone states
if "mic_session" not in st.session_state:
    st.session_state.mic_session = None
if "mic_is_running" not in st.session_state:
    st.session_state.mic_is_running = False
if "mic_model_size" not in st.session_state:
    st.session_state.mic_model_size = "base"
if "mic_chunk_sec" not in st.session_state:
    st.session_state.mic_chunk_sec = 3.0
if "mic_device_idx" not in st.session_state:
    st.session_state.mic_device_idx = None
if "mic_error" not in st.session_state:
    st.session_state.mic_error = None


def init_text_session():
    cfg = StreamingConfig(
        chunk_size=st.session_state.chunk_size_val,
        simulated_delay=0.0,
    )
    st.session_state.session = StreamingSimulationSession(
        transcript_text=st.session_state.transcript_text,
        config=cfg,
        min_frequency=st.session_state.min_freq_val,
        top_k=st.session_state.top_k_val,
    )


def init_audio_session():
    st.session_state.audio_error = None
    st.session_state.audio_processing_stage = "Ready"
    asr_inst = get_cached_asr(st.session_state.audio_model_size)
    num_spk = (
        st.session_state.audio_num_speakers
        if (st.session_state.audio_enable_diarization and st.session_state.audio_spk_mode == "Known number of speakers")
        else None
    )
    sess = AudioStreamingSession(
        chunk_duration_seconds=st.session_state.audio_chunk_sec,
        model_size=st.session_state.audio_model_size,
        min_frequency=st.session_state.min_freq_val,
        top_k=st.session_state.top_k_val,
        asr_instance=asr_inst,
        enable_diarization=st.session_state.audio_enable_diarization,
        num_speakers=num_spk,
    )

    try:
        st.session_state.audio_processing_stage = (
            "Running speaker diarization..." if st.session_state.audio_enable_diarization else "Preparing audio..."
        )
        if st.session_state.audio_source_type == "Preloaded Speech Sample (15.1s WAV)" and os.path.exists(SAMPLE_SPEECH_PATH):
            sess.load_audio_source(SAMPLE_SPEECH_PATH, filename="sample_speech.wav")
        elif st.session_state.audio_source_type == "Preloaded Multi-Speaker Sample (21.9s WAV)" and os.path.exists(SAMPLE_MULTI_SPEAKER_PATH):
            sess.load_audio_source(SAMPLE_MULTI_SPEAKER_PATH, filename="multi_speaker_test.wav")
        elif st.session_state.uploaded_audio_bytes is not None:
            sess.load_audio_source(
                st.session_state.uploaded_audio_bytes,
                filename=st.session_state.uploaded_audio_name or "audio.wav",
            )
        if "Preloaded" in st.session_state.audio_source_type and sess.total_chunks > 0:
            while sess.current_chunk_idx < sess.total_chunks:
                sess.step()
            sess.status = "COMPLETED"
        st.session_state.audio_processing_stage = "Ready"
    except AudioLoadError as e:
        st.session_state.audio_error = str(e)
        st.session_state.audio_is_running = False
        st.session_state.audio_processing_stage = "Error"
    except Exception as e:
        st.session_state.audio_error = f"The audio file could not be decoded: {str(e)}"
        st.session_state.audio_is_running = False
        st.session_state.audio_processing_stage = "Error"

    st.session_state.audio_session = sess


def get_current_audio_data_uri() -> str:
    """Return base64 data URI for the currently selected audio source."""
    try:
        if st.session_state.audio_source_type == "Preloaded Speech Sample (15.1s WAV)" and os.path.exists(SAMPLE_SPEECH_PATH):
            with open(SAMPLE_SPEECH_PATH, "rb") as f:
                raw_bytes = f.read()
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            return f"data:audio/wav;base64,{b64}"
        elif st.session_state.audio_source_type == "Preloaded Multi-Speaker Sample (21.9s WAV)" and os.path.exists(SAMPLE_MULTI_SPEAKER_PATH):
            with open(SAMPLE_MULTI_SPEAKER_PATH, "rb") as f:
                raw_bytes = f.read()
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            return f"data:audio/wav;base64,{b64}"
        elif st.session_state.uploaded_audio_bytes is not None:
            ext = "wav"
            if st.session_state.uploaded_audio_name:
                parts = st.session_state.uploaded_audio_name.split(".")
                if len(parts) > 1:
                    raw_ext = parts[-1].lower()
                    if raw_ext in ("wav", "mp3", "m4a"):
                        ext = "mp4" if raw_ext == "m4a" else raw_ext
            b64 = base64.b64encode(st.session_state.uploaded_audio_bytes).decode("ascii")
            return f"data:audio/{ext};base64,{b64}"
    except Exception:
        pass
    return ""


def build_synced_segments_payload(emitted_segments: list, ranked_phrases: list, audio_duration: float = 0.0) -> list:
    """
    Format emitted subtitle segments into synchronized, non-overlapping intervals
    with pre-rendered highlighted phrase HTML for browser playback sync.
    """
    if not emitted_segments:
        return []

    sorted_segs = sorted(emitted_segments, key=lambda s: (s.get("start_time", 0.0), s.get("end_time", 0.0)))
    grouped = []
    i = 0
    while i < len(sorted_segs):
        j = i + 1
        st_i = sorted_segs[i].get("start_time", 0.0)
        et_i = sorted_segs[i].get("end_time", 0.0)
        group = [sorted_segs[i]]
        while j < len(sorted_segs) and abs(sorted_segs[j].get("start_time", 0.0) - st_i) < 0.001 and abs(sorted_segs[j].get("end_time", 0.0) - et_i) < 0.001:
            group.append(sorted_segs[j])
            j += 1

        if len(group) == 1:
            grouped.append(group[0])
        else:
            total_words = sum(max(1, len(g.get("text", "").split())) for g in group)
            dur = max(1.0, et_i - st_i)
            cur_st = st_i
            for g in group:
                w = max(1, len(g.get("text", "").split()))
                seg_dur = (w / total_words) * dur
                c = dict(g)
                c["start_time"] = round(cur_st, 2)
                cur_st += seg_dur
                c["end_time"] = round(cur_st, 2)
                grouped.append(c)
        i = j

    for idx in range(len(grouped)):
        if idx > 0 and grouped[idx]["start_time"] < grouped[idx - 1]["end_time"]:
            grouped[idx]["start_time"] = grouped[idx - 1]["end_time"]
        if grouped[idx]["end_time"] <= grouped[idx]["start_time"]:
            grouped[idx]["end_time"] = round(grouped[idx]["start_time"] + 1.5, 2)

    payload = []
    for idx, seg in enumerate(grouped):
        st_sec = float(seg.get("start_time", 0.0))
        et_sec = float(seg.get("end_time", 0.0))
        sm, ss = int(st_sec // 60), st_sec % 60
        em, es = int(et_sec // 60), et_sec % 60
        raw_text = seg.get("text", "")
        hl_html = highlight_phrases_in_text(raw_text, ranked_phrases)

        payload.append({
            "idx": idx,
            "seg_num": idx + 1,
            "start_time": st_sec,
            "end_time": et_sec,
            "start_str": f"{sm:02d}:{ss:05.2f}",
            "end_str": f"{em:02d}:{es:05.2f}",
            "text": raw_text,
            "html": hl_html,
            "speaker": seg.get("speaker", "Speaker 1"),
            "word_count": seg.get("word_count", len(raw_text.split())),
            "char_count": seg.get("char_count", len(raw_text)),
            "reason": seg.get("reason", "N/A"),
        })

    return payload


def init_mic_session():
    asr_inst = get_cached_asr(st.session_state.mic_model_size)
    sess = LiveMicrophoneSession(
        chunk_duration_seconds=st.session_state.mic_chunk_sec,
        model_size=st.session_state.mic_model_size,
        device_index=st.session_state.mic_device_idx,
        min_frequency=st.session_state.min_freq_val,
        top_k=st.session_state.top_k_val,
        asr_instance=asr_inst,
    )
    st.session_state.mic_session = sess


if st.session_state.session is None:
    init_text_session()

if st.session_state.audio_session is None:
    init_audio_session()

if st.session_state.mic_session is None:
    init_mic_session()

# Check Gemini status
gemini_client = GeminiSemanticClient()
gemini_status_text = "Available & Ready" if gemini_client.is_available() else "Not configured (Local deterministic mode)"

status_colors = {
    "IDLE": "#64748b",
    "READY": "#0284c7",
    "LISTENING": "#10b981",
    "PROCESSING": "#3b82f6",
    "PAUSED": "#f59e0b",
    "STOPPED": "#d97706",
    "COMPLETED": "#10b981",
    "ERROR": "#ef4444",
}

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Controls & Config")

    # Mode Selector
    st.subheader("1. Pipeline Mode")
    mode_options = [
        "📁 Mode A — Uploaded Audio (Local ASR)",
        "🎙️ Mode B — Live Microphone (Real-Time ASR)",
        "📄 Mode C — Text Streaming (Simulation Regression)",
    ]
    curr_idx = 0
    if "Live Microphone" in st.session_state.app_mode:
        curr_idx = 1
    elif "Text Streaming" in st.session_state.app_mode:
        curr_idx = 2

    app_mode = st.radio(
        "Select Operation Mode",
        options=mode_options,
        index=curr_idx,
    )
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
        st.session_state.is_running = False
        st.session_state.audio_is_running = False
        st.session_state.mic_is_running = False
        if st.session_state.mic_session and st.session_state.mic_session.capture.is_recording:
            st.session_state.mic_session.stop()
        st.rerun()

    st.markdown("---")

    if "Mode A" in st.session_state.app_mode:
        # MODE A — UPLOADED AUDIO SIDEBAR CONTROLS
        audio_src_options = [
            "Preloaded Speech Sample (15.1s WAV)",
            "Preloaded Multi-Speaker Sample (21.9s WAV)",
            "Upload Audio File (WAV, MP3, M4A)",
        ]
        src_index = audio_src_options.index(st.session_state.audio_source_type) if st.session_state.audio_source_type in audio_src_options else 0
        src_choice = st.selectbox(
            "Audio Input Source",
            options=audio_src_options,
            index=src_index,
        )

        if src_choice != st.session_state.audio_source_type:
            st.session_state.audio_source_type = src_choice
            init_audio_session()
            st.rerun()

        if src_choice == "Upload Audio File (WAV, MP3, M4A)":
            uploaded_file = st.file_uploader(
                "Upload Speech Audio (WAV, MP3, M4A)",
                type=["wav", "mp3", "m4a"],
                help="Supported formats: WAV, MP3, and M4A (Max 150 MB). Processed 100% locally on CPU/GPU.",
            )
            if uploaded_file is not None:
                content = uploaded_file.getvalue()
                if content != st.session_state.uploaded_audio_bytes:
                    st.session_state.uploaded_audio_bytes = content
                    st.session_state.uploaded_audio_name = uploaded_file.name
                    st.session_state.audio_error = None
                    init_audio_session()
                    st.rerun()

        st.markdown("---")
        st.subheader("3. ASR Model Configuration")
        model_choice = st.selectbox(
            "faster-whisper Model Size",
            options=["tiny", "base", "small", "medium"],
            index=["tiny", "base", "small", "medium"].index(st.session_state.audio_model_size),
            help="Local ASR model running on CPU with INT8 quantization. Model cache is isolated strictly to F: drive.",
        )
        if model_choice != st.session_state.audio_model_size:
            st.session_state.audio_model_size = model_choice
            init_audio_session()
            st.rerun()

        st.session_state.audio_chunk_sec = st.slider(
            "Audio Chunk Duration (seconds)",
            min_value=2.0,
            max_value=10.0,
            value=float(st.session_state.audio_chunk_sec),
            step=1.0,
            help="Duration of each audio chunk processed progressively by local ASR.",
        )

        st.markdown("---")
        st.subheader("4. Speaker Diarization")
        enable_diar = st.checkbox(
            "Enable speaker diarization",
            value=st.session_state.audio_enable_diarization,
            help="Identifies speaker turns and associates ASR text with speakers.",
        )
        if enable_diar != st.session_state.audio_enable_diarization:
            st.session_state.audio_enable_diarization = enable_diar
            init_audio_session()
            st.rerun()

        if st.session_state.audio_enable_diarization:
            st.caption("ℹ️ *Speaker 1 / Speaker 2 are anonymous speaker labels, not real biometric identities.*")
            spk_mode = st.radio(
                "Speaker Detection",
                options=["Automatic", "Known number of speakers"],
                index=0 if st.session_state.audio_spk_mode == "Automatic" else 1,
            )
            if spk_mode != st.session_state.audio_spk_mode:
                st.session_state.audio_spk_mode = spk_mode
                init_audio_session()
                st.rerun()

            if spk_mode == "Known number of speakers":
                num_spk = st.slider(
                    "Number of speakers (1–10)",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.audio_num_speakers),
                    step=1,
                )
                if num_spk != st.session_state.audio_num_speakers:
                    st.session_state.audio_num_speakers = num_spk
                    init_audio_session()
                    st.rerun()

    elif "Mode B" in st.session_state.app_mode:
        # MODE B — LIVE MICROPHONE SIDEBAR CONTROLS
        st.subheader("2. Microphone Hardware")
        available_devices = get_microphone_devices()
        dev_labels = ["System Default Microphone"] + [d.display_name for d in available_devices]
        dev_indices = [None] + [d.index for d in available_devices]

        selected_dev_label = st.selectbox(
            "Audio Input Device",
            options=dev_labels,
            index=0,
            help="Select local audio input device for microphone streaming.",
        )
        chosen_idx = dev_indices[dev_labels.index(selected_dev_label)]
        if chosen_idx != st.session_state.mic_device_idx:
            st.session_state.mic_device_idx = chosen_idx
            init_mic_session()

        st.markdown("---")
        st.subheader("3. Live ASR Configuration")
        mic_model_choice = st.selectbox(
            "faster-whisper Model Size",
            options=["tiny", "base", "small", "medium"],
            index=["tiny", "base", "small", "medium"].index(st.session_state.mic_model_size),
            help="Local ASR model running on CPU INT8. Model cache is strictly isolated to F: drive.",
            key="mic_model_selector",
        )
        if mic_model_choice != st.session_state.mic_model_size:
            st.session_state.mic_model_size = mic_model_choice
            init_mic_session()
            st.rerun()

        st.session_state.mic_chunk_sec = st.slider(
            "Live Chunk Window (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=float(st.session_state.mic_chunk_sec),
            step=0.5,
            help="Audio buffer window before triggering sequential ASR transcription. Default 3.0s.",
        )

    else:
        # MODE C — TEXT STREAMING SIDEBAR CONTROLS
        st.subheader("2. Transcript Source")
        sample_choice = st.selectbox(
            "Choose Preloaded Transcript",
            options=list(SAMPLE_FILES.keys()) + ["Custom Input"],
            index=0,
        )

        if sample_choice != st.session_state.selected_sample:
            st.session_state.selected_sample = sample_choice
            if sample_choice in SAMPLE_FILES:
                st.session_state.transcript_text = get_default_transcript(sample_choice)
                init_text_session()

        if sample_choice == "Custom Input":
            custom_txt = st.text_area(
                "Enter Unpunctuated Text",
                value=st.session_state.transcript_text,
                height=150,
            )
            if custom_txt != st.session_state.transcript_text:
                st.session_state.transcript_text = custom_txt
                init_text_session()

        st.markdown("---")
        st.subheader("3. Simulation Pacing")
        st.session_state.chunk_size_val = st.slider(
            "Words Per Chunk",
            min_value=1,
            max_value=8,
            value=st.session_state.chunk_size_val,
            step=1,
        )
        st.session_state.delay_val = st.slider(
            "Playback Delay (seconds)",
            min_value=0.01,
            max_value=0.5,
            value=st.session_state.delay_val,
            step=0.01,
        )

    st.markdown("---")
    st.subheader("NLP Extraction Filters")
    st.session_state.min_freq_val = st.number_input(
        "Min Phrase Frequency",
        min_value=1,
        max_value=10,
        value=st.session_state.min_freq_val,
        step=1,
    )
    st.session_state.top_k_val = st.number_input(
        "Top K Phrases Limit",
        min_value=5,
        max_value=50,
        value=st.session_state.top_k_val,
        step=5,
    )

    st.markdown("---")
    st.caption(f"🤖 **Semantic Client**: {gemini_status_text}")
    st.caption(f"📁 **Model Storage**: `F:\\Real-Time Subtitle Segmenter\\models`")

    with st.expander("📊 Evaluation & Benchmarks", expanded=False):
        st.caption("**Phase 10 Benchmark Summary**")
        st.caption("• Mean ASR WER: **3.17%** | CER: **1.83%**")
        st.caption("• Mean Offline RTF: **0.08x** (CPU INT8)")
        st.caption("• Subtitle Boundary Accuracy: **58.3%** (±0.5s)")
        st.caption("• Phrase Recall: **100.0%**")
        st.caption("📄 Report: `evaluation/reports/phase10_report.md`")
        st.caption("💾 Results: `evaluation/results/benchmark_results.json`")



# ==============================================================================
# ROUTE 1: MODE A — UPLOADED AUDIO STREAMING (PHASE 6)
# ==============================================================================
if "Mode A" in st.session_state.app_mode:
    asess: AudioStreamingSession = st.session_state.audio_session
    meta = asess.metadata

    st.title("🎬 Progressive Audio Subtitle Segmenter & ASR Phrase Extractor")
    st.markdown(
        "Process uploaded speech audio (**WAV, MP3, M4A**) into **progressive subtitle segments** "
        "and **real-time recurring n-gram collocations** using **local faster-whisper ASR** on CPU."
    )

    # Audio Error Banner or Metadata Display
    if st.session_state.audio_error:
        st.error(f"⚠️ {st.session_state.audio_error}")
        err_c1, _ = st.columns([2.5, 4.5])
        with err_c1:
            if st.button("🔄 Clear & Upload Another Audio File", key="audio_err_clear_btn", type="primary"):
                st.session_state.uploaded_audio_bytes = None
                st.session_state.uploaded_audio_name = ""
                st.session_state.audio_error = None
                st.session_state.audio_is_running = False
                init_audio_session()
                st.rerun()
    elif meta is not None:
        render_html(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px;">
                <div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.15rem;">🎵</span>
                        <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{meta.filename}</span>
                        <span style="background: #e2e8f0; color: #334155; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">{meta.format}</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">⏱️ {meta.formatted_duration}</span>
                        <span style="background: #f1f5f9; color: #475569; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🔊 {meta.sample_rate / 1000:.1f} kHz</span>
                        <span style="background: #f1f5f9; color: #475569; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">🎚️ {'Mono' if meta.channels == 1 else 'Stereo'}</span>
                        <span style="background: #f1f5f9; color: #475569; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem;">📦 {meta.formatted_size_kb}</span>
                    </div>
                </div>
            </div>
            """
        )

    else:
        st.warning("⚠️ No audio loaded. Please select a preloaded sample or upload an audio file (WAV, MP3, M4A).")

    # Control Action Buttons Bar
    col_b1, col_b2, col_b3, col_b4, col_status = st.columns([1.5, 1, 1, 1, 2.5])

    with col_b1:
        start_btn = st.button("▶️ Start ASR & Play Synced", type="primary", key="audio_start_btn")
    with col_b2:
        pause_btn = st.button("⏸️ Pause", key="audio_pause_btn")
    with col_b3:
        resume_btn = st.button("⏯️ Resume", key="audio_resume_btn")
    with col_b4:
        reset_btn = st.button("🔄 Reset", key="audio_reset_btn")

    if reset_btn:
        st.session_state.audio_is_running = False
        st.session_state.audio_autoplay = False
        asess.reset()
        st.rerun()

    if pause_btn:
        st.session_state.audio_is_running = False
        st.session_state.audio_autoplay = False
        asess.status = "PAUSED"
        st.rerun()

    if resume_btn:
        if asess.status in ("PAUSED", "COMPLETED"):
            st.session_state.audio_autoplay = True
        elif asess.status == "READY":
            if asess.total_chunks > 0:
                with st.spinner("⚡ Transcribing audio chunks with local faster-whisper ASR..."):
                    while asess.current_chunk_idx < asess.total_chunks:
                        asess.step()
                asess.status = "COMPLETED"
                st.session_state.audio_autoplay = True
        st.rerun()

    if start_btn:
        asess.reset()
        if asess.total_chunks > 0:
            with st.spinner("⚡ Transcribing audio chunks and extracting recurring key phrases with local Whisper ASR..."):
                while asess.current_chunk_idx < asess.total_chunks:
                    asess.step()
            asess.status = "COMPLETED"
        st.session_state.audio_is_running = False
        st.session_state.audio_autoplay = True
        st.rerun()

    curr_status = asess.status
    status_color = status_colors.get(curr_status, "#64748b")
    processed_dur = min(asess.current_chunk_idx * asess.chunk_duration, meta.duration_seconds if meta else 0.0)
    total_dur = meta.duration_seconds if meta else 0.0
    progress_val = round((asess.current_chunk_idx / max(1, asess.total_chunks)) * 100, 1) if asess.total_chunks > 0 else 0.0

    with col_status:
        render_html(
            f"""
            <div style="text-align: right; padding-top: 5px;">
                <span style="background-color: {status_color}; color: white; padding: 5px 12px; border-radius: 12px; font-weight: bold; font-size: 0.85rem;">
                    ● {curr_status}
                </span>
                <span style="margin-left: 10px; font-size: 0.85rem; color: #475569;">
                    Audio: <b>{processed_dur:.1f} / {total_dur:.1f}s</b> ({progress_val}%) | Chunk {asess.current_chunk_idx}/{asess.total_chunks}
                </span>
            </div>
            """
        )

    st.progress(progress_val / 100.0)

    # Processing Stage Feedback (Phase 9)
    if st.session_state.audio_is_running:
        st.info(f"⏳ **Processing Status**: {st.session_state.audio_processing_stage}")

    # RECURRING KEY PHRASE TICKER AREA
    st.markdown("### 🏷️ Recurring Key Phrase Ticker")
    ranked_phrases = asess.current_rankings.get("ranked_phrases", [])

    if ranked_phrases:
        ticker_chips = []
        for p in ranked_phrases[:8]:
            phrase_name = p.get("phrase", "").upper()
            category = p.get("quality_category", "LOW")
            score = p.get("quality_score", 0.0)
            freq = p.get("frequency", 0)

            if category == "HIGH":
                style = "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
                badge = "★ HIGH"
            elif category == "MEDIUM":
                style = "background: #e0f2fe; color: #075985; border: 1px solid #38bdf8;"
                badge = "▲ MED"
            else:
                style = "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;"
                badge = "• LOW"

            chip_html = f"""
            <span style="{style} padding: 4px 10px; border-radius: 16px; margin: 0 4px; display: inline-block; font-size: 0.85rem;">
                <b>{phrase_name}</b> <span style="opacity: 0.8; font-size: 0.75rem;">({freq}x | {badge} | {score:.2f})</span>
            </span>
            """
            ticker_chips.append(chip_html.strip())

        render_html(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; line-height: 2.2;">
                {" ".join(ticker_chips)}
            </div>
            """
        )
    elif asess.status == "COMPLETED":
        st.info("ℹ️ No recurring phrases were found.")
    else:
        st.info("Waiting for recurring phrases... (Start audio processing to transcribe chunks and discover phrases)")

    st.write("")

    # MAIN TWO-COLUMN DASHBOARD DISPLAY
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("📺 Progressive Subtitle Feed")

        audio_uri = get_current_audio_data_uri()
        synced_payload = build_synced_segments_payload(
            asess.emitted_segments,
            ranked_phrases,
            meta.duration_seconds if meta else 0.0,
        )
        payload_json = json.dumps(synced_payload)
        autoplay_val = "true" if st.session_state.audio_autoplay else "false"
        st.session_state.audio_autoplay = False

        first_seg = synced_payload[0] if synced_payload else None
        init_spk = first_seg.get("speaker", "Speaker 1") if first_seg and asess.enable_diarization else ""
        init_time = f"{first_seg['start_str']} → {first_seg['end_str']}" if first_seg else "00:00.00 → 00:00.00"
        init_html = first_seg["html"] if first_seg else (
            "<span style='color: #94a3b8; font-style: italic;'>Subtitles will appear here in real-time sync with audio playback as chunks are transcribed. Click '▶️ Start ASR & Play Synced' above to begin.</span>"
            if asess.status != "COMPLETED"
            else "<span style='color: #94a3b8;'>ℹ️ No speech was detected in this audio.</span>"
        )
        init_wc = first_seg["word_count"] if first_seg else 0
        init_cc = first_seg["char_count"] if first_seg else 0
        init_reason = first_seg["reason"] if first_seg else "N/A"
        spk_badge_initial = (
            f"<span style='background: #9333ea; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin-right: 8px; font-weight: bold;'>● {init_spk.upper()}</span>"
            if init_spk
            else ""
        )

        timeline_items_html_list = []
        if synced_payload:
            for item in synced_payload:
                s_num = item["seg_num"]
                s_start = item["start_time"]
                s_range = f"{item['start_str']} → {item['end_str']}"
                s_wc = item["word_count"]
                s_spk = item["speaker"] if asess.enable_diarization else None
                s_spk_span = f'<span style="font-weight: 700; color: #9333ea; font-size: 0.8rem;">● {s_spk}</span>' if s_spk else ""
                card_html = f"""
                <div class="sync-timeline-item" id="timeline-item-{item['idx']}" onclick="window.seekToSubtitle({s_start})" style="border-left: 4px solid #cbd5e1; padding: 10px 14px; margin-bottom: 10px; background: #ffffff; border-radius: 0 8px 8px 0; border: 1px solid #e2e8f0; border-left-width: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); cursor: pointer; transition: all 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">
                        <span>
                            <b>#{s_num}</b> &nbsp;
                            <code style="color: #475569;">{s_range}</code> &nbsp;·&nbsp; {s_wc} words
                            <span class="sync-active-indicator" style="display: none; background: #0284c7; color: white; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-left: 8px;">▶ ACTIVE NOW</span>
                        </span>
                        {s_spk_span}
                    </div>
                    <div style="font-size: 1rem; color: #0f172a; line-height: 1.55;">
                        {item['html']}
                    </div>
                </div>
                """
                timeline_items_html_list.append(card_html.strip())
            timeline_items_html = "".join(timeline_items_html_list)
        elif asess.status == "COMPLETED":
            timeline_items_html = "<div style='color: #64748b; font-size: 0.9rem; padding: 12px; background: #f8fafc; border-radius: 6px; border: 1px dashed #cbd5e1;'>ℹ️ No speech segments detected in this audio file.</div>"
        else:
            timeline_items_html = "<div style='color: #64748b; font-size: 0.9rem; padding: 12px; background: #f8fafc; border-radius: 6px; border: 1px dashed #cbd5e1;'>No finalized subtitles yet. Click '▶️ Start ASR & Play Synced' above to begin transcription.</div>"

        synced_ui_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 2px 2px 14px 2px;
    background: transparent;
    color: #0f172a;
}}
* {{
    box-sizing: border-box;
}}
.sync-spd-btn:hover {{
    filter: brightness(1.2);
}}
.sync-timeline-item:hover {{
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15) !important;
    border-color: #38bdf8 !important;
}}
</style>
</head>
<body>
<div style="background: #0f172a; color: #ffffff; border-radius: 12px; padding: 18px 22px; box-shadow: 0 4px 16px rgba(0,0,0,0.25); margin-bottom: 16px; border: 1px solid #1e293b;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span id="sync-spk-badge">{spk_badge_initial}</span>
            <span id="sync-status-indicator" style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.5px;">● SYNCED AUDIO SUBTITLE FEED</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 0.75rem; color: #94a3b8; margin-right: 2px;">Speed:</span>
            <button type="button" class="sync-spd-btn" onclick="window.setSyncSpeed(0.75, this)" style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">0.75x</button>
            <button type="button" class="sync-spd-btn" onclick="window.setSyncSpeed(1.0, this)" style="background: #0284c7; color: #ffffff; border: 1px solid #0284c7; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; font-weight: bold;">1.0x</button>
            <button type="button" class="sync-spd-btn" onclick="window.setSyncSpeed(1.25, this)" style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">1.25x</button>
            <button type="button" class="sync-spd-btn" onclick="window.setSyncSpeed(1.5, this)" style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">1.5x</button>
            <button type="button" class="sync-spd-btn" onclick="window.setSyncSpeed(2.0, this)" style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">2.0x</button>
            <span id="sync-time-badge" style="color: #94a3b8; font-family: monospace; font-size: 0.85rem; background: #1e293b; padding: 2px 8px; border-radius: 4px; margin-left: 6px;">{init_time}</span>
        </div>
    </div>

    <audio id="synced-audio-player" src="{audio_uri}" controls preload="auto" style="width: 100%; height: 40px; border-radius: 8px; margin-bottom: 14px; outline: none;"></audio>

    <div id="sync-subtitle-card" style="background: #1e293b; border-radius: 8px; padding: 16px 20px; border: 1px solid #334155; min-height: 80px; display: flex; align-items: center;">
        <div id="sync-subtitle-text" style="font-size: 1.35rem; font-weight: 500; line-height: 1.6; color: #f8fafc; width: 100%;">
            {init_html}
        </div>
    </div>

    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 10px; margin-top: 12px;">
        <div style="display: flex; gap: 16px;">
            <span>Words: <b id="sync-word-count" style="color: #f1f5f9;">{init_wc}</b></span>
            <span>Chars: <b id="sync-char-count" style="color: #f1f5f9;">{init_cc}</b></span>
            <span>Boundary Trigger: <b id="sync-reason" style="color: #38bdf8;">{init_reason}</b></span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span id="sync-speed-label" style="color: #a78bfa; font-weight: 600; font-size: 0.75rem;">1.0x Speed</span>
            <span id="sync-curtime-badge" style="color: #38bdf8; font-family: monospace; font-size: 0.8rem; background: #0c4a6e; padding: 2px 8px; border-radius: 4px;">▶ 00:00.0</span>
        </div>
    </div>
</div>

<div style="margin-top: 4px; margin-bottom: 16px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: 600; font-size: 0.95rem; color: #0f172a;">📜 Subtitle History Timeline</span>
        <span style="font-size: 0.75rem; color: #64748b; font-style: italic;">Click any card to jump audio playback</span>
    </div>
    <div id="sync-timeline-container" style="max-height: 480px; overflow-y: auto; padding-right: 4px;">
        {timeline_items_html}
    </div>
</div>

<script id="synced-data" type="application/json">
{payload_json}
</script>

<script>
(function() {{
    const dataElem = document.getElementById("synced-data");
    const segments = dataElem ? JSON.parse(dataElem.textContent) : [];
    const shouldAutoplay = {autoplay_val};
    const audio = document.getElementById("synced-audio-player");
    if (!audio) return;

    let lastActiveIdx = -1;

    function renderActive(idx) {{
        if (!segments || segments.length === 0 || idx < 0 || idx >= segments.length) return;
        const seg = segments[idx];

        const txtElem = document.getElementById("sync-subtitle-text");
        if (txtElem) txtElem.innerHTML = seg.html;

        const timeBadge = document.getElementById("sync-time-badge");
        if (timeBadge) timeBadge.innerText = `${{seg.start_str}} → ${{seg.end_str}}`;

        const wcElem = document.getElementById("sync-word-count");
        if (wcElem) wcElem.innerText = seg.word_count;

        const ccElem = document.getElementById("sync-char-count");
        if (ccElem) ccElem.innerText = seg.char_count;

        const reasonElem = document.getElementById("sync-reason");
        if (reasonElem) reasonElem.innerText = seg.reason;

        const spkElem = document.getElementById("sync-spk-badge");
        if (spkElem) {{
            spkElem.innerHTML = seg.speaker ? `<span style="background: #9333ea; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin-right: 8px; font-weight: bold;">● ${{seg.speaker.toUpperCase()}}</span>` : "";
        }}

        document.querySelectorAll(".sync-timeline-item").forEach((el, i) => {{
            const badge = el.querySelector(".sync-active-indicator");
            if (i === idx) {{
                el.style.borderLeft = "5px solid #0284c7";
                el.style.background = "#f0f9ff";
                el.style.boxShadow = "0 0 10px rgba(2, 132, 199, 0.25)";
                if (badge) badge.style.display = "inline-block";
                el.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
            }} else {{
                el.style.borderLeft = "4px solid #cbd5e1";
                el.style.background = "#ffffff";
                el.style.boxShadow = "none";
                if (badge) badge.style.display = "none";
            }}
        }});
    }}

    function onAudioTick() {{
        if (!audio) return;
        const cur = audio.currentTime;

        const m = Math.floor(cur / 60);
        const s = (cur % 60).toFixed(1);
        const curPill = document.getElementById("sync-curtime-badge");
        if (curPill) {{
            const mm = String(m).padStart(2, '0');
            const ss = String(s).padStart(4, '0');
            curPill.innerText = `▶ ${{mm}}:${{ss}}`;
        }}

        if (!segments || segments.length === 0) return;

        let activeIdx = -1;
        for (let i = 0; i < segments.length; i++) {{
            const isLast = (i === segments.length - 1);
            const nextStart = isLast ? Infinity : segments[i + 1].start_time;
            if (cur >= segments[i].start_time && cur < nextStart) {{
                activeIdx = i;
                break;
            }}
        }}
        if (activeIdx === -1 && segments.length > 0) {{
            activeIdx = (cur < segments[0].start_time) ? 0 : segments.length - 1;
        }}

        if (activeIdx !== lastActiveIdx && activeIdx >= 0) {{
            lastActiveIdx = activeIdx;
            renderActive(activeIdx);
        }}
    }}

    audio.addEventListener("timeupdate", onAudioTick);
    audio.addEventListener("seeked", onAudioTick);

    audio.addEventListener("play", function() {{
        const ind = document.getElementById("sync-status-indicator");
        if (ind) {{
            ind.innerText = "● PLAYING (SYNCED)";
            ind.style.color = "#10b981";
        }}
        onAudioTick();
    }});

    audio.addEventListener("pause", function() {{
        const ind = document.getElementById("sync-status-indicator");
        if (ind) {{
            ind.innerText = "⏸️ PAUSED";
            ind.style.color = "#f59e0b";
        }}
    }});

    audio.addEventListener("ended", function() {{
        const ind = document.getElementById("sync-status-indicator");
        if (ind) {{
            ind.innerText = "✓ PLAYBACK FINISHED";
            ind.style.color = "#38bdf8";
        }}
    }});

    audio.addEventListener("ratechange", function() {{
        const spd = document.getElementById("sync-speed-label");
        if (spd) spd.innerText = `${{audio.playbackRate}}x Speed`;
    }});

    window.seekToSubtitle = function(sec) {{
        if (audio) {{
            audio.currentTime = sec;
            audio.play().catch(function(e) {{}});
        }}
    }};

    window.setSyncSpeed = function(speed, btn) {{
        if (audio) {{
            audio.playbackRate = speed;
            document.querySelectorAll(".sync-spd-btn").forEach(b => {{
                b.style.background = "#1e293b";
                b.style.color = "#cbd5e1";
                b.style.borderColor = "#334155";
                b.style.fontWeight = "normal";
            }});
            if (btn) {{
                btn.style.background = "#0284c7";
                btn.style.color = "#ffffff";
                btn.style.borderColor = "#0284c7";
                btn.style.fontWeight = "bold";
            }}
        }}
    }};

    if (shouldAutoplay) {{
        audio.play().catch(function(err) {{
            console.log("Autoplay notice:", err);
        }});
    }}

    onAudioTick();
}})();
</script>
</body>
</html>
"""

        st.iframe(
            synced_ui_html,
            height=720,
            alt="Synchronized Real-Time Subtitle and Audio Feed",
        )

        # Dedicated Speaker Turns Table (Phase 8)
        if asess.enable_diarization and asess.emitted_segments:
            with st.expander("👥 Speaker Turns Timeline Table", expanded=True):
                df_turns = pd.DataFrame([
                    {
                        "Time": f"[{int(s.get('start_time', 0.0)//60):02d}:{s.get('start_time', 0.0)%60:05.2f} → {int(s.get('end_time', 0.0)//60):02d}:{s.get('end_time', 0.0)%60:05.2f}]",
                        "Speaker": s.get("speaker", "Speaker 1"),
                        "Subtitle": s.get("text", ""),
                    }
                    for s in asess.emitted_segments
                ])
                st.dataframe(df_turns, hide_index=True)

    with col_right:
        st.subheader("📊 Phrase Intelligence & Analytics")

        # 2x2 Structured Session KPI Grid
        kpi_r1_c1, kpi_r1_c2 = st.columns(2)
        kpi_r1_c1.metric("Recognized Words", len(asess.reconstructed_tokens))
        kpi_r1_c2.metric("Subtitles Emitted", len(asess.emitted_segments))

        kpi_r2_c1, kpi_r2_c2 = st.columns(2)
        kpi_r2_c1.metric("Ranked Phrases", len(ranked_phrases))
        rtf_val = f"{asess.real_time_factor:.2f}x" if asess.real_time_factor is not None else "N/A"
        kpi_r2_c2.metric(
            "Real-Time Factor (RTF)",
            rtf_val,
            help="Total processing time / audio duration. Values < 1.0 indicate faster-than-playback offline processing.",
        )

        # Performance summary note
        if asess.metadata and asess.total_processing_time_s > 0:
            render_html(
                f"""
                <div style="background: #f1f5f9; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; color: #475569; margin: 8px 0 16px 0;">
                    ⏱️ Audio: <b>{asess.metadata.formatted_duration}</b> &nbsp;·&nbsp;
                    Processing: <b>{asess.total_processing_time_s:.2f}s</b>
                    {f' (Diarization: {asess.diarization_time_s:.2f}s)' if asess.enable_diarization else ''} &nbsp;·&nbsp;
                    RTF: <b>{rtf_val}</b> (Offline INT8)
                </div>
                """
            )
        else:
            st.write("")

        # Ranked Phrases Table
        st.markdown("#### Ranked Collocations")
        if ranked_phrases:
            cat_filter = st.selectbox(
                "Filter by Quality Category",
                ["ALL", "HIGH", "MEDIUM", "LOW"],
                index=0,
                key="audio_cat_filt",
            )
            filtered_phrases = ranked_phrases
            if cat_filter != "ALL":
                filtered_phrases = [p for p in ranked_phrases if p.get("quality_category") == cat_filter]

            if filtered_phrases:
                df_phrases = pd.DataFrame(
                    [
                        {
                            "Rank": idx + 1,
                            "Phrase": p.get("phrase", ""),
                            "N-gram": f"{p.get('ngram_size')}-gram",
                            "Frequency": p.get("frequency", 0),
                            "PMI": f"{p.get('pmi', 0.0):.2f}",
                            "Score": f"{p.get('quality_score', 0.0):.2f}",
                            "Quality": p.get("quality_category", "LOW"),
                            "POS Pattern": p.get("pos_pattern", ""),
                        }
                        for idx, p in enumerate(filtered_phrases[:15])
                    ]
                )
                st.dataframe(df_phrases, hide_index=True)
            else:
                st.info(f"No phrases found in category '{cat_filter}'.")
        elif asess.status == "COMPLETED":
            st.info("ℹ️ No recurring phrases were found.")
        else:
            st.caption("No candidate phrases ranked yet.")

        # Overlapping / Nested phrase hierarchy
        nested_pairs = [
            (p.get("phrase"), p.get("parent_phrase"))
            for p in ranked_phrases
            if p.get("parent_phrase") and p.get("parent_phrase") != "none"
        ]
        if nested_pairs:
            with st.expander("🔗 Overlapping & Nested Phrase Hierarchy", expanded=False):
                for sub, parent in nested_pairs:
                    st.markdown(f"• **{parent}**  \n&nbsp;&nbsp;&nbsp;&nbsp;└── *{sub}*")

        # Reconstructed Transcript Area
        st.markdown("---")
        with st.expander("📝 Reconstructed Complete Transcript", expanded=False):
            if asess.reconstructed_transcript:
                st.write(asess.reconstructed_transcript)
            elif asess.status == "COMPLETED":
                st.info("ℹ️ No speech was detected in this audio.")
            else:
                st.caption("Transcript will appear as audio chunks are transcribed.")

        # Diarization Summary (Phase 8)
        if asess.enable_diarization and asess.status == "COMPLETED" and asess.emitted_segments:
            st.markdown("#### 👥 Speaker Diarization Summary")
            d_sum = compute_diarization_summary(asess.emitted_segments, asess.speaker_segments)
            st.write(f"**Speakers detected:** {d_sum['speaker_count']}")
            spk_items = d_sum.get("speaker_list", list(d_sum.get("speakers", {}).values()) if isinstance(d_sum.get("speakers"), dict) else d_sum.get("speakers", []))
            for spk in spk_items:
                st.write(f"- **{spk['speaker']}**: {spk['duration_seconds']}s ({spk['percentage']}%) | {spk['turn_count']} turns")
        elif asess.status == "COMPLETED" and not asess.enable_diarization:
            st.caption("ℹ️ Speaker diarization is disabled.")

        # Export Controls Area (2x2 grid for clean, un-squished layout)
        st.markdown("#### 💾 Export Audio Artifacts")
        if asess.emitted_segments:
            txt_data = asess.export_txt()
            json_data = asess.export_json()
            csv_data = asess.export_csv()
            sub_csv_data = asess.export_subtitles_csv()

            exp_c1, exp_c2 = st.columns(2)
            exp_c1.download_button(
                "📥 Transcript (.txt)",
                data=txt_data,
                file_name="transcript.txt",
                mime="text/plain",
                key="audio_export_txt",
            )
            exp_c2.download_button(
                "📥 Session Data (.json)",
                data=json_data,
                file_name="transcript.json",
                mime="application/json",
                key="audio_export_json",
            )

            exp_c3, exp_c4 = st.columns(2)
            exp_c3.download_button(
                "📥 Phrases (.csv)",
                data=csv_data if csv_data else "Rank,Phrase,NgramSize,Frequency,Score,Quality,POS",
                file_name="phrases.csv",
                mime="text/csv",
                key="audio_export_csv",
            )
            exp_c4.download_button(
                "📥 Subtitles (.csv)",
                data=sub_csv_data,
                file_name="subtitles.csv",
                mime="text/csv",
                key="audio_export_subtitles_csv",
            )
        else:
            st.caption("ℹ️ Export options will be available once audio processing produces results.")


    # Progressive streaming loop for Mode A
    if st.session_state.audio_is_running:
        if asess.current_chunk_idx < asess.total_chunks:
            chunk_num = asess.current_chunk_idx + 1
            st.session_state.audio_processing_stage = f"Running transcription (Chunk {chunk_num} of {asess.total_chunks})..."
            step_event = asess.step()
            st.session_state.audio_processing_stage = "Building subtitles & extracting phrases..."
            st.rerun()
        else:
            st.session_state.audio_processing_stage = "Finalizing results..."
            st.session_state.audio_is_running = False
            asess.status = "COMPLETED"
            st.session_state.audio_autoplay = True
            st.rerun()


# ==============================================================================
# ROUTE 2: MODE B — LIVE MICROPHONE STREAMING (PHASE 7)
# ==============================================================================
elif "Mode B" in st.session_state.app_mode:
    msess: LiveMicrophoneSession = st.session_state.mic_session

    st.title("🎙️ Live Microphone Subtitle Segmenter & Real-Time ASR Extractor")
    st.markdown(
        "Continuous live speech captioning directly from your **local microphone**. "
        "Audio is captured in short buffers, transcribed via local **faster-whisper ASR**, "
        "and structured into **real-time subtitles** and **recurring key n-gram phrases**."
    )

    # Hardware & Status Banner
    col_hw1, col_hw2, col_hw3, col_hw4, col_hw5 = st.columns(5)
    dev_name = "Default Microphone"
    devs = get_microphone_devices()
    for d in devs:
        if d.index == st.session_state.mic_device_idx:
            dev_name = d.name
            break

    col_hw1.metric("🎤 Device", dev_name[:18] + ("..." if len(dev_name) > 18 else ""))
    col_hw2.metric("🤖 ASR Engine", f"Whisper {st.session_state.mic_model_size.upper()}")
    col_hw3.metric("⚡ Compute", "CPU (INT8)")
    col_hw4.metric("⏱️ Buffer Chunk", f"{st.session_state.mic_chunk_sec:.1f} sec")
    col_hw5.metric("📦 Cache Root", "F: Drive")

    # Control Action Buttons Bar
    col_b1, col_b2, col_b3, col_status = st.columns([1.5, 1.5, 1.5, 3.5])

    with col_b1:
        mic_start_btn = st.button("▶️ Start Microphone", type="primary", key="mic_start_btn")
    with col_b2:
        mic_stop_btn = st.button("⏹️ Stop Microphone", key="mic_stop_btn")
    with col_b3:
        mic_reset_btn = st.button("🔄 Reset", key="mic_reset_btn")

    if mic_reset_btn:
        st.session_state.mic_is_running = False
        msess.reset()
        st.rerun()

    if mic_stop_btn:
        st.session_state.mic_is_running = False
        msess.stop()
        st.rerun()

    if mic_start_btn:
        try:
            msess.start()
            st.session_state.mic_is_running = True
            st.session_state.mic_error = None
        except Exception as e:
            st.session_state.mic_is_running = False
            st.session_state.mic_error = str(e)
        st.rerun()

    mic_curr_status = msess.status
    mic_status_color = status_colors.get(mic_curr_status, "#64748b")
    elapsed_live = msess.elapsed_live_seconds
    total_audio = msess.total_audio_seconds
    chunks_done = len(msess.chunk_history)

    with col_status:
        render_html(
            f"""
            <div style="text-align: right; padding-top: 5px;">
                <span style="background-color: {mic_status_color}; color: white; padding: 5px 12px; border-radius: 12px; font-weight: bold; font-size: 0.85rem;">
                    ● {mic_curr_status}
                </span>
                <span style="margin-left: 10px; font-size: 0.85rem; color: #475569;">
                    Live Time: <b>{elapsed_live:.1f}s</b> | Chunks: <b>{chunks_done}</b> | Tokens: <b>{len(msess.reconstructed_tokens)}</b>
                </span>
            </div>
            """
        )

    if st.session_state.mic_error:
        st.error(f"⚠️ Microphone Capture Error: {st.session_state.mic_error}")

    # Optional Browser Microphone Alternative
    with st.expander("🌐 Browser Audio Input Alternative (One-Click Recording)", expanded=False):
        st.caption("Record speech directly through your browser if host microphone access is constrained:")
        browser_audio = st.audio_input("Record Voice", key="browser_mic_input")
        if browser_audio is not None:
            # Process browser recorded audio into live session
            try:
                b_bytes = browser_audio.getvalue()
                b_samples, b_meta = load_audio(b_bytes, filename="browser_mic.wav")
                msess.capture.feed_samples(b_samples)
                # Process chunks immediately
                while True:
                    ev = msess.step()
                    if ev is None:
                        break
                st.success(f"Processed {b_meta.formatted_duration} audio from browser recording!")
            except Exception as e:
                st.error(f"Failed to decode browser audio: {e}")

    # LIVE METRICS BAR
    st.markdown("### ⚡ Real-Time Live Performance Metrics")
    kpi_m1, kpi_m2, kpi_m3, kpi_m4 = st.columns(4)
    kpi_m1.metric("⏱️ Live Elapsed", f"{elapsed_live:.1f} s")
    kpi_m2.metric("📦 Chunks Processed", chunks_done)
    kpi_m3.metric("📝 Tokens Processed", len(msess.reconstructed_tokens))
    latest_asr = msess.asr_latencies_ms[-1] if msess.asr_latencies_ms else 0.0
    latest_e2e = msess.e2e_latencies_ms[-1] if msess.e2e_latencies_ms else 0.0
    kpi_m4.metric("⚡ Latency (ASR / E2E)", f"{latest_asr:.0f} ms / {latest_e2e:.0f} ms")

    # RECURRING KEY PHRASE TICKER AREA
    st.markdown("### 🏷️ Recurring Key Phrase Ticker")
    mic_ranked_phrases = msess.current_rankings.get("ranked_phrases", [])

    if mic_ranked_phrases:
        mic_ticker_chips = []
        for p in mic_ranked_phrases[:8]:
            p_name = p.get("phrase", "").upper()
            p_cat = p.get("quality_category", "LOW")
            p_score = p.get("quality_score", 0.0)
            p_freq = p.get("frequency", 0)

            if p_cat == "HIGH":
                style = "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
                badge = "★ HIGH"
            elif p_cat == "MEDIUM":
                style = "background: #e0f2fe; color: #075985; border: 1px solid #38bdf8;"
                badge = "▲ MED"
            else:
                style = "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;"
                badge = "• LOW"

            c_html = f"""
            <span style="{style} padding: 4px 10px; border-radius: 16px; margin: 0 4px; display: inline-block; font-size: 0.85rem;">
                <b>{p_name}</b> <span style="opacity: 0.8; font-size: 0.75rem;">({p_freq}x | {badge} | {p_score:.2f})</span>
            </span>
            """
            mic_ticker_chips.append(c_html.strip())

        render_html(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; line-height: 2.2;">
                {" ".join(mic_ticker_chips)}
            </div>
            """
        )
    else:
        st.info("Waiting for speech... (Press ▶️ Start Microphone and speak into your device)")

    st.write("")

    # MAIN TWO-COLUMN DASHBOARD DISPLAY FOR LIVE MICROPHONE
    mic_col_left, mic_col_right = st.columns([1.1, 0.9])

    with mic_col_left:
        st.subheader("📺 Live Subtitle Stream")

        # Active Latest Subtitle Card
        latest_mic_seg = msess.emitted_subtitles[-1] if msess.emitted_subtitles else None
        if latest_mic_seg:
            m_seg_text = latest_mic_seg.get("text", "")
            m_hl_html = highlight_phrases_in_text(m_seg_text, mic_ranked_phrases)
            sm, ss = int(latest_mic_seg.get("start_time", 0.0) // 60), latest_mic_seg.get("start_time", 0.0) % 60
            em, es = int(latest_mic_seg.get("end_time", 0.0) // 60), latest_mic_seg.get("end_time", 0.0) % 60
            m_time_str = f"[{sm:02d}:{ss:04.1f} → {em:02d}:{es:04.1f}]"

            render_html(
                f"""
                <div style="background: #0f172a; color: #ffffff; padding: 20px 24px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.2); margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="color: #10b981; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.5px;">● LIVE CAPTION</span>
                        <span style="color: #94a3b8; font-family: monospace; font-size: 0.85rem; background: #1e293b; padding: 2px 8px; border-radius: 4px;">{m_time_str}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 500; line-height: 1.6; color: #f8fafc; margin-bottom: 14px;">
                        {m_hl_html}
                    </div>
                    <div style="display: flex; gap: 16px; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 10px;">
                        <span>Words: <b style="color: #f1f5f9;">{latest_mic_seg.get('word_count', len(m_seg_text.split()))}</b></span>
                        <span>Chars: <b style="color: #f1f5f9;">{latest_mic_seg.get('char_count', len(m_seg_text))}</b></span>
                        <span>Boundary: <b style="color: #10b981;">{latest_mic_seg.get('reason', 'N/A')}</b></span>
                    </div>
                </div>
                """
            )
        else:
            st.info("Live subtitles will appear here as speech is captured and processed.")

        # Subtitle History Timeline
        with st.expander("📜 Subtitle Timeline Log", expanded=True):
            if msess.emitted_subtitles:
                for idx, seg in enumerate(reversed(msess.emitted_subtitles), 1):
                    seg_num = len(msess.emitted_subtitles) - idx + 1
                    s_txt = seg.get("text", "")
                    s_html = highlight_phrases_in_text(s_txt, mic_ranked_phrases)
                    st_sec, et_sec = seg.get("start_time", 0.0), seg.get("end_time", 0.0)
                    sm, ss = int(st_sec // 60), st_sec % 60
                    em, es = int(et_sec // 60), et_sec % 60

                    render_html(
                        f"""
                        <div style="border-left: 3px solid #10b981; padding: 10px 14px; margin-bottom: 10px; background: #ffffff; border-radius: 0 8px 8px 0; border: 1px solid #e2e8f0; border-left-width: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">
                                <b>#{seg_num}</b> &nbsp; <code style="color: #475569;">{sm:02d}:{ss:04.1f} → {em:02d}:{es:04.1f}</code> · {seg.get('word_count', len(s_txt.split()))} words · <i>{seg.get('reason', 'boundary')}</i>
                            </div>
                            <div style="font-size: 1rem; color: #0f172a; line-height: 1.55;">
                                {s_html}
                            </div>
                        </div>
                        """
                    )
            else:
                st.caption("No finalized subtitles yet.")

    with mic_col_right:
        st.subheader("📊 Phrase Intelligence & Analytics")

        # 2x2 Structured KPI Grid
        m_kpi1, m_kpi2 = st.columns(2)
        m_kpi1.metric("Subtitles Emitted", len(msess.emitted_subtitles))
        m_kpi2.metric("Ranked Phrases", len(mic_ranked_phrases))

        # Ranked Phrases Table
        st.markdown("#### Ranked Phrases")
        if mic_ranked_phrases:
            mic_cat_filt = st.selectbox("Category Filter", ["ALL", "HIGH", "MEDIUM", "LOW"], index=0, key="mic_cat_filt")
            filtered_phrases = mic_ranked_phrases
            if mic_cat_filt != "ALL":
                filtered_phrases = [p for p in mic_ranked_phrases if p.get("quality_category") == mic_cat_filt]

            if filtered_phrases:
                df_phrases = pd.DataFrame(
                    [
                        {
                            "Rank": idx + 1,
                            "Phrase": p.get("phrase", ""),
                            "N-gram": f"{p.get('ngram_size')}-gram",
                            "Frequency": p.get("frequency", 0),
                            "Score": f"{p.get('quality_score', 0.0):.2f}",
                            "Quality": p.get("quality_category", "LOW"),
                            "POS Pattern": p.get("pos_pattern", ""),
                        }
                        for idx, p in enumerate(filtered_phrases[:15])
                    ]
                )
                st.dataframe(df_phrases, hide_index=True)
            else:
                st.info(f"No phrases found in category '{mic_cat_filt}'.")
        else:
            st.caption("No candidate phrases ranked yet.")

        # Reconstructed Transcript
        st.markdown("---")
        with st.expander("📝 Reconstructed Live Transcript", expanded=False):
            st.write(msess.get_summary().get("full_transcript") or "*(Transcript appears as speech is recognized)*")

        # Session Summary (if stopped)
        if msess.status == "STOPPED":
            st.markdown("---")
            st.markdown("#### 🏁 Final Live Session Summary")
            summary = msess.get_summary()
            s_c1, s_c2 = st.columns(2)
            s_c1.write(f"**Live Duration:** {summary['live_duration_seconds']}s")
            s_c1.write(f"**Audio Processed:** {summary['audio_duration_seconds']}s")
            s_c1.write(f"**Chunks:** {summary['chunks_processed']}")
            s_c1.write(f"**Subtitles:** {summary['subtitles_generated']}")

            s_c2.write(f"**Tokens:** {summary['tokens_processed']}")
            s_c2.write(f"**Avg ASR Latency:** {summary['average_asr_latency_ms']} ms")
            s_c2.write(f"**Avg E2E Latency:** {summary['average_e2e_latency_ms']} ms")
            s_c2.write(f"**ASR Model:** {summary['asr_model_size']}")

        # Export Controls
        st.markdown("#### 💾 Export Microphone Session")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        txt_data = msess.export_txt()
        json_data = msess.export_json()
        csv_data = msess.export_csv()

        exp_col1.download_button(
            "📥 Transcript (TXT)",
            data=txt_data,
            file_name="live_mic_transcript.txt",
            mime="text/plain",
            key="mic_export_txt",
        )
        exp_col2.download_button(
            "📥 Session (JSON)",
            data=json_data,
            file_name="live_mic_session.json",
            mime="application/json",
            key="mic_export_json",
        )
        exp_col3.download_button(
            "📥 Phrases (CSV)",
            data=csv_data if csv_data else "Rank,Phrase,NgramSize,Frequency,Score,Quality,POS",
            file_name="live_mic_phrases.csv",
            mime="text/csv",
            key="mic_export_csv",
        )

    # Live streaming loop for Mode B
    if st.session_state.mic_is_running:
        step_ev = msess.step()
        if step_ev is not None:
            # Chunk was available and processed
            st.rerun()
        else:
            # Buffer still accumulating, wait briefly then rerun to poll buffer
            time.sleep(0.2)
            st.rerun()


# ==============================================================================
# ROUTE 3: MODE C — TEXT STREAMING MODE (PHASE 5 REGRESSION)
# ==============================================================================
else:
    sess: StreamingSimulationSession = st.session_state.session

    col_b1, col_b2, col_b3, col_b4, col_status = st.columns([1.2, 1, 1, 1, 2.5])

    with col_b1:
        start_btn = st.button("▶️ Start Text", type="primary", key="text_start_btn")
    with col_b2:
        pause_btn = st.button("⏸️ Pause", key="text_pause_btn")
    with col_b3:
        resume_btn = st.button("⏯️ Resume", key="text_resume_btn")
    with col_b4:
        reset_btn = st.button("🔄 Reset", key="text_reset_btn")

    if reset_btn:
        st.session_state.is_running = False
        sess.reset()
        st.rerun()

    if pause_btn:
        st.session_state.is_running = False
        sess.status = "PAUSED"
        st.rerun()

    if resume_btn:
        if sess.status in ("PAUSED", "PROCESSING") and sess.chunk_index < sess.total_chunks:
            st.session_state.is_running = True
        elif sess.status == "READY":
            st.session_state.is_running = True
        st.rerun()

    if start_btn:
        sess.reset()
        st.session_state.is_running = True
        st.rerun()

    curr_status = sess.status
    status_color = status_colors.get(curr_status, "#64748b")

    with col_status:
        progress_val = (
            round((sess.chunk_index / max(1, sess.total_chunks)) * 100, 1)
            if sess.total_chunks > 0
            else 0.0
        )
        render_html(
            f"""
            <div style="text-align: right; padding-top: 5px;">
                <span style="background-color: {status_color}; color: white; padding: 5px 12px; border-radius: 12px; font-weight: bold; font-size: 0.85rem;">
                    ● {curr_status}
                </span>
                <span style="margin-left: 10px; font-size: 0.85rem; color: #475569;">
                    Tokens: <b>{sess.tokens_processed} / {sess.total_tokens}</b> ({progress_val}%) | Chunk {sess.chunk_index}/{sess.total_chunks}
                </span>
            </div>
            """
        )

    st.progress(progress_val / 100.0)

    # RECURRING KEY PHRASE TICKER AREA
    st.markdown("### 🏷️ Recurring Key Phrase Ticker")
    ranked_phrases = sess.current_rankings.get("ranked_phrases", [])

    if ranked_phrases:
        ticker_chips = []
        for p in ranked_phrases[:8]:
            phrase_name = p.get("phrase", "").upper()
            category = p.get("quality_category", "LOW")
            score = p.get("quality_score", 0.0)
            freq = p.get("frequency", 0)

            if category == "HIGH":
                style = "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
                badge = "★ HIGH"
            elif category == "MEDIUM":
                style = "background: #e0f2fe; color: #075985; border: 1px solid #38bdf8;"
                badge = "▲ MED"
            else:
                style = "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;"
                badge = "• LOW"

            chip_html = f"""
            <span style="{style} padding: 4px 10px; border-radius: 16px; margin: 0 4px; display: inline-block; font-size: 0.85rem;">
                <b>{phrase_name}</b> <span style="opacity: 0.8; font-size: 0.75rem;">({freq}x | {badge} | {score:.2f})</span>
            </span>
            """
            ticker_chips.append(chip_html.strip())

        render_html(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; line-height: 2.2;">
                {" ".join(ticker_chips)}
            </div>
            """
        )
    else:
        st.info("Waiting for recurring phrases... (Start streaming simulation to discover phrases)")

    st.write("")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("📺 Streaming Subtitle Output")

        latest_seg = sess.emitted_segments[-1] if sess.emitted_segments else None
        if latest_seg:
            seg_text = latest_seg.get("text", "")
            hl_html = highlight_phrases_in_text(seg_text, ranked_phrases)

            render_html(
                f"""
                <div style="background: #0f172a; color: #ffffff; padding: 20px 24px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.2); margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="color: #38bdf8; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.5px;">● ACTIVE SUBTITLE FRAME</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; background: #1e293b; padding: 2px 8px; border-radius: 4px;">Segment #{latest_seg.get('segment_id', len(sess.emitted_segments))}</span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 500; line-height: 1.6; color: #f8fafc; margin-bottom: 14px;">
                        {hl_html}
                    </div>
                    <div style="display: flex; gap: 16px; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 10px;">
                        <span>Words: <b style="color: #f1f5f9;">{latest_seg.get('word_count', len(seg_text.split()))}</b></span>
                        <span>Chars: <b style="color: #f1f5f9;">{latest_seg.get('char_count', len(seg_text))}</b></span>
                        <span>Reason: <b style="color: #38bdf8;">{latest_seg.get('reason', 'N/A')}</b></span>
                    </div>
                </div>
                """
            )
        else:
            st.info("Subtitles will appear here progressively as streaming chunks arrive.")

        with st.expander("📜 Subtitle History Timeline", expanded=True):
            if sess.emitted_segments:
                for idx, seg in enumerate(reversed(sess.emitted_segments), 1):
                    seg_num = len(sess.emitted_segments) - idx + 1
                    s_txt = seg.get("text", "")
                    s_html = highlight_phrases_in_text(s_txt, ranked_phrases)

                    render_html(
                        f"""
                        <div style="border-left: 3px solid #3b82f6; padding: 10px 14px; margin-bottom: 10px; background: #ffffff; border-radius: 0 8px 8px 0; border: 1px solid #e2e8f0; border-left-width: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">
                                <b>#{seg_num}</b> · {seg.get('word_count', len(s_txt.split()))} words · <i>{seg.get('reason', 'boundary')}</i>
                            </div>
                            <div style="font-size: 1rem; color: #0f172a; line-height: 1.55;">
                                {s_html}
                            </div>
                        </div>
                        """
                    )
            else:
                st.caption("No finalized subtitles yet.")

    with col_right:
        st.subheader("📊 Phrase Intelligence & Analytics")

        # 2x2 Structured KPI Grid
        t_kpi1, t_kpi2 = st.columns(2)
        t_kpi1.metric("Processed Tokens", sess.tokens_processed)
        t_kpi2.metric("Subtitles Emitted", len(sess.emitted_segments))

        t_kpi3, t_kpi4 = st.columns(2)
        t_kpi3.metric("Ranked Phrases", len(ranked_phrases))
        avg_lat = np.mean(sess.latencies_ms) if sess.latencies_ms else 0.0
        t_kpi4.metric("Avg Latency", f"{avg_lat:.1f} ms")

        st.markdown("#### Ranked Collocations")
        if ranked_phrases:
            cat_filter = st.selectbox("Filter Category", ["ALL", "HIGH", "MEDIUM", "LOW"], index=0, key="text_cat_filt")
            filtered_phrases = ranked_phrases
            if cat_filter != "ALL":
                filtered_phrases = [p for p in ranked_phrases if p.get("quality_category") == cat_filter]

            if filtered_phrases:
                df_phrases = pd.DataFrame(
                    [
                        {
                            "Rank": idx + 1,
                            "Phrase": p.get("phrase", ""),
                            "N-gram": f"{p.get('ngram_size')}-gram",
                            "Frequency": p.get("frequency", 0),
                            "Score": f"{p.get('quality_score', 0.0):.2f}",
                            "Quality": p.get("quality_category", "LOW"),
                            "POS Pattern": p.get("pos_pattern", ""),
                        }
                        for idx, p in enumerate(filtered_phrases[:15])
                    ]
                )
                st.dataframe(df_phrases, hide_index=True)
            else:
                st.info(f"No phrases found in category '{cat_filter}'.")
        else:
            st.caption("No candidate phrases ranked yet.")

        st.markdown("---")
        st.markdown("#### 💾 Export Artifacts")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        exp_data = sess.generate_exports()

        exp_col1.download_button(
            "📥 Subtitles (JSON)",
            data=exp_data["segments_json"],
            file_name="subtitle_segments.json",
            mime="application/json",
            key="text_export_subtitles",
        )
        exp_col2.download_button(
            "📥 Phrases (CSV)",
            data=exp_data["phrases_csv"],
            file_name="ranked_phrases.csv",
            mime="text/csv",
            key="text_export_phrases",
        )
        exp_col3.download_button(
            "📥 Summary (JSON)",
            data=exp_data["summary_json"],
            file_name="session_summary.json",
            mime="application/json",
            key="text_export_summary",
        )

    # Simulation interval loop for Mode C
    if st.session_state.is_running:
        if sess.chunk_index < sess.total_chunks:
            time.sleep(st.session_state.delay_val)
            step_event = sess.step()
            st.rerun()
        else:
            st.session_state.is_running = False
            sess.status = "COMPLETED"
            st.rerun()
