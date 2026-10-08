# PROJECT OVERVIEW

## 1. Project Title
**Real-Time Subtitle Segmenter and N-gram Phrase Extractor**  
*Offline Speech-to-Subtitle Pipeline with Collocation Intelligence*

---

## 2. Problem Statement
Raw Automatic Speech Recognition (ASR) outputs typically produce continuous, unpunctuated streams of words ("wall-of-text"). In video captioning, meeting transcription, and broadcast production, displaying raw, unsegmented ASR output degrades readability and user comprehension. Furthermore, viewers and analysts lack instant visibility into recurring key topics, domain concepts, or speaker turn changes without manually reading entire transcripts or relying on expensive, privacy-invasive cloud APIs.

Existing commercial solutions predominantly rely on proprietary cloud platforms (e.g., OpenAI Whisper API, Google Speech-to-Text), which introduce:
- Data privacy vulnerabilities (uploading proprietary or sensitive audio to third-party cloud servers).
- Recurring API subscription costs and latency bottlenecks.
- Black-box hallucination and non-deterministic text generation when LLMs are used for phrase extraction.

---

## 3. Motivation
This project addresses the need for an **entirely local, offline-capable, and deterministic** speech intelligence system. By combining local acoustic speech recognition with statistical and linguistic Natural Language Processing (NLP), the system transforms static audio recordings into clean, syntactically coherent subtitle segments and extracts ranked recurring n-gram collocations without any cloud dependencies or external network requests.

---

## 4. Primary Objectives
1. **Local-First Audio Processing**: Ingest user-uploaded audio files (`WAV`, `MP3`, `M4A`) and transcribe them locally using quantized INT8 CPU models.
2. **Syntactically Coherent Subtitle Segmentation**: Break continuous unpunctuated text into human-readable subtitle chunks (5–14 words, max 80 characters) using multi-signal linguistic cues (discourse markers, clause conjunctions, length balancing) rather than arbitrary sentence splitters.
3. **Statistical & Grammatical Phrase Extraction**: Discover recurring multi-word collocations (bigrams and trigrams) using Pointwise Mutual Information (PMI), Tri-PMI, Part-of-Speech (POS) pattern filtering, and content-word density.
4. **Deterministic Explainable Phrase Ranking**: Score and rank key phrases into `HIGH`, `MEDIUM`, and `LOW` categories using a deterministic formula with consistent tie-breaking.
5. **Optional Speaker Diarization**: Provide an optional, lightweight offline diarizer that attributes speech turns to anonymous speaker labels (`Speaker 1`, `Speaker 2`) and triggers subtitle cuts at speaker transitions.
6. **Production-Ready Web Dashboard**: Deliver an intuitive, responsive Streamlit dashboard featuring live subtitle feeds, phrase tickers, analytics, and multiple export formats (`TXT`, `JSON`, `CSV`).
7. **Strict System Isolation**: Ensure all model caches, virtual environments, audio inputs, and generated artifacts reside strictly on the designated project drive (`F:\Real-Time Subtitle Segmenter`), with zero OpenAI usage and zero cloud dependencies.

---

## 5. Main Features
- **Static Audio File Upload**: Drag-and-drop support for WAV, MP3, and M4A audio formats up to 150 MB.
- **Local Speech Recognition**: Powered by `faster-whisper` running on CPU with INT8 quantization, achieving sub-4% Word Error Rate (WER) on standard speech.
- **Chunked Progressive Streaming Simulation**: Simulates real-time chunked ingestion so users observe progressive subtitle emissions and dynamic phrase ranking updates.
- **Multi-Signal Subtitle Boundaries**: Prevents awkward cuts inside meaningful multi-word phrases and balances subtitle readability.
- **Collocation Intelligence**: Surfaces recurring technical, policy, and domain phrases with statistical association scores.
- **Parent-Child Phrase Hierarchy**: Visualizes nested collocations (e.g., `"renewable energy"` within `"renewable energy transition"`).
- **Optional Local Speaker Diarization**: Detects speaker turns using frame-level acoustic energy and spectral centroid clustering, flushing subtitle buffers upon speaker changes.
- **Multi-Format Export Engine**: One-click download of human-readable TXT reports, machine-readable JSON session payloads, ranked phrases CSV, and timestamped subtitle CSV files.
- **Error Resilient & Validated**: Comprehensive input validation against empty, corrupted, or unsupported audio files with one-click session recovery.

---

## 6. Supported Input
| Parameter | Specification |
| :--- | :--- |
| **File Formats** | `WAV`, `MP3`, `M4A` |
| **Max File Size** | 150 MB (configurable via `MAX_AUDIO_SIZE_MB`) |
| **Audio Channels** | Mono or Stereo (automatically downmixed to mono) |
| **Target Sample Rate** | 16,000 Hz (automatically resampled during decoding) |
| **Target Quantization** | 16-bit PCM converted to normalized `float32` array $[-1.0, 1.0]$ |

---

## 7. High-Level Processing Pipeline
```text
Static Audio Upload (WAV, MP3, M4A)
        ↓
Audio Loading & Validation (PyAV / Mono 16kHz float32)
        ↓
Audio Chunking (Default 5.0-second sequential chunks)
        ↓
Local faster-whisper ASR (CPU INT8 / GPU)
        ↓
[Optional] Speaker Diarization & Temporal Alignment
        ↓
Subtitle Segmentation (Discourse markers, clause cues, phrase protection)
        ↓
N-gram Phrase Extraction (Exhaustive bigrams & trigrams)
        ↓
Statistical & POS Filtering (PMI, Tri-PMI, stopword penalties, content density)
        ↓
Phrase Quality Scoring & Deterministic Ranking (HIGH / MEDIUM / LOW)
        ↓
Streamlit Results Presentation & Multi-Format Exports (TXT / JSON / CSV)
```

---

## 8. Export Formats
1. **Formatted Transcript (`transcript.txt`)**: Human-readable report containing audio metadata, optional speaker timeline, finalized subtitle entries with timestamps, complete transcript, and top ranked key phrases.
2. **Machine-Readable Session (`transcript.json`)**: Complete structured JSON document containing audio metadata, ASR configuration, performance metrics (latency, RTF), diarization summary, subtitle segments array, phrase statistics, and ranked phrases.
3. **Collocations CSV (`phrases.csv`)**: Tabular export of ranked phrases containing columns `Rank`, `Phrase`, `NgramSize`, `Frequency`, `Score`, `Quality`, and `POS`.
4. **Subtitles CSV (`subtitles.csv`)**: Timestamped subtitle entries with columns `start`, `end`, `speaker`, and `text`.

---

## 9. Technology Stack
- **Core Language**: Python 3.12 (64-bit)
- **Frontend Dashboard**: Streamlit 1.30+
- **Speech Recognition Engine**: `faster-whisper` 1.2+ (powered by CTranslate2 on CPU with INT8 quantization)
- **Audio Decoding / Resampling**: PyAV (FFmpeg bindings) and Python standard library `wave`
- **Natural Language Processing**: NLTK 3.8+ (tokenization and POS tagging)
- **Scientific Computing**: NumPy 1.26+
- **Testing Framework**: `pytest` 8.0+ (203 automated unit, integration, and evaluation tests)
- **Model Storage & Cache Isolation**: Local `models/` directory on `F:` drive (`HF_HUB_CACHE`, `HF_HOME`)

---

## 10. Key Design Decisions
1. **Static Audio Upload as Primary Workflow**: Rather than requiring unstable browser microphone drivers or live WebRTC capture during evaluation, the primary product workflow focuses on static audio files uploaded by the user.
2. **Streamlit as Permanent Official Frontend**: Replaces earlier React web prototypes with a cohesive, Python-native Streamlit dashboard. React was formally deprecated and removed to eliminate Node.js dependencies and cross-origin state synchronization issues.
3. **Deterministic NLP over Generative LLMs**: Rather than querying probabilistic LLMs for phrase extraction, the system uses deterministic statistical metrics (PMI, Tri-PMI, frequency) and rule-based POS tagging. This ensures 100% reproducible results, zero API billing, zero hallucinations, and instantaneous sub-second execution.
4. **Zero OpenAI and Cloud Services**: Zero external cloud APIs (0 OpenAI packages, 0 imports, 0 network calls). All processing occurs on the user's host machine.
5. **Strict F: Drive Isolation**: All virtual environments, model weights, audio inputs, test artifacts, and caches are restricted to `F:\Real-Time Subtitle Segmenter`.

---

## 11. Current Project Status
- **Development Phases**: Phase 1 through Phase 11 complete.
- **Automated Test Suite**: **203 / 203 tests passing** (100% pass rate).
- **Benchmarking & Validation**:
  - ASR Mean Word Error Rate (WER): **3.17%**
  - ASR Mean Character Error Rate (CER): **1.83%**
  - Average Warm Processing Time: **1.16s** (for ~14.3s audio)
  - Real-Time Factor (RTF): **0.077x** (~13x faster than real-time playback on standard CPU).
- **Submission Readiness**: Fully documented, tested, and packaged for academic demonstration and technical defense.
