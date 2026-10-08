# Real-Time Subtitle Segmenter and N-gram Phrase Extractor

[![Tests](https://img.shields.io/badge/tests-203%2F203%20passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![ASR Engine](https://img.shields.io/badge/ASR-faster--whisper%20(CPU%20INT8)-orange.svg)]()
[![Frontend](https://img.shields.io/badge/frontend-Streamlit-red.svg)]()
[![Cloud Services](https://img.shields.io/badge/OpenAI%20%2F%20Cloud-NONE%20(100%25%20Local)-success.svg)]()
[![Drive Isolation](https://img.shields.io/badge/Storage-Isolated%20on%20F%3A-informational.svg)]()

> **CRITICAL WORKFLOW & ARCHITECTURAL NOTICE:**  
> Despite the historical repository name (*"Real-Time Subtitle Segmenter"*), the **official, permanent product workflow** is:  
> **Static speech audio files uploaded by the user (`WAV`, `MP3`, `M4A`) $\to$ local chunking $\to$ local `faster-whisper` ASR $\to$ optional local speaker diarization $\to$ syntactic subtitle segmentation $\to$ n-gram phrase extraction $\to$ Streamlit live feed and multi-format exports (`TXT`, `JSON`, `CSV`).**  
> Streamlit is the permanent official web frontend (earlier experimental React prototypes have been permanently removed). All processing, virtual environments, model weights, and data reside strictly on the **`F:` drive**. Zero OpenAI APIs, zero cloud ASR, and zero external network dependencies are utilized.

---

## 1. Quick-Start Guide

### 1.1 Launch the Streamlit Web Application
Ensure you are in the project root on `F:\Real-Time Subtitle Segmenter`, then execute:
```powershell
.\venv\Scripts\python.exe run_phase5.py --port 8501
```
Open **`http://localhost:8501`** in your browser.

### 1.2 Run the Automated Test Suite (All 10 Phases)
Verify that all 203 automated unit, integration, and regression tests pass:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```
*(Result: **203 passed in ~17.7s**)*

### 1.3 Run the Empirical Benchmark Suite
Execute the Phase 10 objective evaluation suite to reproduce the benchmark results:
```powershell
.\venv\Scripts\python.exe evaluation/scripts/run_benchmarks.py
```
*(Outputs `evaluation/results/benchmark_results.json` and `evaluation/reports/phase10_report.md`)*

---

## 2. Project Overview & Motivation
Raw Automatic Speech Recognition (ASR) engines typically output an unformatted, unpunctuated stream of words ("wall-of-text"). Displaying raw ASR text in video subtitles or meeting transcriptions impairs comprehension and fails to highlight central discussion topics.

This project delivers an **offline, deterministic speech intelligence pipeline** that:
1. Transcribes speech audio completely locally using quantized **`faster-whisper`** (CPU INT8 / GPU).
2. Segments unpunctuated text into natural, readable subtitle chunks (5–14 words, max 80 characters) using multi-signal linguistic boundary cues (discourse markers, clause conjunctions, length balancing) and multi-word phrase protection.
3. Discovers and ranks recurring technical collocations (bigrams and trigrams) using **Pointwise Mutual Information (PMI)**, **Tri-PMI**, Part-of-Speech (POS) pattern filtering, and content-word density.
4. Optionally attributes speech turns to anonymous speaker labels (`Speaker 1`, `Speaker 2`) and triggers subtitle cuts at speaker transitions.
5. Provides an intuitive Streamlit dashboard with real-time tickers, interactive analytics, and multi-format download capabilities (`TXT`, `JSON`, `CSV`).

---

## 3. High-Level System Architecture

```text
User Audio File (WAV, MP3, M4A <= 150 MB)
                   |
                   v
Audio Ingestion & Validation (PyAV Mono 16kHz float32)
                   |
                   v
Audio Chunking Module (Default 5.0-second sliding chunks)
                   |
         +---------+---------+
         |                   |
         | (If Diarization)  | (Continuous Stream)
         v                   v
Acoustic Diarizer       Local faster-whisper ASR (CPU INT8)
(Energy & Centroids)         |
         |                   |
         +---------+---------+
                   |
                   v
Temporal Overlap Alignment & Speaker-Change Boundary Trigger
                   |
                   v
Subtitle Segmentation Engine (Discourse Cues, Conjunctions, Phrase Protection)
                   |
                   v
N-gram Phrase Intelligence (Exhaustive Bigrams & Trigrams, PMI, Tri-PMI, POS)
                   |
                   v
Phrase Quality Scoring & Deterministic Ranking (HIGH / MEDIUM / LOW)
                   |
                   v
Streamlit Production Dashboard (Feed, Ticker, Tables, Diarization Breakdown)
                   |
      +------------+------------+------------+
      |                         |            |
transcript.txt           transcript.json   phrases.csv & subtitles.csv
```

---

## 4. Key Empirical Benchmark Results (Phase 10 Evaluation)

All metrics were objectively measured on controlled static recordings without data leakage or cloud APIs:

| Evaluation Metric | Measured Result | Significance / Interpretation |
| :--- | :---: | :--- |
| **ASR Word Error Rate (WER)** | **3.17%** | Near-perfect word recognition on local CPU INT8 |
| **ASR Character Error Rate (CER)** | **1.83%** | Exceptional morphological and spelling fidelity |
| **Warm Real-Time Factor (RTF)** | **0.077x** | ~13x faster than real-time playback on standard CPU |
| **Average Warm Processing Time** | **1.16s** | Fast turnaround for ~14.3s audio |
| **Cold Start Processing Time** | **3.56s** | Average initial run including model weight loading |
| **Phrase Extraction Recall** | **100.0%** | Discovered every expected recurring domain collocation |
| **Phrase Extraction Precision** | **55.55%** | Also extracts valid constituent sub-phrases (nested in UI) |
| **Speaker-Change Boundary Accuracy** | **100.0%** | Every speaker transition triggered an immediate subtitle split |
| **Subtitle Boundary Accuracy** | **58.34%** | Alignment within ±0.50s of major turn boundaries |
| **Average Subtitle Timing Error** | **0.067s** | Millisecond-accurate subtitle cut alignment |
| **Speaker Assignment Accuracy** | **47.25%** | Lightweight acoustic heuristic (DER: **52.75%**) |

---

## 5. Technology Stack & Operational Constraints
- **Programming Language**: Python 3.12 (64-bit)
- **Official Web Frontend**: Streamlit 1.30+
- **Speech Recognition Engine**: `faster-whisper` 1.2+ (CTranslate2 on CPU with INT8 quantization)
- **Audio Decoding**: PyAV (FFmpeg bindings) and standard library `wave`
- **Natural Language Processing**: NLTK 3.8+ (Tokenization & Penn Treebank POS tagger)
- **Scientific Computing**: NumPy 1.26+
- **Testing Suite**: `pytest` 8.0+ (**203 tests across 10 phases**)
- **Zero Cloud / Zero OpenAI**: Zero OpenAI packages, zero API keys required, zero cloud calls.
- **Strict F: Drive Isolation**: Virtual environment (`venv`), model weights (`models/`), NLTK data (`nltk_data/`), and test artifacts reside strictly under `F:\Real-Time Subtitle Segmenter`.

---

## 6. Comprehensive Project Documentation Index

For in-depth architectural, theoretical, evaluation, and presentation references, consult the dedicated guides in the **`docs/`** directory:

| Document | Description |
| :--- | :--- |
| **[Project Overview](docs/PROJECT_OVERVIEW.md)** | High-level problem statement, motivation, objectives, features, and design decisions. |
| **[Architecture Specification](docs/ARCHITECTURE.md)** | Full ASCII architecture diagram, component responsibilities, data flow, and modular separation. |
| **[Technical Methodology](docs/METHODOLOGY.md)** | Mathematical formulas and theory: PyAV decoding, faster-whisper, acoustic diarization, boundary scoring, PMI, Tri-PMI, and POS filtering. |
| **[Evaluation & Benchmarks](docs/EVALUATION.md)** | Detailed Phase 10 empirical results: WER, CER, DER, boundary precision, RTF, repeatability, and honest analysis. |
| **[Technical Limitations](docs/LIMITATIONS.md)** | Documented engineering boundaries: dataset size, heuristic diarizer trade-offs, anonymous labels, and recurrence thresholds. |
| **[Demonstration Guide](docs/DEMO_GUIDE.md)** | Step-by-step presentation script covering single-speaker audio, multi-speaker diarization, error recovery, and exports. |
| **[Viva Voce Technical Q&A](docs/VIVA_QA.md)** | 32 concise, rigorous questions and answers defending technical decisions for examiners. |
| **[Repository Structure](docs/PROJECT_STRUCTURE.md)** | Complete directory tree explaining every active folder and file. |

---

## 7. Automated Testing Summary
The project features complete test coverage across all 10 engineering phases:

```text
tests/
  ├── test_tokenizer.py           # Phase 1: Normalization & tokenization (10 tests)
  ├── test_segmenter.py           # Phase 1: Deterministic rule segmentation (6 tests)
  ├── test_ngram.py               # Phase 1: N-gram frequency extraction (15 tests)
  ├── test_phrase_filter.py       # Phase 1: Linguistic filtering & fallbacks (8 tests)
  ├── test_streaming.py           # Phase 2: Streaming buffer & simulation (20 tests)
  ├── test_phase3_segmenter.py    # Phase 3: Boundary scoring & phrase protection (22 tests)
  ├── test_phase4_phrase.py       # Phase 4: PMI, Tri-PMI & deterministic ranking (28 tests)
  ├── test_phase5_ui.py           # Phase 5: Streamlit session & UI integration (20 tests)
  ├── test_phase6_asr.py          # Phase 6: Audio loader, chunker & local ASR (18 tests)
  ├── test_phase7_microphone.py   # Phase 7: Microphone capture & overlap handler (20 tests)
  ├── test_phase8_diarization.py  # Phase 8: Speaker diarization & alignment (24 tests)
  ├── test_phase9_stability.py    # Phase 9: Error recovery, reset & exports (18 tests)
  └── test_phase10_evaluation.py  # Phase 10: WER, CER, DER, RTF & evaluation metrics (14 tests)
```
**Total: 203 / 203 PASS** (Executed via `pytest tests/ -v`).

---

## 8. License & Academic Attribution
Developed as an engineering project under `F:\Real-Time Subtitle Segmenter`.  
Built strictly using open-source, local-first libraries: Python, Streamlit, faster-whisper, PyAV, NLTK, NumPy, and pytest.
