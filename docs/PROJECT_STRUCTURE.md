# REPOSITORY & PROJECT STRUCTURE

This document details the directory layout and file organization of the **Real-Time Subtitle Segmenter and N-gram Phrase Extractor** project under `F:\Real-Time Subtitle Segmenter`.

---

## 1. Directory Tree Overview

```text
F:\Real-Time Subtitle Segmenter\
├── data/                       # Input sample transcripts and speech audio
│   ├── input/                  # Reference test WAV files and sample texts
│   └── output/                 # Generated export files (TXT, JSON, CSV)
├── docs/                       # Official project documentation (Phase 11)
│   ├── PROJECT_OVERVIEW.md     # High-level goals, features, and stack
│   ├── ARCHITECTURE.md         # Detailed system design, ASCII diagram, and data flow
│   ├── METHODOLOGY.md          # Technical algorithmic and linguistic methodology
│   ├── EVALUATION.md           # Empirical benchmark results and analysis
│   ├── LIMITATIONS.md          # Technical trade-offs, constraints, and boundaries
│   ├── DEMO_GUIDE.md           # Step-by-step demonstration walkthrough
│   ├── VIVA_QA.md              # 32 technical defense questions and answers
│   └── PROJECT_STRUCTURE.md    # This directory and file guide
├── evaluation/                 # Phase 10 Benchmark suite and artifacts
│   ├── datasets/               # Controlled test audio files (single, multi, short)
│   ├── references/             # Ground truth reference transcripts, turns, and phrases
│   ├── reports/                # phase10_report.md generated benchmark report
│   ├── results/                # benchmark_results.json machine-readable results
│   └── scripts/                # Evaluation runners (run_benchmarks.py, metrics.py)
├── models/                     # Isolated local faster-whisper model weights (F: drive)
├── nltk_data/                  # Isolated NLTK tokenizers and POS tagger models (F: drive)
├── src/                        # Core Python source codebase
│   ├── audio/                  # Audio I/O, chunking, ASR, diarization, and streaming pipeline
│   │   ├── asr.py              # Local faster-whisper model encapsulation (CPU INT8)
│   │   ├── chunker.py          # AudioChunker splitting audio arrays into discrete chunks
│   │   ├── diarization.py      # AcousticEnergyDiarizer and temporal alignment
│   │   ├── live_session.py     # Live microphone session state (Phase 7 compatibility)
│   │   ├── loader.py           # PyAV audio loader, validator, and resampler
│   │   ├── microphone.py       # Sounddevice microphone capture (Phase 7 compatibility)
│   │   ├── overlap.py          # Conservative text overlap remover for live chunking
│   │   └── pipeline.py         # AudioStreamingSession orchestrating static audio workflow
│   ├── boundary_scorer.py      # Multi-signal subtitle boundary scoring engine
│   ├── collocation_scorer.py   # Statistical collocation calculator
│   ├── improved_segmenter.py   # Syntactic streaming subtitle segmenter
│   ├── ngram_extractor.py      # Exhaustive bigram and trigram extraction
│   ├── phrase_filter.py        # Stopword pruning and linguistic heuristics
│   ├── phrase_protection.py    # PhraseProtector preventing cuts inside collocations
│   ├── phrase_ranker.py        # Phrase Quality Score, tiers (HIGH/MED/LOW), hierarchy
│   ├── streaming_buffer.py     # FIFO token buffer for incremental streaming
│   ├── streaming_phrase_extractor.py # Incremental session phrase extraction
│   ├── streaming_segmenter.py  # Base streaming segmenter configuration
│   ├── text_processor.py       # Text cleaner and whitespace standardizer
│   ├── tokenizer.py            # Word tokenizer isolating punctuation
│   ├── ui_app.py               # Official Streamlit production web dashboard
│   └── ui_session.py           # Text streaming simulation session manager
├── tests/                      # Automated test suite (203/203 passing tests)
│   ├── test_ngram.py           # Phase 1 N-gram unit tests
│   ├── test_phrase_filter.py   # Phase 1 linguistic filter tests
│   ├── test_segmenter.py       # Phase 1 deterministic segmenter tests
│   ├── test_tokenizer.py       # Phase 1 normalization & tokenization tests
│   ├── test_streaming.py       # Phase 2 streaming buffer & simulation tests
│   ├── test_phase3_segmenter.py# Phase 3 boundary scoring & phrase protection tests
│   ├── test_phase4_phrase.py   # Phase 4 PMI, Tri-PMI & ranking tests
│   ├── test_phase5_ui.py       # Phase 5 Streamlit UI integration tests
│   ├── test_phase6_asr.py      # Phase 6 Audio loader, chunker & local ASR tests
│   ├── test_phase7_microphone.py # Phase 7 Live microphone capture tests
│   ├── test_phase8_diarization.py # Phase 8 Speaker diarization & alignment tests
│   ├── test_phase9_stability.py# Phase 9 Stability, error recovery & exports tests
│   └── test_phase10_evaluation.py # Phase 10 Metrics & evaluation tests
├── venv/                       # Isolated Python 3.12 virtual environment (F: drive)
├── requirements.txt            # Minimal verified production dependencies
├── run_phase5.py               # Primary application launcher script (Streamlit on port 8501)
└── README.md                   # Primary repository landing page
```

---

## 2. Directory Purpose Breakdown

### `src/` — Production Source Code
Contains the core business logic, organized into focused modules.
- **`src/audio/`**: Encapsulates all audio decoding, chunking, acoustic models, and pipeline orchestration. Completely isolated from the UI layer.
- **NLP Modules**: `boundary_scorer.py`, `improved_segmenter.py`, `phrase_ranker.py`, `ngram_extractor.py`, and `phrase_filter.py` handle all computational linguistics deterministically without cloud dependencies.
- **`src/ui_app.py`**: The official permanent frontend application written in Streamlit. Contains layout, reactive widgets, and event dispatchers.

### `evaluation/` — Benchmark Suite & Datasets
Introduced in Phase 10 to establish empirical credibility.
- **`datasets/`**: Stores static test recordings (`single_speaker.wav`, `multi_speaker.wav`, `short_speech.wav`).
- **`references/`**: Stores human-verified reference text, speaker turn CSVs, and expected collocations.
- **`scripts/`**: Houses `run_benchmarks.py` and `metrics.py` for automated evaluation.
- **`results/` & `reports/`**: Machine-readable JSON payloads and markdown evaluation summaries.

### `tests/` — Automated Regression Suite
Contains 13 comprehensive test modules covering all 10 completed development phases.
- Executed via `pytest tests/ -v`.
- **100% pass rate: 203 out of 203 tests passing.**
- Enforces strict backward compatibility and protects against behavioral regressions.

### `models/` & `nltk_data/` — Isolated Local Weights
- All pre-downloaded `faster-whisper` CTranslate2 model checkpoints and NLTK taggers are contained strictly within these directories on the `F:` drive.
- No files are written to `C:\Users\<User>\.cache` or roaming app data.
