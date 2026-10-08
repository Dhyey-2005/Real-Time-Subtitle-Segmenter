# SYSTEM ARCHITECTURE

## 1. Architectural Overview
The **Real-Time Subtitle Segmenter and N-gram Phrase Extractor** employs a modular, pipeline-oriented architecture designed for deterministic execution, strict drive isolation, and sub-second local processing. The system bridges digital signal processing (audio decoding and chunking), local acoustic modeling (ASR and optional diarization), and statistical natural language processing (syntactic segmentation and collocation ranking) into a unified Streamlit dashboard.

---

## 2. End-to-End System Architecture Diagram

```text
+-------------------------------------------------------------------------------+
|                                  USER / BROWSER                               |
+-------------------------------------------------------------------------------+
                                       |
                                       | Uploads WAV / MP3 / M4A (Max 150 MB)
                                       v
+-------------------------------------------------------------------------------+
|                           STREAMLIT WEB DASHBOARD                             |
|                               (src/ui_app.py)                                 |
+-------------------------------------------------------------------------------+
                                       |
                                       | Audio Bytes / Path
                                       v
+-------------------------------------------------------------------------------+
|                       AUDIO INGESTION & VALIDATION MODULE                      |
|                              (src/audio/loader.py)                            |
|  - Extension & Size Bounds Check (WAV, MP3, M4A <= 150 MB)                    |
|  - PyAV Container Decoding                                                    |
|  - Mono Downmixing & 16 kHz Resampling                                        |
|  - Normalized float32 Audio Array [-1.0, 1.0]                                 |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                             AUDIO CHUNKING MODULE                             |
|                             (src/audio/chunker.py)                            |
|  - Splits audio into sequential AudioChunk objects (default: 5.0s)            |
|  - Preserves precise start/end timestamps and sample slices                   |
+-------------------------------------------------------------------------------+
                                       |
         +-----------------------------+-----------------------------+
         |                                                           |
         | (If Diarization Enabled)                                  | (Continuous Chunk Flow)
         v                                                           v
+-----------------------------------+       +-----------------------------------+
|    SPEAKER DIARIZATION MODULE     |       |          LOCAL ASR ENGINE         |
|   (src/audio/diarization.py)      |       |         (src/audio/asr.py)        |
| - AcousticEnergyDiarizer (NumPy)  |       | - faster-whisper (tiny/base/small)|
| - Frame Energy & Spectral Centroid|       | - CTranslate2 Engine on CPU INT8  |
| - K-Means Clustering              |       | - Zero OpenAI / Zero Cloud Calls  |
| - Speaker Intervals [t0, t1, Spk] |       | - ASRChunkResult (Text + Timings) |
+-----------------------------------+       +-----------------------------------+
         |                                                           |
         +-----------------------------+-----------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                     ALIGNMENT & BOUNDARY TRIGGER ENGINE                       |
|                          (src/audio/pipeline.py)                              |
|  - align_asr_with_diarization(): Maximum temporal overlap resolution         |
|  - Speaker transitions trigger immediate subtitle buffer flush cuts           |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                         SUBTITLE SEGMENTATION ENGINE                          |
|             (src/improved_segmenter.py, src/boundary_scorer.py)               |
|  - Streaming FIFO Token Buffer (src/streaming_buffer.py)                      |
|  - Multi-Signal Scoring: Discourse cues (+65), Conjunctions (+35), Length     |
|  - Phrase Protection (-85 penalty for splits inside domain phrases)          |
|  - Hard bounds enforced: 5–14 words, max 80 characters per subtitle frame     |
+-------------------------------------------------------------------------------+
                                       |
                                       | Emitted Subtitle Segments
                                       v
+-------------------------------------------------------------------------------+
|                   N-GRAM PHRASE INTELLIGENCE PIPELINE                         |
|     (src/streaming_phrase_extractor.py, src/phrase_ranker.py)                 |
|  - Cross-segment token stream reconstruction                                  |
|  - Exhaustive N-gram counting: Bigrams (2-gram) and Trigrams (3-gram)          |
|  - Statistical Association: PMI and Tri-PMI association calculations         |
|  - Linguistic Filtering: Stopword pruning & Penn Treebank POS validation     |
|  - Content-word density scoring                                               |
|  - Deterministic Quality Scoring: Score >= 6.5 (HIGH), >= 4.0 (MED), < 4.0    |
|  - Parent-child phrase linking (sub-phrase nesting)                           |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                      STREAMLIT REAL-TIME PRESENTATION                         |
|  - Live Subtitle Feed with Timestamps, Speaker Badges, Phrase Highlights      |
|  - Recurring Key Phrase Ticker Bar (Top 8 collocations)                       |
|  - Phrase Analytics Table with N-gram size, Frequency, PMI, Score, POS        |
|  - Nested Phrase Hierarchy Expander                                           |
|  - Speaker Diarization Breakdown (Turn counts, Durations, Percentages)        |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                           MULTI-FORMAT EXPORT ENGINE                          |
|  - transcript.txt: Formatted human-readable report with speaker tags          |
|  - transcript.json: Complete machine-readable session dump                    |
|  - phrases.csv: Ranked n-gram collocations with statistics                    |
|  - subtitles.csv: Timestamped subtitle segments with speaker labels           |
+-------------------------------------------------------------------------------+
```

---

## 3. Detailed Component Responsibilities

### 3.1 Audio Ingestion & Validation (`src/audio/loader.py`)
- **Responsibilities**:
  - Validates user input before decoding: verifies non-empty files ($> 0$ bytes), allowed extensions (`.wav`, `.mp3`, `.m4a`), and file size limits (default 150 MB).
  - Uses PyAV to decode diverse audio containers into uniform mono 16 kHz 16-bit PCM arrays.
  - Converts samples into normalized 1D `float32` NumPy arrays in range $[-1.0, 1.0]$.
  - Produces immutable `AudioMetadata` dataclasses containing filename, format, duration, sample rate, channels, and file size.
  - Protects users from crashes by converting low-level FFmpeg errors into clear, human-readable `AudioLoadError` exceptions.

### 3.2 Audio Chunking (`src/audio/chunker.py`)
- **Responsibilities**:
  - Divides continuous audio sample arrays into discrete, sequential `AudioChunk` objects.
  - Supports configurable chunk duration (default 5.0 seconds).
  - Preserves exact millisecond-accurate start and end timestamps for every chunk.
  - Handles the final partial chunk cleanly without padding silence or dropping remaining speech frames.

### 3.3 Local ASR Module (`src/audio/asr.py`)
- **Responsibilities**:
  - Encapsulates `faster-whisper` (CTranslate2 inference engine) running strictly on local CPU with INT8 quantization or local GPU.
  - Enforces model cache isolation on the `F:` drive by pre-configuring `HF_HUB_CACHE` and `HF_HOME` to `F:\Real-Time Subtitle Segmenter\models`.
  - Transcribes individual `AudioChunk` objects into structured `ASRChunkResult` objects containing raw text, normalized text, processing latency, and word-level timestamps.
  - Guarantees zero OpenAI dependencies, zero cloud API calls, and zero external telemetry.

### 3.4 Speaker Diarization Module (`src/audio/diarization.py`)
- **Responsibilities**:
  - Provides modular offline speaker clustering via `AcousticEnergyDiarizer`.
  - Operates 100% locally on CPU using NumPy without requiring PyTorch, CUDA, or Hugging Face authentication tokens.
  - Computes frame-level short-time energy (RMS), zero-crossing rates (ZCR), and spectral centroid features across voice-active regions.
  - Performs K-Means clustering into discrete speaker clusters (`Speaker 1`, `Speaker 2`).
  - Supports automatic speaker count estimation or explicit user-configured speaker counts ($1$–$10$).
  - Emits chronological `SpeakerSegment` objects with start time, end time, and normalized speaker label.

### 3.5 Temporal Alignment & Pipeline Orchestrator (`src/audio/pipeline.py`)
- **Responsibilities**:
  - Coordinates progressive execution via `AudioStreamingSession`.
  - Matches ASR transcription segments with detected speaker intervals using **maximum temporal overlap** with deterministic tie-breaking.
  - Flushes the subtitle segmentation buffer when a speaker transition occurs, guaranteeing that distinct speakers are never combined into a single subtitle frame.
  - Exposes sequential `.step()` execution, session reset, and export serialization methods.

### 3.6 Subtitle Segmentation Engine (`src/improved_segmenter.py`, `src/boundary_scorer.py`)
- **Responsibilities**:
  - Maintains an incremental FIFO rolling token buffer (`StreamingBuffer`).
  - Employs a multi-signal boundary scoring algorithm that evaluates candidate split points based on discourse markers, matured clause conjunctions, and balanced word lengths.
  - Integrates `PhraseProtector` (`src/phrase_protection.py`) to prevent splitting across protected multi-word collocations.
  - Enforces hard constraints unconditionally: minimum 5 words, maximum 14 words, and maximum 80 characters per subtitle frame.

### 3.7 N-gram Collocation & Ranking Engine (`src/phrase_ranker.py`, `src/ngram_extractor.py`)
- **Responsibilities**:
  - Reconstructs a continuous, normalized token stream across finalized subtitle segments.
  - Exhaustively counts bigrams ($N=2$) and trigrams ($N=3$).
  - Calculates Pointwise Mutual Information (PMI) and Tri-PMI to measure statistical word co-occurrence beyond baseline unigram probabilities.
  - Applies Penn Treebank Part-of-Speech (POS) pattern filtering to penalize stopword-heavy n-grams and reward meaningful noun/entity phrases.
  - Deterministically calculates a unified **Phrase Quality Score** and classifies phrases into `HIGH`, `MEDIUM`, and `LOW` tiers.
  - Identifies parent-child relationships between overlapping bigrams and trigrams.

### 3.8 User Interface (`src/ui_app.py`)
- **Responsibilities**:
  - Official, permanent frontend built with Streamlit.
  - Provides controls for audio uploading, chunk sizing, model size selection, diarization toggling, and NLP filter thresholds.
  - Renders the progressive subtitle feed, active subtitle frame, recurring phrase ticker, ranked phrase data table, speaker turns table, and download buttons.
  - Displays user-friendly error banners with one-click session recovery.

---

## 4. Data Flow Walkthrough

```text
1. User uploads 'meeting.wav' (21.9s, stereo, 44.1kHz).
2. AudioLoader downmixes to mono, resamples to 16kHz, normalizes to float32 array.
3. AudioMetadata extracted: duration=21.86s, format=WAV, size=964KB.
4. [If Diarization ON] AcousticEnergyDiarizer segments audio into 19 speech intervals.
5. AudioChunker splits samples into 5 sequential chunks (5.0s each).
6. Loop:
   a. Pipeline feeds Chunk 1 to LocalASR.
   b. LocalASR outputs transcribed text: "Today we are discussing artificial intelligence..."
   c. Temporal alignment assigns "Speaker 1" to Chunk 1 based on timestamp overlap.
   d. Text tokens appended to StreamingBuffer.
   e. BoundaryScorer detects discourse boundary, emits Subtitle Frame #1.
   f. Emitted tokens processed by StreamingPhraseExtractor.
   g. PMI, POS, and Quality scores calculated.
   h. Streamlit UI re-renders active subtitle and phrase ticker.
7. End of stream reached: remaining buffer flushed, final results and exports ready.
```

---

## 5. Architectural Separation of Concerns
| Layer | Module Path | Dependencies | State Scope |
| :--- | :--- | :--- | :--- |
| **Audio I/O** | `src/audio/loader.py`, `src/audio/chunker.py` | PyAV, wave, numpy | Stateless functions |
| **Acoustic ASR** | `src/audio/asr.py` | faster-whisper, CTranslate2 | Model cached in singleton |
| **Diarization** | `src/audio/diarization.py` | numpy | Session-level instance |
| **Segmentation** | `src/improved_segmenter.py`, `src/boundary_scorer.py` | Python stdlib, regex | FIFO token buffer |
| **NLP Collocations**| `src/phrase_ranker.py`, `src/pmi_calculator.py` | NLTK, math, collections | Running token vocabulary |
| **Orchestration** | `src/audio/pipeline.py` | Combines above layers | `AudioStreamingSession` |
| **Presentation** | `src/ui_app.py` | Streamlit, pandas | `st.session_state` |
| **Evaluation** | `evaluation/scripts/` | Python stdlib, numpy | Standalone benchmark suite |

---

## 6. Deprecated React Architecture Notice
Earlier experimental phases explored a React / Node.js web client. That architecture has been **permanently deprecated, decoupled, and removed**. Streamlit is the sole, permanent production frontend. There are zero JavaScript build steps, zero npm dependencies, and zero cross-origin API servers required to run this project.
