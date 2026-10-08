# SYSTEM METHODOLOGY

## 1. Introduction
This document details the theoretical foundation, algorithms, and practical implementation choices employed in the **Real-Time Subtitle Segmenter and N-gram Phrase Extractor**. The methodology combines digital signal processing, speech recognition, rule-based computational linguistics, and information theory.

---

## 2. Audio Processing Methodology

### 2.1 Audio Ingestion & Decoding
Audio files (`WAV`, `MP3`, `M4A`) are decoded into raw uncompressed audio samples using **PyAV**, a Pythonic binding to FFmpeg:
- **Demuxing & Decoding**: Opens container streams and extracts raw audio packets.
- **Resampling**: Uses `av.AudioResampler` to convert varying source sample rates (e.g., 22.05 kHz, 44.1 kHz, 48 kHz) into a uniform **16,000 Hz** rate required by the Whisper acoustic model.
- **Channel Downmixing**: Multi-channel inputs (stereo, 5.1 surround) are downmixed to a single mono audio stream by averaging channel amplitudes.
- **Normalization**: 16-bit signed integer PCM samples ($-32768$ to $+32767$) are normalized to 32-bit floating-point numbers in the interval $[-1.0, 1.0]$:
  $$x_{\text{norm}}[n] = \frac{x_{\text{int16}}[n]}{32768.0}$$

### 2.2 Audio Chunking
To enable progressive subtitle emission and responsive UI feedback without loading entire minutes of audio into memory at once, audio is divided into sequential chunks:
- **Chunk Duration**: Configurable duration $T_{\text{chunk}}$ (default: 5.0 seconds).
- **Sample Length**: Each chunk contains $N = T_{\text{chunk}} \times 16000$ samples.
- **Timestamp Tracking**: Each chunk preserves its absolute temporal offsets:
  $$t_{\text{start}} = i \times T_{\text{chunk}}, \quad t_{\text{end}} = \min((i + 1) \times T_{\text{chunk}}, T_{\text{total}})$$
- **Terminal Chunk Handling**: The final partial chunk is processed as-is without zero-padding, preventing silence artifacts.

---

## 3. Speech Recognition (ASR) Methodology

### 3.1 Local faster-whisper Model
Transcription is performed using `faster-whisper`, a reimplementation of OpenAI's Whisper model utilizing **CTranslate2**:
- **Why faster-whisper over standard Whisper**: CTranslate2 provides a highly optimized C++ inference engine with 8-bit integer quantization (INT8), yielding up to $4\times$ faster execution and $2\times$ lower memory consumption on CPU compared to standard PyTorch implementations.
- **Model Sizes**: Supports `tiny`, `base`, `small`, and `medium` checkpoints stored locally on the `F:` drive.
- **Decoding Configuration**: Operates with beam size $1$ (greedy decoding) for maximum speed and deterministic output, with automatic language detection defaulting to English.

---

## 4. Speaker Diarization Methodology

### 4.1 Acoustic Energy Clustering (`AcousticEnergyDiarizer`)
To maintain complete independence from cloud services, external APIs, and heavy neural frameworks, the system provides a local NumPy-based acoustic diarizer:
1. **Frame Extraction**: Audio is sliced into overlapping frames of length $500\text{ ms}$ with a hop size of $250\text{ ms}$.
2. **Feature Extraction**:
   - **Root-Mean-Square Energy (RMS)**: Measures acoustic volume and distinguishes voiced speech from background silence:
     $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x[n]^2}$$
   - **Zero-Crossing Rate (ZCR)**: Measures signal noisiness and unvoiced phoneme transitions:
     $$\text{ZCR} = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$
   - **Spectral Centroid**: Computes the "center of mass" of the frequency spectrum via Real FFT, distinguishing pitch differences between speakers:
     $$\text{Centroid} = \frac{\sum_{k} f_k |X[k]|}{\sum_{k} |X[k]|}$$
3. **Clustering**: Frames with energy exceeding the silence threshold ($0.005$) are clustered using K-Means into discrete speaker classes.
4. **Smoothing**: Consecutive speaker micro-segments under $0.6$ seconds are merged to eliminate high-frequency acoustic jitter.

### 4.2 Temporal Overlap Alignment
ASR utterances and diarization intervals are aligned using **maximum temporal overlap**:
- For each ASR utterance $[t_{\text{start}}^{\text{asr}}, t_{\text{end}}^{\text{asr}}]$ and speaker interval $[t_{\text{start}}^{\text{spk}}, t_{\text{end}}^{\text{spk}}]$, the overlap duration is:
  $$\text{Overlap} = \max\left(0, \min(t_{\text{end}}^{\text{asr}}, t_{\text{end}}^{\text{spk}}) - \max(t_{\text{start}}^{\text{asr}}, t_{\text{start}}^{\text{spk}})\right)$$
- The speaker exhibiting the largest overlap duration is assigned to the utterance.
- **Deterministic Tie-Breaking**: If two speakers share identical overlap, the interval whose midpoint is closest to the ASR segment's midpoint is selected.

---

## 5. Subtitle Segmentation Methodology

### 5.1 The Need for Syntactic Segmentation
Raw ASR outputs lack punctuation and natural clause structures. Simple fixed-word segmentation (e.g., splitting every 8 words) frequently cuts sentences in awkward positions (e.g., splitting between *"renewable"* and *"energy"* or after prepositions like *"announced a"*).

### 5.2 Multi-Signal Boundary Scoring
The `BoundaryScorer` evaluates all candidate split points in the token buffer and calculates a composite score:
$$\text{Score}(i) = S_{\text{discourse}} + S_{\text{clause}} + S_{\text{phrase}} + S_{\text{length}} - P_{\text{function}} - P_{\text{protection}}$$

1. **Discourse Transition Signals (+65.0)**: Major rhetorical markers (*however*, *therefore*, *meanwhile*, *in conclusion*, *furthermore*, *as a result*) indicate a fresh idea and trigger an immediate boundary before the marker.
2. **Clause Conjunction Boundaries (+35.0)**: Coordinating conjunctions (*and*, *but*, *because*) only trigger a split when the preceding clause contains a verb and has matured ($\ge 7$ words). Short fragments preceding conjunctions receive a penalty ($-25.0$) to avoid breaking lists like *"solar and wind"*.
3. **Phrase Completion (+35.0)**: Awards a bonus when a split lands immediately following a recognized multi-word collocation.
4. **Tail Function Word Penalties (-30.0)**: Penalizes splits immediately following prepositions (*with*, *for*, *to*), determiners (*the*, *a*, *this*), or auxiliary verbs (*is*, *are*).
5. **Length Balancing (+5.0 to +15.0)**: Optimizes segment length around a target balance of $8$–$9$ words.

### 5.3 Multi-Word Phrase Protection
The `PhraseProtector` maintains a lookup of key domain phrases and collocations. If a candidate split falls strictly inside a protected phrase $[t_k, t_{k+1}]$, a severe penalty ($-85.0$) is applied. This guarantees that multi-word concepts (e.g., *"clean energy infrastructure"*) remain intact within a single subtitle frame.

### 5.4 Hard Architectural Constraints
Regardless of heuristic scoring, three hard constraints are strictly enforced:
- **Minimum Words**: Subtitles must contain at least 5 words (except at end-of-stream flush).
- **Maximum Words**: Subtitles are forcibly cut at 14 words.
- **Maximum Characters**: Subtitles are forcibly cut at 80 characters.
- **Speaker Transitions**: Any detected speaker change immediately flushes the buffer to prevent combining multiple speakers in one subtitle frame.

---

## 6. N-gram Phrase Extraction Methodology

### 6.1 Tokenization & Cross-Segment Reconstruction
Subtitles arrive incrementally, but meaningful phrases often span across consecutive subtitle boundaries. The system continuously appends emitted tokens to a unified session token stream, ensuring no cross-boundary collocations are missed.
- **Normalization**: Lowercases text, collapses whitespace, and preserves alphanumeric tokens while isolating punctuation.

### 6.2 N-gram Generation
The system exhaustively extracts all consecutive bigrams ($N=2$) and trigrams ($N=3$):
$$\text{Bigrams} = \{(w_i, w_{i+1})\}_{i=1}^{M-1}, \quad \text{Trigrams} = \{(w_i, w_{i+1}, w_{i+2})\}_{i=1}^{M-2}$$

### 6.3 Pointwise Mutual Information (PMI)
Raw frequency alone fails to distinguish meaningful phrases from common word pairs (e.g., *"of the"* occurs frequently but carries no semantic value). The system calculates Pointwise Mutual Information (PMI) to measure statistical association:
$$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)}$$
Where:
- $P(w_1, w_2) = \frac{\text{Count}(w_1, w_2)}{N_{\text{bigrams}}}$
- $P(w_1) = \frac{\text{Count}(w_1)}{N_{\text{unigrams}}}$, $P(w_2) = \frac{\text{Count}(w_2)}{N_{\text{unigrams}}}$

A high positive PMI indicates that words co-occur far more frequently than expected by random chance.

### 6.4 Specific Mutual Information for Trigrams (Tri-PMI)
For trigrams $(w_1, w_2, w_3)$, the association is calculated using True Joint Specific Mutual Information:
$$\text{Tri-PMI}(w_1, w_2, w_3) = \log_2 \frac{P(w_1, w_2, w_3)}{P(w_1) P(w_2) P(w_3)}$$

### 6.5 Part-of-Speech (POS) Filtering
To eliminate grammatically meaningless phrases, tokens are tagged using NLTK's averaged perceptron tagger (Penn Treebank tagset):
- **Noun Phrase Bonus (+1.5 to +2.0)**: Preferred patterns receiving strong bonuses include:
  - `JJ + NN` (Adjective + Noun: *"renewable energy"*)
  - `NN + NN` (Noun + Noun: *"energy transition"*)
  - `NNP + NNP` (Proper Nouns: *"Microsoft David"*)
  - `JJ + NN + NN` or `NN + NN + NN` (*"clean energy infrastructure"*)
- **Stopword Pruning**:
  - All-stopword phrases (e.g., *"in the"*, *"and it was"*) receive a severe penalty ($-8.0$).
  - Boundary stopwords (e.g., phrases beginning or ending with *the*, *a*, *of*, *in*) receive a penalty ($-2.5$).

### 6.6 Content-Word Density
Measures the proportion of substantive content words (nouns, main verbs, adjectives) versus functional glue words:
$$\text{Content Ratio} = \frac{\sum_{w \in \text{phrase}} \mathbb{I}(w \in \text{Content Words})}{|\text{phrase}|}$$

### 6.7 Phrase Quality Score
The unified deterministic quality score combines recurrence, association, density, and grammar:
$$\text{Quality Score} = 1.5 \log_2(\text{Freq} + 1) + 0.35 \max(0, \text{PMI}) + 2.5(\text{Content Ratio}) + \text{POS Bonus} + \text{Length Bonus} - \text{Penalties}$$

### 6.8 Quality Categories & Deterministic Ranking
Phrases are categorized using explainable thresholds:
- **`HIGH` Quality**: Quality Score $\ge 6.5$ (High-value domain concepts and technical terms).
- **`MEDIUM` Quality**: $4.0 \le \text{Quality Score} < 6.5$ (Useful context phrases and supporting collocations).
- **`LOW` Quality**: Quality Score $< 4.0$ (Marginal phrases or common conversational pairs).

**Deterministic Sorting Order**:
1. Quality Score ($\downarrow$ descending)
2. Frequency ($\downarrow$ descending)
3. Alphabetical Phrase String ($\uparrow$ ascending, ensures 100% reproducible tie-breaking)

### 6.9 Nested Phrase Hierarchy
When an active bigram (e.g., *"renewable energy"*) is contained within a recurring trigram (e.g., *"renewable energy transition"*), the system links them in a parent-child relationship. This avoids visual clutter on the ticker while exposing the full conceptual hierarchy to users.

---

## 7. Performance Optimization Strategy
1. **Model Cache Re-use**: ASR models are loaded once into memory and reused across sessions, eliminating multi-second cold-start latency on subsequent runs.
2. **PyAV Generator Streaming**: Decodes audio frames incrementally without buffering massive uncompressed WAV data in memory.
3. **No Heavy Deep Learning for Diarization**: The lightweight acoustic diarizer runs in under $0.02$ seconds using NumPy matrix math on CPU.
4. **Vectorized Levenshtein for Benchmarking**: Dynamic programming matrix edit distance evaluates ASR error rates deterministically in milliseconds.
