# COMPREHENSIVE TECHNICAL VIVA Q&A REFERENCE

This document contains 32 concise, technically rigorous questions and answers designed for academic viva voce, technical defense, and engineering reviews.

---

### Q1: What is the core problem this project solves?
**Answer**: Raw Automatic Speech Recognition (ASR) engines produce continuous, unpunctuated streams of words ("wall-of-text"). This project ingests static audio, transcribes it locally, segments the unpunctuated stream into human-readable subtitle chunks using linguistic boundary scoring, attributes speech turns to speakers via optional diarization, and extracts ranked recurring n-gram collocations for real-time topic intelligence.

---

### Q2: Why use `faster-whisper` instead of standard OpenAI Whisper or Hugging Face Transformers?
**Answer**: `faster-whisper` is built on **CTranslate2**, a specialized C++ inference engine for Transformer models. It implements 8-bit integer quantization (INT8) on CPU, delivering up to $4\times$ faster inference speeds and $2\times$ lower memory footprint compared to standard PyTorch implementations, enabling sub-2-second transcription on standard consumer CPUs without a discrete GPU.

---

### Q3: Why local ASR instead of cloud transcription APIs (e.g., Google STT or OpenAI API)?
**Answer**: Local ASR ensures complete data privacy (confidential audio never leaves the host machine), eliminates recurring API costs, avoids internet latency or outages, and guarantees deterministic repeatability without cloud API schema drift.

---

### Q4: Why is Streamlit the official permanent frontend rather than React?
**Answer**: Streamlit provides a pure Python architecture where the user interface directly shares in-memory NumPy arrays, token lists, and Python object references with the pipeline. React was deprecated and removed because it introduced redundant Node.js runtime dependencies, cross-origin HTTP serialization overhead, and state synchronization complexity without adding functional value to the core offline workflow.

---

### Q5: Why is static audio upload the primary product workflow rather than real-time microphone capture?
**Answer**: In professional captioning, legal transcription, and archival workflows, static audio files (`WAV`, `MP3`, `M4A`) represent the dominant use case. Static audio provides auditable, reproducible test inputs free from browser audio driver bugs, fluctuating microphone gains, or client-side WebRTC drops. While live microphone capture was implemented in earlier compatibility phases, static audio is the stable, permanent product workflow.

---

### Q6: Why divide the audio into chunks rather than transcribing the entire file in a single pass?
**Answer**: Chunking (default 5.0s) enables **progressive streaming simulation**. It allows the UI to stream subtitle frames and update phrase tickers incrementally, bounds memory usage to small audio buffers, and models real-world broadcast captioning pipelines where text arrives progressively.

---

### Q7: Why is syntactic subtitle segmentation necessary if Whisper already outputs text?
**Answer**: Whisper ASR outputs text in arbitrary segment boundaries based on acoustic silence or fixed token windows, often cutting sentences abruptly (e.g., splitting between an adjective and a noun or after prepositions). Our `ImprovedStreamingSegmenter` evaluates discourse markers, clause conjunctions, human reading speed bounds (5–14 words, max 80 characters), and multi-word phrase protection to ensure natural subtitle readability.

---

### Q8: What are N-grams, and why are they used for phrase extraction?
**Answer**: An N-gram is a contiguous sequence of $N$ items from a given text stream. In our system, word-level N-grams capture recurring domain terms (e.g., bigrams like *"renewable energy"* and trigrams like *"clean energy infrastructure"*). They provide a deterministic, computationally lightweight method to discover technical collocations without relying on heavy generative language models.

---

### Q9: Why specifically bigrams ($N=2$) and trigrams ($N=3$)?
**Answer**: In linguistic discourse analysis, over 85% of meaningful technical and policy collocations are 2-word or 3-word noun phrases. Unigrams ($N=1$) are too broad and often ambiguous without context, while 4-grams and higher suffer from extreme data sparsity and rarely recur with statistical significance in short speech recordings.

---

### Q10: What is Pointwise Mutual Information (PMI) and why is it used?
**Answer**: Pointwise Mutual Information measures the statistical association between two words by comparing their joint co-occurrence probability against the probability of their independent co-occurrence:
$$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1)P(w_2)}$$
Raw frequency alone favors generic stopword pairs like *"of the"*. PMI quantifies how much more often words appear together than by random chance, surfacing genuine domain collocations.

---

### Q11: What is Tri-PMI?
**Answer**: Tri-PMI extends Pointwise Mutual Information to 3-word sequences (trigrams) using True Joint Specific Mutual Information:
$$\text{Tri-PMI}(w_1, w_2, w_3) = \log_2 \frac{P(w_1, w_2, w_3)}{P(w_1)P(w_2)P(w_3)}$$
It ensures that 3-word collocations like *"clean energy infrastructure"* are evaluated as a cohesive tripartite semantic unit rather than two disjoint bigrams.

---

### Q12: Why is Part-of-Speech (POS) filtering necessary alongside PMI?
**Answer**: PMI can occasionally score idiosyncratic pairs highly if the individual words are rare. NLTK Penn Treebank POS filtering enforces grammatical validity by awarding bonuses to recognized noun phrase patterns (`JJ + NN`, `NN + NN`, `NNP + NNP`) and penalizing functional glue words, prepositions, and determiners.

---

### Q13: Why enforce a minimum frequency threshold ($\text{min\_frequency} \ge 2$)?
**Answer**: Collocations, by definition, represent *recurring* patterns of speech. A phrase uttered only once does not demonstrate statistical association in the context of the speech session. Enforcing $\text{min\_frequency} \ge 2$ filters out one-off conversational phrases and isolates the true central themes of the recording.

---

### Q14: How is the Phrase Quality Score calculated?
**Answer**: It is a composite deterministic formula combining recurrence, statistical association, content density, and grammar:
$$\text{Score} = 1.5\log_2(\text{Freq} + 1) + 0.35\max(0, \text{PMI}) + 2.5(\text{Content Ratio}) + \text{POS Bonus} + \text{Length Bonus} - \text{Penalties}$$
Penalties include $-8.0$ for all-stopword n-grams and $-2.5$ for boundary stopwords.

---

### Q15: Why is deterministic ranking important?
**Answer**: Deterministic ranking ensures that identical audio transcripts always produce the exact same ranked phrase list. The sorting order prioritizes Quality Score ($\downarrow$), then Frequency ($\downarrow$), with ties broken alphabetically by phrase string ($\uparrow$). This eliminates non-deterministic ordering bugs across app re-renders.

---

### Q16: What is Speaker Diarization?
**Answer**: Speaker diarization is the task of determining "who spoke when" in an audio recording by partitioning the audio stream into homogeneous segments attributed to distinct speakers.

---

### Q17: How does the local `AcousticEnergyDiarizer` work?
**Answer**: It is a lightweight, 100% offline acoustic clustering module implemented in NumPy. It computes Root-Mean-Square (RMS) energy, Zero-Crossing Rate (ZCR), and spectral centroids across overlapping 500ms frames, filters out silence, and applies K-Means clustering to partition voice-active frames into anonymous speaker clusters (`Speaker 1`, `Speaker 2`).

---

### Q18: Why is Speaker Diarization optional in the UI?
**Answer**: Single-speaker audio (e.g., lectures, solo presentations, voice notes) does not require diarization. Keeping diarization optional (defaulting to OFF) ensures maximum backward compatibility and saves computation time on single-speaker recordings.

---

### Q19: Why is the diarization accuracy relatively low (47.25% accuracy / 52.75% DER)?
**Answer**: The `AcousticEnergyDiarizer` is an intentional trade-off: it uses basic acoustic signal heuristics (energy and spectral centroid) to avoid heavy neural weights (PyTorch/CUDA) or cloud token requirements. In the multi-speaker benchmark, both synthetic voices share similar digital frequency profiles, causing feature overlap. However, it achieved **100% accuracy in detecting speaker transitions**, successfully triggering subtitle boundaries at speaker changes.

---

### Q20: What is Word Error Rate (WER)?
**Answer**: Word Error Rate is the gold-standard metric for evaluating speech recognition accuracy:
$$\text{WER} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Reference Word Count}}$$
It measures the minimum edit operations needed to transform the hypothesis text into the reference text. In Phase 10, our system achieved a **3.17% mean WER**.

---

### Q21: What is Character Error Rate (CER)?
**Answer**: Character Error Rate measures Levenshtein edit distance at the individual character level divided by total reference characters. It provides insight into spelling or morphological errors. In Phase 10, our system achieved a **1.83% mean CER**.

---

### Q22: What is Real-Time Factor (RTF)?
**Answer**: Real-Time Factor is the ratio of total processing time to the duration of the audio:
$$\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$$
An $\text{RTF} < 1.0$ indicates that the system processes audio faster than real-time playback. Our system achieved a warm **RTF of 0.077x**, meaning it processes audio roughly $13\times$ faster than real-time playback on CPU.

---

### Q23: What is Diarization Error Rate (DER)?
**Answer**: Diarization Error Rate is the percentage of time attributed to the wrong speaker, missed speech, or false alarm speech relative to total ground truth speech duration. In our benchmark, the offline heuristic diarizer achieved a DER of **52.75%**.

---

### Q24: How was the evaluation performed?
**Answer**: It was performed using an automated, standalone benchmark suite (`evaluation/scripts/run_benchmarks.py`) across three controlled test recordings totaling 42.79 seconds. It measured WER, CER, speaker assignment accuracy under optimal label permutation, subtitle boundary precision (±0.5s tolerance), phrase precision/recall/F1, and cold vs warm processing times over 3 repetitions.

---

### Q25: How was data leakage avoided during evaluation?
**Answer**: Ground truth reference files were stored strictly under `evaluation/references/`. The production pipeline received only raw audio file paths or byte buffers. Ground truth transcripts and annotations were never passed into the ASR, diarizer, or NLP modules during evaluation.

---

### Q26: Why is Phrase Recall 100% but Precision 55.55%?
**Answer**: Recall is 100% because every expected key domain phrase (e.g., *"clean energy infrastructure"*, *"renewable energy transition"*) was successfully discovered. Precision is 55.55% because the n-gram extractor also extracted valid constituent sub-phrases (e.g., *"renewable energy"*, *"clean energy"*) which were counted as false positives by the strict automated test script, but are useful and nested in the UI.

---

### Q27: How does the subtitle segmenter decide where to split unpunctuated text?
**Answer**: It uses a composite scoring engine:
- Strong discourse markers (+65.0 bonus)
- Coordinating conjunctions preceding mature clauses (+35.0 bonus)
- Completed key phrases (+35.0 bonus)
- Function word split penalties (-30.0)
- Protected phrase penalties (-85.0 to prevent splitting inside key collocations)
- Hard limits: 5–14 words, max 80 characters, and immediate cuts on speaker changes.

---

### Q28: What are the system's primary technical limitations?
**Answer**:
1. Diarizer uses basic acoustic heuristics rather than neural speaker embeddings.
2. The benchmark dataset is compact (42.79s) and evaluated on clean synthesized speech.
3. Recurring phrases require frequency $\ge 2$; single-occurrence key terms are not extracted.
4. Anonymous speaker labels (`Speaker 1`, `Speaker 2`) without biometric identification.

---

### Q29: What engineering improvements would you implement next?
**Answer**:
1. Export a lightweight neural speaker embedding model (e.g., ONNX WeSpeaker or Pyannote) to boost diarization accuracy on overlapping timbres.
2. Incorporate local TF-IDF or KeyBERT embeddings to capture high-salience single-occurrence keywords.
3. Memory-map multi-hour audio files for long lecture transcription.

---

### Q30: Why is OpenAI strictly not used in this project?
**Answer**: To eliminate API subscription costs, prevent proprietary user audio from leaving the machine, avoid network downtime vulnerabilities, and adhere to strict local deployment requirements.

---

### Q31: How is F: drive isolation guaranteed?
**Answer**: Environment variables `HF_HOME`, `HF_HUB_CACHE`, and NLTK data directories are programmatically set to `F:\Real-Time Subtitle Segmenter\models` and `F:\Real-Time Subtitle Segmenter\nltk_data` at module initialization, preventing model downloads or temporary caches from being written to the `C:` drive roaming profile.

---

### Q32: How are overlapping/nested phrases handled?
**Answer**: When a bigram is a constituent substring of a recurring trigram (e.g., *"clean energy"* inside *"clean energy infrastructure"*), the system links them in a parent-child hierarchy in `phrase_ranker.py`. In the UI, sub-phrases are grouped under their parent trigram to keep the ticker clean and expose semantic hierarchy.
