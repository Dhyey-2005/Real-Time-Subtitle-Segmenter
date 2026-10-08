# STEP-BY-STEP DEMONSTRATION & PRESENTATION GUIDE

## 1. Prerequisites & Environment Setup
Open a PowerShell terminal and navigate to the project root:
```powershell
cd "F:\Real-Time Subtitle Segmenter"
```

Verify that the local Python 3.12 virtual environment is present:
```powershell
.\venv\Scripts\python.exe --version
```
*(Expected output: Python 3.12.x)*

---

## 2. Launching the Application
Execute the official project entrypoint script:
```powershell
.\venv\Scripts\python.exe run_phase5.py --port 8501
```

Once launched, the terminal will display:
```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```
Open **`http://localhost:8501`** in Google Chrome or Microsoft Edge.

---

## 3. Step-by-Step Demonstration Script

### Demo Scenario 1: Single-Speaker Speech & Subtitle Segmentation
1. **Select Pipeline Mode**:
   - In the sidebar under **"1. Pipeline Mode"**, ensure **"📁 Mode A — Uploaded Audio (Local ASR)"** is selected.
2. **Select Audio Source**:
   - Under **"2. Audio Source Selection"**, choose **"Preloaded Speech Sample (15.1s WAV)"** (or upload `data/input/sample_speech.wav`).
3. **Inspect Audio Metadata**:
   - Observe the 6-metric audio card in the main view:
     - **Filename**: `sample_speech.wav`
     - **Format**: `WAV`
     - **Duration**: `00:15.12`
     - **Sample Rate**: `22.1 kHz`
     - **Channels**: `1 (Mono)`
     - **File Size**: `651.1 KB`
   - Play the embedded audio player to confirm clear speech playback.
4. **Configure ASR & Diarization**:
   - In the sidebar, select **faster-whisper Model Size**: `tiny` or `base`.
   - Set **Audio Chunk Duration**: `5.0` seconds.
   - Leave **"Enable speaker diarization"** unchecked (**OFF**).
5. **Start Processing**:
   - Click the green **"▶️ Start ASR"** button.
   - Observe the progressive status banner:
     - `Running transcription (Chunk 1 of 4)...`
     - Progress bar updating from 0% to 100%.
     - Live KPI cards updating: Recognized Words, Subtitles, Ranked Phrases.
6. **Examine Progressive Subtitles**:
   - **Active Subtitle Frame**: Shows the most recent subtitle with timestamp (`00:11.00 → 00:15.00`) and highlighted recurring phrases.
   - **Subtitle History Timeline**: Expand the timeline to view past finalized subtitle chunks. Notice that splits occur naturally at clause boundaries and discourse markers, not mid-phrase.
7. **Examine Collocation Intelligence**:
   - **Recurring Key Phrase Ticker**: Look at the top chips bar displaying recurring terms like `RENEWABLE ENERGY TRANSITION` with frequency, quality tier badge (`★ HIGH`), and quality score.
   - **Ranked Collocations Table**: Filter by `HIGH`, `MEDIUM`, or `LOW`. Observe the `N-gram`, `Freq`, `PMI`, `Score`, and Penn Treebank `POS` pattern (`JJ NN NN`).
   - **Nested Phrase Hierarchy**: Expand **"🔗 Overlapping & Nested Phrase Hierarchy"** to show how sub-phrases like *"renewable energy"* are nested under *"renewable energy transition"*.
8. **Export Results**:
   - Under **"💾 Export Audio Artifacts"**, click:
     - **📥 Transcript (TXT)** $\to$ downloads `transcript.txt`
     - **📥 Session (JSON)** $\to$ downloads `transcript.json`
     - **📥 Phrases (CSV)** $\to$ downloads `phrases.csv`
     - **📥 Subtitles (CSV)** $\to$ downloads `subtitles.csv`
   - Open any exported file to demonstrate structured data integrity.

---

### Demo Scenario 2: Multi-Speaker Audio with Speaker Diarization
1. **Change Audio Source**:
   - In the sidebar, change **Audio Source Selection** to **"Preloaded Multi-Speaker Sample (21.9s WAV)"** (or upload `data/input/multi_speaker_test.wav`).
2. **Enable Speaker Diarization**:
   - In the sidebar under **"4. Speaker Diarization"**, check the box: **"Enable speaker diarization"**.
   - Note the guidance caption: *"Speaker 1 / Speaker 2 are anonymous speaker labels, not real biometric identities."*
   - Select **Speaker Detection**: `Automatic` (or `Known number of speakers` set to `2`).
3. **Execute Processing**:
   - Click **"▶️ Start ASR"**.
   - Watch the pipeline execute acoustic diarization and chunked transcription.
4. **Verify Diarization Features**:
   - **Speaker Badges in Subtitle Feed**: Observe that subtitles are now tagged with distinct purple badges: `● SPEAKER 1` and `● SPEAKER 2`.
   - **Speaker-Change Boundary Splits**: Point out that when the dialogue shifts between speakers, the subtitle cuts immediately, preventing two speakers from ever sharing a single subtitle frame.
   - **Speaker Turns Timeline Table**: Expand this dedicated table showing chronological speaker turn intervals.
   - **Speaker Diarization Summary**: Review detected speaker counts, turn counts, speech duration, and percentage share.
5. **Inspect Performance**:
   - Notice the **Real-Time Factor (RTF)** metric (`~0.09x`), showing that multi-speaker audio with diarization processed in under $2$ seconds on CPU.

---

### Demo Scenario 3: Error Handling & Recovery
1. **Upload an Invalid File**:
   - In the sidebar, select **"Upload Audio File (WAV, MP3, M4A)"**.
   - Rename any small text file or create an empty `.wav` file (0 bytes) and upload it.
2. **Observe Graceful Error Handling**:
   - The UI immediately displays a clean, user-facing error message:
     `⚠️ The uploaded audio file is empty (0 bytes).`
   - **No Python stack trace is displayed to the user.**
3. **One-Click Recovery**:
   - Click the prominent button: **"🔄 Clear & Upload Another Audio File"**.
   - The error clears, session state resets, and the dashboard returns to a clean, ready state.

---

## 4. Key Talking Points for Evaluators
- **100% Local & Private**: No audio, text, or tokens ever leave this machine.
- **Quantized Efficiency**: Using INT8 Whisper on CPU achieves sub-2-second turnaround on 20-second audio without needing an expensive GPU.
- **Explainable NLP**: Every phrase quality score is mathematically traceable back to PMI, POS rules, and frequency, eliminating black-box LLM hallucinations.
- **Production Polish**: 203 automated tests passing, strict drive isolation on `F:`, and comprehensive multi-format export capabilities.
