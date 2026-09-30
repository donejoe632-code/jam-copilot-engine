# 🎸 Jam Copilot Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![DSP Powered](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-orange.svg)](https://scipy.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Zero--Bloat%20Prototypes-success.svg)](https://github.com/donejoe632-code/jam-copilot-engine)

Interactive, zero-bloat Audio ML & DSP prototypes for live guitarists: real-time BPM tracking, dynamic accompaniment generation, and fundamental pitch/chord transcription.

---

## ⚡ Overview

**Jam Copilot Engine** is a lightweight audio DSP engine designed to accompany live guitarists in real time without heavy frameworks or GPU dependencies. Built strictly on top of `numpy`, `scipy`, and `soundfile`, it installs in seconds and executes with minimal CPU overhead.

### Key Capabilities
- **Autocorrelation BPM Estimator (`AudioAnalyzer`)**: Extracts transient onset envelopes and computes autocorrelation lag metrics with parabolic peak interpolation for sub-BPM accuracy.
- **Dynamic Accompaniment Synthesizer (`AccompanimentGenerator`)**: Procedurally synthesizes tight kick drums, backbeat snares, 8th-note hi-hats, and synchronized root-fifth basslines matching input musical cues.
- **Pitch & Chroma Transcriber (`PitchChromaTranscriber`)**: Extracts fundamental frequency ($F_0$) via normalized autocorrelation and maps spectral energies into 12-semitone chroma vectors for key/chord detection.

---

## 🏛 DSP Signal Processing Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │              Input Guitar Audio Signal                 │
  │     (Live guitar riff / synthetic plucked string)      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │                 Onset & BPM Extractor                  │
  │  • Short-time energy framing & half-wave diff (dE/dt)  │
  │  • Normalized autocorrelation lag analysis             │
  │  • Parabolic peak interpolation: [60 - 200 BPM]        │
  └──────────┬───────────────────────────────┬─────────────┘
             │ Estimated BPM                 │ Root Key & RMS
             ▼                               ▼
  ┌────────────────────────────────────────────────────────┐
  │               Accompaniment Synthesis                  │
  │  • Procedural Kick (pitch-sweep sub) & Snare (noise)   │
  │  • Rhythmic 8th-note Hi-hat groove                     │
  │  • Synced root-fifth walking bassline                  │
  │  • Soft-saturation tanh dynamic limiter                │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │                  Final Mixed Output                    │
  │     Live Guitar (0.85) + Procedural Backing (0.90)     │
  │                  ──► output_jam.wav                    │
  └────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart (10 Seconds)

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/donejoe632-code/jam-copilot-engine.git
cd jam-copilot-engine

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install lightweight dependencies
pip install -r requirements.txt
```

### 2. Run the Demos

#### Demo 1: Real-Time BPM Tracker
Simulates a live syncopated guitar performance and estimates the tempo and dynamics:
```bash
python demos/demo_bpm_tracker.py
```

*Example Output:*
```
============================================================
🎸 JAM COPILOT - AUDIO DSP BPM TRACKER DEMO
============================================================
[1/3] Synthesizing guitar riff @ 124.0 BPM (4.5s)...
[2/3] Running AudioAnalyzer DSP pipeline...
      Processing completed in 3.12 ms
[3/3] Analysis Telemetry:
------------------------------------------------------------
  • Ground Truth BPM  : 124.0
  • Estimated BPM     : 124.0
  • Error             : 0.00%
  • Tempo Confidence  : 88.4%
  • RMS Energy        : 0.2814
  • Peak Amplitude    : 0.9500
  • Onsets Detected   : 9 attacks
------------------------------------------------------------
✅ SUCCESS: BPM estimation is within 5% tolerance window.
```

#### Demo 2: Dynamic Accompaniment & Jam Session Synthesis
Extracts musical cues from a guitar riff, synthesizes a synchronized drum and bass track, and exports the final mix to `output_jam.wav`:
```bash
python demos/demo_jam_synth.py
```

---

## 🧪 Testing

Execute the test suite with `pytest`:

```bash
pytest tests/ -v
```

### Test Coverage
- `test_audio_normalization`: Peak normalization bounds and zero-division resilience.
- `test_rms_computation`: Exact validation against theoretical sine wave RMS ($A/\sqrt{2}$).
- `test_bpm_extraction`: Multi-tempo pulse train autocorrelation verification (100, 120, 140 BPM).
- `test_accompaniment_duration`: Sample-accurate duration and boundary safety.
- `test_pitch_tracking_and_conversion`: Fundamental frequency extraction on A2 (110 Hz).
- `test_chroma_feature_dimension`: 12-semitone pitch class normalization.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
