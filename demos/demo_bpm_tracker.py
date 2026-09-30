"""Demo: Real-time rhythmic BPM tracking on synthetic guitar riffs."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.analyzer import AudioAnalyzer


def synthesize_guitar_strum(
    freq_hz: float,
    duration_sec: float,
    sr: int = 22050,
) -> np.ndarray:
    """Synthesize physical Karplus-Strong / plucked string approximation."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Fundamental + harmonics with non-linear decay
    fundamental = np.sin(2 * np.pi * freq_hz * t) * np.exp(-t * 8.0)
    harmonic2 = 0.5 * np.sin(2 * np.pi * freq_hz * 2.0 * t) * np.exp(-t * 14.0)
    harmonic3 = 0.25 * np.sin(2 * np.pi * freq_hz * 3.0 * t) * np.exp(-t * 22.0)
    strum = fundamental + harmonic2 + harmonic3
    return (strum / np.max(np.abs(strum))).astype(np.float32)


def generate_synthetic_riff(
    bpm: float = 124.0,
    total_sec: float = 4.0,
    sr: int = 22050,
) -> np.ndarray:
    """Generate rhythmic syncopated guitar riff at specific BPM."""
    total_samples = int(sr * total_sec)
    riff = np.zeros(total_samples, dtype=np.float32)

    sec_per_beat = 60.0 / bpm
    samples_per_beat = int(sr * sec_per_beat)

    # Chords: A2 (110Hz), C3 (130.8Hz), D3 (146.8Hz)
    chords = [110.0, 110.0, 130.81, 146.83]

    cur_sample = 0
    beat_idx = 0
    while cur_sample < total_samples:
        freq = chords[beat_idx % len(chords)]
        strum = synthesize_guitar_strum(freq, sec_per_beat * 0.85, sr=sr)
        end = min(cur_sample + len(strum), total_samples)
        riff[cur_sample:end] += strum[: end - cur_sample]

        cur_sample += samples_per_beat
        beat_idx += 1

    return AudioAnalyzer.normalize(riff)


def main():
    print("=" * 60)
    print("[*] JAM COPILOT - AUDIO DSP BPM TRACKER DEMO")
    print("=" * 60)

    sr = 22050
    ground_truth_bpm = 124.0
    duration_sec = 4.5

    print(f"\n[1/3] Synthesizing guitar riff @ {ground_truth_bpm} BPM ({duration_sec}s)...")
    audio = generate_synthetic_riff(bpm=ground_truth_bpm, total_sec=duration_sec, sr=sr)
    print(f"      Generated {len(audio)} samples ({duration_sec:.1f}s) @ {sr} Hz")

    print("\n[2/3] Running AudioAnalyzer DSP pipeline...")
    analyzer = AudioAnalyzer(sample_rate=sr)

    t0 = time.perf_counter()
    rms = analyzer.compute_rms(audio)
    peak = analyzer.compute_peak(audio)
    onsets = analyzer.detect_onsets(audio)
    estimated_bpm, confidence = analyzer.estimate_bpm(audio)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"      Processing completed in {dt_ms:.2f} ms")

    print("\n[3/3] Analysis Telemetry:")
    print("-" * 60)
    print(f"  - Ground Truth BPM  : {ground_truth_bpm:.1f}")
    print(f"  - Estimated BPM     : {estimated_bpm:.1f}")
    error_pct = abs(estimated_bpm - ground_truth_bpm) / ground_truth_bpm * 100.0
    print(f"  - Error             : {error_pct:.2f}%")
    print(f"  - Tempo Confidence  : {confidence * 100:.1f}%")
    print(f"  - RMS Energy        : {rms:.4f}")
    print(f"  - Peak Amplitude    : {peak:.4f}")
    print(f"  - Onsets Detected   : {len(onsets)} attacks")
    print(f"  - Onset Times (s)   : {[round(t, 2) for t in onsets[:8]]} ...")
    print("-" * 60)

    if error_pct < 5.0:
        print("[+] SUCCESS: BPM estimation is within 5% tolerance window.")
    else:
        print("[-] NOTICE: High deviation detected.")


if __name__ == "__main__":
    main()
