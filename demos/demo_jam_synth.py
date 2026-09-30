"""Demo: Full Jam Copilot pipeline: tempo estimation, root key detection, accompaniment synthesis, and export."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from src.analyzer import AudioAnalyzer
from src.generator import AccompanimentGenerator
from src.transcriber import PitchChromaTranscriber


def generate_input_guitar_riff(
    bpm: float = 120.0,
    duration_sec: float = 8.0,
    sr: int = 22050,
) -> np.ndarray:
    """Generate multi-bar guitar riff in A minor."""
    total_samples = int(sr * duration_sec)
    riff = np.zeros(total_samples, dtype=np.float32)

    sec_per_beat = 60.0 / bpm
    samples_per_beat = int(sr * sec_per_beat)

    # A minor chord tones: A2 (110Hz), C3 (130.8Hz), E3 (164.8Hz), G2 (98.0Hz)
    chords = [110.0, 130.81, 164.81, 98.00]

    cur_sample = 0
    beat = 0
    while cur_sample < total_samples:
        root_f = chords[(beat // 2) % len(chords)]
        dur = sec_per_beat * 0.8
        n_samples = int(sr * dur)
        t = np.linspace(0, dur, n_samples, endpoint=False)

        # Plucked timbre
        pluck = (
            0.6 * np.sin(2 * np.pi * root_f * t)
            + 0.3 * np.sin(2 * np.pi * root_f * 2.0 * t)
            + 0.1 * np.sin(2 * np.pi * root_f * 3.0 * t)
        ) * np.exp(-t * 10.0)

        end = min(cur_sample + n_samples, total_samples)
        riff[cur_sample:end] += pluck[: end - cur_sample].astype(np.float32)

        cur_sample += samples_per_beat
        beat += 1

    return AudioAnalyzer.normalize(riff, target_peak=0.7)


def main():
    print("=" * 60)
    print("[*] JAM COPILOT - REAL-TIME DSP ACCOMPANIMENT GENERATOR")
    print("=" * 60)

    sr = 22050
    duration_sec = 8.0
    target_bpm = 120.0
    output_filename = "output_jam.wav"

    print(f"\n[1/4] Generating live input guitar performance ({duration_sec}s @ {target_bpm} BPM)...")
    guitar_signal = generate_input_guitar_riff(bpm=target_bpm, duration_sec=duration_sec, sr=sr)
    print(f"      Signal size: {len(guitar_signal)} samples, Peak: {AudioAnalyzer.compute_peak(guitar_signal):.2f}")

    print("\n[2/4] Analyzing audio cues (tempo and root key)...")
    t0 = time.perf_counter()

    analyzer = AudioAnalyzer(sample_rate=sr)
    transcriber = PitchChromaTranscriber(sample_rate=sr)

    estimated_bpm, confidence = analyzer.estimate_bpm(guitar_signal)
    detected_key = transcriber.detect_dominant_pitch_class(guitar_signal)
    rms_energy = analyzer.compute_rms(guitar_signal)

    analysis_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"      - Detected Tempo : {estimated_bpm:.1f} BPM (Confidence: {confidence * 100:.1f}%)")
    print(f"      - Detected Key   : {detected_key}")
    print(f"      - Average RMS    : {rms_energy:.4f}")
    print(f"      - Analysis Time  : {analysis_time_ms:.2f} ms")

    print("\n[3/4] Synthesizing synchronized procedural backing track...")
    t1 = time.perf_counter()

    generator = AccompanimentGenerator(sample_rate=sr)
    backing_track = generator.generate_backing_track(
        bpm=estimated_bpm,
        duration_sec=duration_sec,
        root_note=detected_key,
        drum_level=0.85,
        bass_level=0.75,
    )

    synth_time_ms = (time.perf_counter() - t1) * 1000.0
    print(f"      Accompaniment synthesized in {synth_time_ms:.2f} ms ({len(backing_track)} samples)")

    print("\n[4/4] Mixing and exporting final audio file...")
    # Mix live guitar (0.75) and procedural backing (0.80)
    final_mix = AccompanimentGenerator.mix_signals(
        tracks=[guitar_signal, backing_track],
        weights=[0.85, 0.90],
    )

    out_path = Path(output_filename).resolve()
    sf.write(str(out_path), final_mix, sr, subtype="PCM_16")

    print(f"      Exported to: {out_path.name}")
    print(f"      File size  : {out_path.stat().st_size / 1024:.1f} KB")
    print(f"      Sample Rate: {sr} Hz, Format: 16-bit PCM WAV")
    print("-" * 60)
    print("[+] Jam session backing track successfully rendered!")


if __name__ == "__main__":
    main()
