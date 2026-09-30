"""Unit tests for Jam Copilot DSP modules: normalization, BPM estimation, and synthesis."""

import numpy as np
import pytest

from src.analyzer import AudioAnalyzer
from src.generator import AccompanimentGenerator
from src.transcriber import PitchChromaTranscriber


@pytest.fixture
def sample_rate():
    return 22050


@pytest.fixture
def analyzer(sample_rate):
    return AudioAnalyzer(sample_rate=sample_rate)


@pytest.fixture
def generator(sample_rate):
    return AccompanimentGenerator(sample_rate=sample_rate)


@pytest.fixture
def transcriber(sample_rate):
    return PitchChromaTranscriber(sample_rate=sample_rate)


def test_audio_normalization(analyzer):
    """Verify peak normalization scales waveform accurately without distortion."""
    audio = np.array([0.1, -0.4, 0.2, -0.8], dtype=np.float32)
    normalized = analyzer.normalize(audio, target_peak=0.95)

    assert pytest.approx(np.max(np.abs(normalized)), rel=1e-4) == 0.95
    assert np.all(np.abs(normalized) <= 0.95)

    # Test silence safety (no division by zero)
    silence = np.zeros(100, dtype=np.float32)
    norm_silence = analyzer.normalize(silence)
    assert np.all(norm_silence == 0.0)


def test_rms_computation(analyzer, sample_rate):
    """Verify RMS energy on known sine wave (theoretical RMS = A / sqrt(2))."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.6
    sine_wave = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    computed_rms = analyzer.compute_rms(sine_wave)
    expected_rms = amplitude / np.sqrt(2)
    assert pytest.approx(computed_rms, rel=1e-2) == expected_rms


@pytest.mark.parametrize("target_bpm", [100.0, 120.0, 140.0])
def test_bpm_extraction(analyzer, sample_rate, target_bpm):
    """Verify autocorrelation BPM estimation across multiple tempos."""
    duration_sec = 4.0
    total_samples = int(sample_rate * duration_sec)
    audio = np.zeros(total_samples, dtype=np.float32)

    interval_samples = int(sample_rate * (60.0 / target_bpm))
    for idx in range(0, total_samples - 300, interval_samples):
        # Sharp exponential click transient
        t_click = np.arange(200) / sample_rate
        burst = np.sin(2 * np.pi * 1200.0 * t_click) * np.exp(-t_click * 80.0)
        audio[idx : idx + 200] += burst.astype(np.float32)

    estimated_bpm, confidence = analyzer.estimate_bpm(audio)

    assert confidence > 0.3
    # Check within 4% of target BPM or octave harmonic
    is_accurate = (
        pytest.approx(target_bpm, rel=0.04) == estimated_bpm
        or pytest.approx(target_bpm / 2.0, rel=0.04) == estimated_bpm
    )
    assert is_accurate


def test_accompaniment_duration(generator, sample_rate):
    """Verify backing track generator matches exact requested duration and sample count."""
    durations = [1.5, 3.25, 5.0]
    for dur in durations:
        backing = generator.generate_backing_track(bpm=120.0, duration_sec=dur)
        expected_samples = int(sample_rate * dur)
        assert len(backing) == expected_samples
        assert backing.dtype == np.float32
        assert not np.isnan(backing).any()
        assert not np.isinf(backing).any()
        assert np.max(np.abs(backing)) <= 1.0


def test_pitch_tracking_and_conversion(transcriber, sample_rate):
    """Verify fundamental frequency extraction on pure A2 (110 Hz) tone."""
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 110.0  # A2
    tone = (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    detected_pitch = transcriber.estimate_pitch_autocorr(tone)
    assert pytest.approx(freq, rel=0.02) == detected_pitch

    note_name = transcriber.hz_to_note_name(detected_pitch)
    assert note_name == "A2"


def test_chroma_feature_dimension(transcriber, sample_rate):
    """Verify chroma vector output has 12 semitone bins and normalizes correctly."""
    audio = np.random.uniform(-0.1, 0.1, 4096).astype(np.float32)
    chroma = transcriber.compute_chroma(audio)

    assert chroma.shape == (12,)
    assert pytest.approx(float(np.sum(chroma)), rel=1e-2) == 1.0
