"""Pitch and chroma transcription module for guitar notes and key detection."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PitchChromaTranscriber:
    """Lightweight pitch tracking and chroma feature extractor using pure NumPy/SciPy."""

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, sample_rate: int = 22050):
        """Initialize transcriber.

        Args:
            sample_rate: Audio sampling frequency in Hz.
        """
        self.sample_rate = sample_rate

    @classmethod
    def hz_to_midi(cls, freq_hz: float) -> float:
        """Convert frequency in Hertz to continuous MIDI note number.

        Args:
            freq_hz: Frequency in Hz.

        Returns:
            Fractional MIDI note value.
        """
        if freq_hz <= 0.0:
            return 0.0
        return 69.0 + 12.0 * float(np.log2(freq_hz / 440.0))

    @classmethod
    def midi_to_note_name(cls, midi_num: float) -> str:
        """Convert MIDI note number to note name with octave (e.g. 69 -> 'A4').

        Args:
            midi_num: MIDI note index.

        Returns:
            String representing note name and octave.
        """
        rounded = int(round(midi_num))
        pitch_class = cls.NOTE_NAMES[rounded % 12]
        octave = (rounded // 12) - 1
        return f"{pitch_class}{octave}"

    @classmethod
    def hz_to_note_name(cls, freq_hz: float) -> str:
        """Convert frequency directly to closest note name.

        Args:
            freq_hz: Pitch frequency in Hz.

        Returns:
            Closest note string (e.g. 110.0 -> 'A2').
        """
        if freq_hz < 20.0:
            return "N/A"
        midi = cls.hz_to_midi(freq_hz)
        return cls.midi_to_note_name(midi)

    def estimate_pitch_autocorr(
        self,
        audio: np.ndarray,
        min_hz: float = 70.0,
        max_hz: float = 880.0,
    ) -> float:
        """Estimate fundamental pitch F0 using normalized autocorrelation.

        Args:
            audio: 1D audio sample array.
            min_hz: Minimum expected frequency (guitar low E is ~82.4 Hz).
            max_hz: Maximum expected frequency.

        Returns:
            Estimated pitch in Hz, or 0.0 if unvoiced.
        """
        if len(audio) < 256:
            return 0.0

        # Center and window audio
        windowed = (audio - np.mean(audio)) * np.hanning(len(audio))

        # Autocorrelation via FFT
        n = len(windowed)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fx = np.fft.rfft(windowed, n_fft)
        r = np.fft.irfft(fx * np.conj(fx))[:n]

        if r[0] < 1e-7:
            return 0.0
        r /= r[0]

        # Convert frequency range to lag range
        min_lag = int(self.sample_rate / max_hz)
        max_lag = int(self.sample_rate / min_hz)
        min_lag = max(1, min_lag)
        max_lag = min(len(r) - 1, max_lag)

        if min_lag >= max_lag:
            return 0.0

        search_slice = r[min_lag : max_lag + 1]
        peak_idx = int(np.argmax(search_slice))
        peak_val = search_slice[peak_idx]

        # Voicing confidence check
        if peak_val < 0.35:
            return 0.0

        best_lag = float(min_lag + peak_idx)

        # Parabolic interpolation
        if 0 < peak_idx < len(search_slice) - 1:
            alpha = search_slice[peak_idx - 1]
            beta = search_slice[peak_idx]
            gamma = search_slice[peak_idx + 1]
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-8:
                best_lag += 0.5 * (alpha - gamma) / denom

        freq = float(self.sample_rate / best_lag)
        return round(freq, 2)

    def compute_chroma(self, audio: np.ndarray, n_fft: int = 2048) -> np.ndarray:
        """Extract 12-dimensional pitch-class chroma energy vector.

        Args:
            audio: 1D audio waveform.
            n_fft: FFT window size.

        Returns:
            1D float32 array of shape (12,) normalized to sum to 1.0.
        """
        chroma = np.zeros(12, dtype=np.float32)
        if len(audio) < n_fft // 2:
            return chroma

        window = np.hanning(min(len(audio), n_fft))
        segment = audio[: len(window)] * window
        spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # Exclude sub-bass and ultra-high noise
        valid = (freqs >= 65.0) & (freqs <= 3500.0)
        valid_freqs = freqs[valid]
        valid_mag = spectrum[valid]

        # Accumulate magnitudes into 12 semitone chroma bins
        for freq, mag in zip(valid_freqs, valid_mag):
            midi = self.hz_to_midi(freq)
            semitone_bin = int(round(midi)) % 12
            chroma[semitone_bin] += mag

        total = np.sum(chroma)
        if total > 1e-6:
            chroma /= total

        return chroma

    def detect_dominant_pitch_class(self, audio: np.ndarray) -> str:
        """Detect dominant musical pitch class ('C', 'C#', ..., 'B') from audio.

        Args:
            audio: 1D audio waveform.

        Returns:
            Dominant note name.
        """
        chroma = self.compute_chroma(audio)
        best_bin = int(np.argmax(chroma))
        return self.NOTE_NAMES[best_bin]
