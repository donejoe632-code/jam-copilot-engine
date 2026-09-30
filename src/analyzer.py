"""Audio DSP module: RMS dynamics, onset detection, and autocorrelation BPM tracking."""

from typing import List, Tuple
import numpy as np
from scipy import signal


class AudioAnalyzer:
    """Real-time and batch DSP analyzer for guitar and musical audio."""

    def __init__(self, sample_rate: int = 22050):
        """Initialize the analyzer with a target sample rate.

        Args:
            sample_rate: Audio sampling frequency in Hz.
        """
        self.sample_rate = sample_rate

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """Compute Root Mean Square (RMS) energy.

        Args:
            audio: 1D audio sample array.

        Returns:
            RMS energy value.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    @staticmethod
    def compute_peak(audio: np.ndarray) -> float:
        """Compute absolute peak amplitude.

        Args:
            audio: 1D audio sample array.

        Returns:
            Peak amplitude in [0.0, 1.0].
        """
        if len(audio) == 0:
            return 0.0
        return float(np.max(np.abs(audio)))

    @staticmethod
    def normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize audio waveform to a target peak amplitude.

        Args:
            audio: 1D audio sample array.
            target_peak: Maximum absolute amplitude of the normalized signal.

        Returns:
            Peak-normalized audio array.
        """
        if len(audio) == 0:
            return audio.copy()
        current_peak = np.max(np.abs(audio))
        if current_peak < 1e-7:
            return audio.copy()
        return (audio / current_peak) * target_peak

    def compute_onset_envelope(
        self,
        audio: np.ndarray,
        hop_size: int = 256,
        frame_size: int = 512,
    ) -> Tuple[np.ndarray, float]:
        """Extract half-wave rectified short-time energy onset envelope.

        Args:
            audio: 1D audio waveform.
            hop_size: Hop size in samples.
            frame_size: Window frame size in samples.

        Returns:
            Tuple of (onset_envelope_array, envelope_sampling_rate).
        """
        if len(audio) < frame_size:
            return np.zeros(1, dtype=np.float32), float(self.sample_rate)

        # Strided window framing
        num_frames = 1 + (len(audio) - frame_size) // hop_size
        shape = (num_frames, frame_size)
        strides = (audio.strides[0] * hop_size, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        # Windowing and short-time energy
        window = np.hanning(frame_size).astype(np.float32)
        windowed = frames * window
        energy = np.sum(np.square(windowed), axis=1)

        # First-order positive difference (half-wave rectification)
        diff = np.diff(energy, prepend=energy[0])
        onset_envelope = np.maximum(0.0, diff)

        # Envelope sampling rate
        env_fs = self.sample_rate / hop_size
        return onset_envelope.astype(np.float32), float(env_fs)

    def detect_onsets(
        self,
        audio: np.ndarray,
        threshold_factor: float = 1.3,
        hop_size: int = 256,
    ) -> List[float]:
        """Detect transient attack onset timestamps in seconds.

        Args:
            audio: 1D audio waveform.
            threshold_factor: Multiplier above moving median for peak picking.
            hop_size: Hop size in samples.

        Returns:
            List of onset times in seconds.
        """
        envelope, env_fs = self.compute_onset_envelope(audio, hop_size=hop_size)
        if len(envelope) < 4:
            return []

        # Adaptive threshold based on moving median
        kernel_size = min(31, len(envelope) if len(envelope) % 2 != 0 else len(envelope) - 1)
        if kernel_size >= 3:
            baseline = signal.medfilt(envelope, kernel_size=kernel_size)
        else:
            baseline = np.median(envelope) * np.ones_like(envelope)

        threshold = baseline * threshold_factor + 1e-4

        # Find local peaks exceeding threshold
        min_distance = max(1, int(0.08 * env_fs))  # minimum 80ms between distinct notes
        peaks, _ = signal.find_peaks(envelope, height=threshold, distance=min_distance)

        onset_timestamps = [float(p / env_fs) for p in peaks]
        return onset_timestamps

    def estimate_bpm(
        self,
        audio: np.ndarray,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        hop_size: int = 256,
    ) -> Tuple[float, float]:
        """Estimate rhythmic tempo (BPM) using autocorrelation of onset strength.

        Args:
            audio: 1D audio waveform (ideally >= 2 seconds).
            min_bpm: Lower bound for BPM detection.
            max_bpm: Upper bound for BPM detection.
            hop_size: Hop size in samples.

        Returns:
            Tuple of (estimated_bpm, confidence_score [0.0, 1.0]).
        """
        envelope, env_fs = self.compute_onset_envelope(audio, hop_size=hop_size)
        if len(envelope) < 16:
            return 120.0, 0.0

        # Mean subtraction and normalization
        norm_env = envelope - np.mean(envelope)
        norm_factor = np.linalg.norm(norm_env)
        if norm_factor < 1e-6:
            return 120.0, 0.0
        norm_env /= norm_factor

        # Autocorrelation
        autocorr = signal.correlate(norm_env, norm_env, mode="full", method="auto")
        center = len(autocorr) // 2
        r = autocorr[center:]

        # Map BPM range to lag indices: lag_sec = 60 / BPM -> lag_idx = lag_sec * env_fs
        min_lag = int(np.floor((60.0 / max_bpm) * env_fs))
        max_lag = int(np.ceil((60.0 / min_bpm) * env_fs))

        min_lag = max(1, min_lag)
        max_lag = min(len(r) - 1, max_lag)

        if min_lag >= max_lag:
            return 120.0, 0.0

        search_band = r[min_lag : max_lag + 1]
        best_rel_idx = int(np.argmax(search_band))
        best_lag = float(min_lag + best_rel_idx)
        peak_val = float(search_band[best_rel_idx])

        # Parabolic interpolation for sub-sample precision
        if 0 < best_rel_idx < len(search_band) - 1:
            y_left = search_band[best_rel_idx - 1]
            y_mid = search_band[best_rel_idx]
            y_right = search_band[best_rel_idx + 1]
            denom = y_left - 2 * y_mid + y_right
            if abs(denom) > 1e-8:
                delta = 0.5 * (y_left - y_right) / denom
                best_lag += float(delta)

        if best_lag <= 0.0:
            return 120.0, 0.0

        estimated_bpm = float(np.clip((60.0 * env_fs) / best_lag, min_bpm, max_bpm))

        # Confidence metric: peak prominence relative to band noise
        mean_val = float(np.mean(search_band))
        std_val = float(np.std(search_band)) + 1e-6
        confidence = float(np.clip((peak_val - mean_val) / (2.5 * std_val), 0.0, 1.0))

        return round(estimated_bpm, 1), round(confidence, 3)
