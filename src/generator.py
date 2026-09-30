"""Procedural accompaniment engine: drums, percussion, and dynamic bass lines."""

from typing import Dict, List, Optional
import numpy as np


class AccompanimentGenerator:
    """Zero-dependency procedural audio accompaniment synthesis engine."""

    # Note name to standard octave 2 bass frequency mapping
    NOTE_FREQUENCIES: Dict[str, float] = {
        "C": 65.41,
        "C#": 69.30,
        "Db": 69.30,
        "D": 73.42,
        "D#": 77.78,
        "Eb": 77.78,
        "E": 82.41,
        "F": 87.31,
        "F#": 92.50,
        "Gb": 92.50,
        "G": 98.00,
        "G#": 103.83,
        "Ab": 103.83,
        "A": 110.00,
        "A#": 116.54,
        "Bb": 116.54,
        "B": 123.47,
    }

    def __init__(self, sample_rate: int = 22050):
        """Initialize generator with sampling rate.

        Args:
            sample_rate: Audio sampling frequency in Hz.
        """
        self.sample_rate = sample_rate

    def synthesize_kick(self, duration_sec: float = 0.22) -> np.ndarray:
        """Synthesize punchy pitch-decayed sub kick drum.

        Args:
            duration_sec: Duration of kick transient in seconds.

        Returns:
            1D float32 audio array.
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        pitch_curve = 135.0 * np.exp(-t * 30.0) + 45.0
        phase = 2.0 * np.pi * np.cumsum(pitch_curve) / self.sample_rate
        amp_decay = np.exp(-t * 22.0)
        kick = np.sin(phase) * amp_decay
        return kick.astype(np.float32)

    def synthesize_snare(self, duration_sec: float = 0.25) -> np.ndarray:
        """Synthesize snappy snare drum with body and wire rattle.

        Args:
            duration_sec: Snare duration in seconds.

        Returns:
            1D float32 audio array.
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Tone body (180 Hz)
        body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 25.0)

        # Noise wire snap
        noise = np.random.uniform(-1.0, 1.0, n_samples) * np.exp(-t * 28.0)
        snare = 0.4 * body + 0.6 * noise
        return snare.astype(np.float32)

    def synthesize_hihat(self, duration_sec: float = 0.08) -> np.ndarray:
        """Synthesize crisp closed hi-hat.

        Args:
            duration_sec: Hi-hat duration in seconds.

        Returns:
            1D float32 audio array.
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, n_samples)
        # Emulate high-pass via difference
        hihat = np.diff(noise, prepend=noise[0]) * np.exp(-t * 70.0) * 0.5
        return hihat.astype(np.float32)

    def synthesize_bass_note(
        self,
        freq_hz: float,
        duration_sec: float,
        release_sec: float = 0.05,
    ) -> np.ndarray:
        """Synthesize warm bass tone with fundamental and second harmonic.

        Args:
            freq_hz: Frequency of the root note in Hz.
            duration_sec: Total note duration in seconds.
            release_sec: Gentle release decay envelope time.

        Returns:
            1D float32 audio array.
        """
        n_samples = int(self.sample_rate * duration_sec)
        if n_samples <= 0:
            return np.empty(0, dtype=np.float32)

        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Fundamental + warm 2nd harmonic
        tone = 0.75 * np.sin(2.0 * np.pi * freq_hz * t) + 0.25 * np.sin(2.0 * np.pi * (freq_hz * 2.0) * t)

        # ADSR Envelope
        env = np.ones(n_samples, dtype=np.float32)
        attack_len = min(int(0.01 * self.sample_rate), n_samples // 4)
        if attack_len > 0:
            env[:attack_len] = np.linspace(0, 1, attack_len)

        release_len = min(int(release_sec * self.sample_rate), n_samples // 2)
        if release_len > 0:
            env[-release_len:] = np.linspace(1, 0, release_len)

        return (tone * env).astype(np.float32)

    def generate_drum_track(self, bpm: float, duration_sec: float) -> np.ndarray:
        """Generate rhythmic drum pattern (kick on 1 & 3, snare on 2 & 4, 8th-note hihats).

        Args:
            bpm: Tempo in beats per minute.
            duration_sec: Length of drum track in seconds.

        Returns:
            1D float32 audio array of exact target length.
        """
        total_samples = int(self.sample_rate * duration_sec)
        drum_track = np.zeros(total_samples, dtype=np.float32)

        sec_per_beat = 60.0 / bpm
        samples_per_beat = int(self.sample_rate * sec_per_beat)

        kick = self.synthesize_kick()
        snare = self.synthesize_snare()
        hihat = self.synthesize_hihat()

        current_sample = 0
        beat_count = 0

        while current_sample < total_samples:
            beat_in_bar = beat_count % 4

            # Kick on beats 0 and 2
            if beat_in_bar in (0, 2):
                end_idx = min(current_sample + len(kick), total_samples)
                drum_track[current_sample:end_idx] += kick[: end_idx - current_sample] * 0.9

            # Snare on backbeats 1 and 3
            elif beat_in_bar in (1, 3):
                end_idx = min(current_sample + len(snare), total_samples)
                drum_track[current_sample:end_idx] += snare[: end_idx - current_sample] * 0.8

            # Hi-hat on downbeat
            end_idx = min(current_sample + len(hihat), total_samples)
            drum_track[current_sample:end_idx] += hihat[: end_idx - current_sample] * 0.4

            # 8th-note offbeat hi-hat
            eighth_sample = current_sample + samples_per_beat // 2
            if eighth_sample < total_samples:
                end_idx = min(eighth_sample + len(hihat), total_samples)
                drum_track[eighth_sample:end_idx] += hihat[: end_idx - eighth_sample] * 0.3

            current_sample += samples_per_beat
            beat_count += 1

        return drum_track

    def generate_bass_track(
        self,
        bpm: float,
        duration_sec: float,
        root_note: str = "A",
    ) -> np.ndarray:
        """Generate root-fifth bassline progression locked to tempo grid.

        Args:
            bpm: Tempo in beats per minute.
            duration_sec: Track duration in seconds.
            root_note: Musical root pitch name (e.g. 'A', 'E', 'D', 'C').

        Returns:
            1D float32 audio array of exact target length.
        """
        total_samples = int(self.sample_rate * duration_sec)
        bass_track = np.zeros(total_samples, dtype=np.float32)

        root_freq = self.NOTE_FREQUENCIES.get(root_note.capitalize(), 110.0)
        fifth_freq = root_freq * 1.4983  # Perfect 5th

        sec_per_beat = 60.0 / bpm
        samples_per_beat = int(self.sample_rate * sec_per_beat)

        current_sample = 0
        beat_count = 0

        while current_sample < total_samples:
            beat_in_bar = beat_count % 4
            # Root note on beats 0 and 2, fifth on beat 3
            target_freq = fifth_freq if beat_in_bar == 3 else root_freq

            # Play 75% note duration for staccato articulation
            note_dur = sec_per_beat * 0.75
            note_samples = self.synthesize_bass_note(target_freq, note_dur)

            end_idx = min(current_sample + len(note_samples), total_samples)
            bass_track[current_sample:end_idx] += note_samples[: end_idx - current_sample] * 0.7

            current_sample += samples_per_beat
            beat_count += 1

        return bass_track

    def generate_backing_track(
        self,
        bpm: float,
        duration_sec: float,
        root_note: str = "A",
        drum_level: float = 0.8,
        bass_level: float = 0.7,
    ) -> np.ndarray:
        """Generate combined backing track (drums + bass) with soft limiting.

        Args:
            bpm: Accompaniment tempo in BPM.
            duration_sec: Desired output duration in seconds.
            root_note: Root pitch for bassline.
            drum_level: Master volume weight for drums.
            bass_level: Master volume weight for bass.

        Returns:
            1D float32 audio array normalized without clipping.
        """
        drums = self.generate_drum_track(bpm, duration_sec)
        bass = self.generate_bass_track(bpm, duration_sec, root_note)

        mix = drums * drum_level + bass * bass_level

        # Soft saturation limiter (tanh)
        mix = np.tanh(mix).astype(np.float32)
        return mix

    @staticmethod
    def mix_signals(
        tracks: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Mix multiple audio signals together with aligned length and weights.

        Args:
            tracks: List of 1D audio sample arrays.
            weights: Optional volume scaling factors per track.

        Returns:
            Mixed and peak-safe 1D float32 audio array.
        """
        if not tracks:
            return np.empty(0, dtype=np.float32)

        max_len = max(len(t) for t in tracks)
        if weights is None:
            weights = [1.0] * len(tracks)

        combined = np.zeros(max_len, dtype=np.float32)
        for track, weight in zip(tracks, weights):
            combined[: len(track)] += track * weight

        # Soft clip to prevent harsh digital overflow
        return np.tanh(combined).astype(np.float32)
