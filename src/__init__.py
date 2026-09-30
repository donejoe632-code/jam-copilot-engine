"""Jam Copilot Engine - Lightweight DSP & Audio ML prototypes."""

from src.analyzer import AudioAnalyzer
from src.generator import AccompanimentGenerator
from src.transcriber import PitchChromaTranscriber

__version__ = "1.0.0"

__all__ = ["AudioAnalyzer", "AccompanimentGenerator", "PitchChromaTranscriber"]
