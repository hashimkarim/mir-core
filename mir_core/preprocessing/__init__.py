"""
Audio preprocessing utilities for beat tracking.

Provides spectrogram computation and feature extraction.

Classes:
    PreProcessor         — madmom SequentialProcessor producing 81-dim log-mel
                           spectrograms at 100 FPS (for BockTCN).
    BeatNetPreProcessor  — Official BeatNet LOG_SPECT pipeline producing 272-dim
                           features (136 filterbank + 136 spectral diff) at 50 FPS.
    BeatNetPlusPreProcessor — Official BeatNet+ LOG_SPECT pipeline producing
                           288-dim features at 50 FPS.
    SimpleMelPreProcessor — Lightweight librosa mel-spectrogram (for quick experiments).
    BeastPreProcessor    — BEAST linear mel-spectrogram pipeline.
    SpecTNTPreProcessor  — SpecTNT harmonic-STFT feature pipeline.

Functions:
    get_preprocessor_for_model — instantiate the canonical frontend for a model.
    infer_tempo   — estimate BPM from beat times via histogram peak picking.
    pad_features  — repeat-pad first/last frames (used by BeatDataset).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .utils import (
    FPS,
    NUM_BANDS,
    FFT_SIZE,
    MASK_VALUE,
    BEATNET_SAMPLE_RATE,
    BEATNET_HOP_LENGTH,
    BEATNET_WIN_LENGTH,
    BEATNET_N_BANDS,
    BEATNET_FEATURE_DIM,
    BEATNET_PLUS_SAMPLE_RATE,
    BEATNET_PLUS_HOP_LENGTH,
    BEATNET_PLUS_WIN_LENGTH,
    BEATNET_PLUS_N_BANDS,
    BEATNET_PLUS_FEATURE_DIM,
    BEAST_SAMPLE_RATE,
    BEAST_N_FFT,
    BEAST_HOP_LENGTH,
    BEAST_N_MELS,
    BEAST_FMIN,
    BEAST_FMAX,
    BEAST_FEATURE_DIM,
    SPECTNT_SAMPLE_RATE,
    SPECTNT_N_FFT,
    SPECTNT_HOP_LENGTH,
    SPECTNT_N_HARMONIC,
    SPECTNT_SEMITONE_SCALE,
    SPECTNT_N_FREQUENCIES,
    infer_tempo,
    pad_features,
)


_LAZY_EXPORTS = {
    "PreProcessor": (".madmom_features", "PreProcessor"),
    "BeatNetPreProcessor": (".madmom_features", "BeatNetPreProcessor"),
    "BeatNetPlusPreProcessor": (".madmom_features", "BeatNetPlusPreProcessor"),
    "SimpleMelPreProcessor": (".mel_features", "SimpleMelPreProcessor"),
    "BeastPreProcessor": (".mel_features", "BeastPreProcessor"),
    "SpecTNTPreProcessor": (".harmonic_features", "SpecTNTPreProcessor"),
    "PREPROCESSOR_BY_MODEL": (".registry", "PREPROCESSOR_BY_MODEL"),
    "get_preprocessor_for_model": (".registry", "get_preprocessor_for_model"),
}


def __getattr__(name: str) -> Any:
    """Load optional/heavy frontend implementations only when requested."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    # Constants
    "FPS",
    "NUM_BANDS",
    "FFT_SIZE",
    "MASK_VALUE",
    "BEATNET_SAMPLE_RATE",
    "BEATNET_HOP_LENGTH",
    "BEATNET_WIN_LENGTH",
    "BEATNET_N_BANDS",
    "BEATNET_FEATURE_DIM",
    "BEATNET_PLUS_SAMPLE_RATE",
    "BEATNET_PLUS_HOP_LENGTH",
    "BEATNET_PLUS_WIN_LENGTH",
    "BEATNET_PLUS_N_BANDS",
    "BEATNET_PLUS_FEATURE_DIM",
    "BEAST_SAMPLE_RATE",
    "BEAST_N_FFT",
    "BEAST_HOP_LENGTH",
    "BEAST_N_MELS",
    "BEAST_FMIN",
    "BEAST_FMAX",
    "BEAST_FEATURE_DIM",
    "SPECTNT_SAMPLE_RATE",
    "SPECTNT_N_FFT",
    "SPECTNT_HOP_LENGTH",
    "SPECTNT_N_HARMONIC",
    "SPECTNT_SEMITONE_SCALE",
    "SPECTNT_N_FREQUENCIES",
    # Classes
    "PreProcessor",
    "BeatNetPreProcessor",
    "BeatNetPlusPreProcessor",
    "SimpleMelPreProcessor",
    "BeastPreProcessor",
    "SpecTNTPreProcessor",
    "PREPROCESSOR_BY_MODEL",
    "get_preprocessor_for_model",
    # Functions
    "infer_tempo",
    "pad_features",
]
