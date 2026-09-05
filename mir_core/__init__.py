"""mir-core: shared Python package for MSc thesis MIR project.

Provides model definitions, preprocessing pipelines, dataset loaders,
evaluation metrics, postprocessing (DBN + particle filter), and training
utilities for beat, downbeat, and tempo tracking research.
"""

__version__ = "0.2.0"

from importlib import import_module
from typing import Any

# Re-export top-level API for convenience.
# Prefer direct submodule imports for AI-friendly context efficiency:
#   from mir_core.models.beatnet.crnn import BeatNetCRNN

from .utils.hashing import canonical_json, stable_digest, stable_hash

try:
    from .models import (
        BeatNetCRNN, BeatNetBatch, BeatNetCRNNBatch, MultiHeadBeatNet,
        ResBlock, TCN, BockTCN,
        BEAST,
        GenreClassifier, GenreRouter, GENRE_LABELS,
    )
except ModuleNotFoundError:
    BeatNetCRNN = None
    BeatNetBatch = None
    BeatNetCRNNBatch = None
    MultiHeadBeatNet = None
    ResBlock = None
    TCN = None
    BockTCN = None
    BEAST = None
    GenreClassifier = None
    GenreRouter = None
    GENRE_LABELS = []

try:
    from .hub import load_model, list_models, ModelType, TrainingMethod, ModelSpec
except ModuleNotFoundError:
    load_model = None
    list_models = None
    ModelType = None
    TrainingMethod = None
    ModelSpec = None

_LAZY_EXPORTS = {
    "PreProcessor": (".preprocessing", "PreProcessor"),
    "BeatNetPreProcessor": (".preprocessing", "BeatNetPreProcessor"),
    "BeatNetPlusPreProcessor": (".preprocessing", "BeatNetPlusPreProcessor"),
    "BeastPreProcessor": (".preprocessing", "BeastPreProcessor"),
    "SpecTNTPreProcessor": (".preprocessing", "SpecTNTPreProcessor"),
    "FPS": (".preprocessing", "FPS"),
    "NUM_BANDS": (".preprocessing", "NUM_BANDS"),
    "FFT_SIZE": (".preprocessing", "FFT_SIZE"),
    "MASK_VALUE": (".preprocessing", "MASK_VALUE"),
    "DBNBeatTracker": (".postprocessing", "DBNBeatTracker"),
    "ParticleFilterTracker": (".postprocessing", "ParticleFilterTracker"),
    "detect_beats": (".postprocessing", "detect_beats"),
}


def __getattr__(name: str) -> Any:
    """Keep optional DSP/decoder dependencies off unrelated import paths."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name, __name__), attribute_name)
    except ModuleNotFoundError:
        # Preserve the historical convenience-API behavior in minimal
        # installations: optional top-level exports resolve to None.
        value = None
    globals()[name] = value
    return value
