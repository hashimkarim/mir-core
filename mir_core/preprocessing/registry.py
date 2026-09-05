"""Model-to-preprocessor contract helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable


def _lazy_constructor(module_name: str, class_name: str) -> Callable[..., object]:
    def construct(*args: Any, **kwargs: Any) -> object:
        module = import_module(module_name, __package__)
        return getattr(module, class_name)(*args, **kwargs)

    construct.__name__ = class_name
    return construct


PREPROCESSOR_BY_MODEL: dict[str, Callable[..., object]] = {
    "bock_tcn": _lazy_constructor(".madmom_features", "PreProcessor"),
    "bocktcn": _lazy_constructor(".madmom_features", "PreProcessor"),
    "beatnet": _lazy_constructor(".madmom_features", "BeatNetPreProcessor"),
    "beatnet_crnn": _lazy_constructor(".madmom_features", "BeatNetPreProcessor"),
    "multihead_beatnet": _lazy_constructor(".madmom_features", "BeatNetPreProcessor"),
    "beatnet_plus": _lazy_constructor(".madmom_features", "BeatNetPlusPreProcessor"),
    "beatnet+": _lazy_constructor(".madmom_features", "BeatNetPlusPreProcessor"),
    "beatnet-plus": _lazy_constructor(".madmom_features", "BeatNetPlusPreProcessor"),
    "beast": _lazy_constructor(".mel_features", "BeastPreProcessor"),
    "spectnt": _lazy_constructor(".harmonic_features", "SpecTNTPreProcessor"),
}


def get_preprocessor_for_model(model_name: str, **kwargs):
    """
    Instantiate the canonical preprocessor for a supported model.

    Preprocessing is part of the architecture contract for the beat-tracking
    models. This helper keeps model/front-end pairing explicit at call sites.
    """
    key = model_name.lower().replace("-", "_")
    if key not in PREPROCESSOR_BY_MODEL:
        supported = ", ".join(sorted(PREPROCESSOR_BY_MODEL))
        raise ValueError(
            f"No canonical preprocessor registered for {model_name!r}. "
            f"Supported model names: {supported}."
        )
    return PREPROCESSOR_BY_MODEL[key](**kwargs)
