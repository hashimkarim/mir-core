"""Portable librosa mel and MFCC frontends for the genre classifiers.

The input boundary is finite mono float32 at the declared sample rate. Window
truncation/padding and optional peak normalization are part of the graph and
its identity. Centered framing and MFCC deltas have finite-window semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .batch_frontend import (
    NativeBatchFrontendArtifact,
    _PreparedFrontend,
    _checked_keys,
    _finite_float,
    _positive_int,
    _state_sha256,
    ensure_batch_frontend_onnx,
    export_batch_frontend_onnx,
)


def prepare_classifier_frontend(
    family: str, config: Mapping[str, Any] | None
) -> _PreparedFrontend:
    import librosa
    import scipy
    import scipy.fft
    import scipy.signal
    import torch
    from torch.nn import functional as functional

    values = _checked_keys(
        config,
        allowed={
            "sample_rate",
            "n_fft",
            "hop_length",
            "n_mels",
            "n_mfcc",
            "fmin",
            "fmax",
            "window_samples",
            "normalize_peak",
        },
        model_family=family,
    )
    if family not in {"mel", "mfcc"}:
        raise ValueError("classifier frontend must be mel or mfcc")
    for key in ("sample_rate", "n_fft", "hop_length", "n_mels", "n_mfcc"):
        if key in values:
            value = values[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or int(value) != value
            ):
                raise ValueError(f"{key} must be a positive integer")
    sample_rate = _positive_int(values, "sample_rate", 22_050)
    n_fft = _positive_int(values, "n_fft", 2_048)
    hop = _positive_int(values, "hop_length", 512)
    bands = _positive_int(values, "n_mels", 128)
    coefficients = _positive_int(values, "n_mfcc", 20)
    fmin = _finite_float(values, "fmin", 0.0)
    fmax = _finite_float(values, "fmax", sample_rate / 2.0)
    window_samples = values.get("window_samples")
    if window_samples is not None:
        if isinstance(window_samples, bool) or int(window_samples) != window_samples:
            raise ValueError("window_samples must be a positive integer or None")
        window_samples = int(window_samples)
        if window_samples < 1:
            raise ValueError("window_samples must be positive")
    normalize_peak = values.get("normalize_peak", False)
    if not isinstance(normalize_peak, bool):
        raise ValueError("normalize_peak must be boolean")
    if n_fft % 2 or not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("invalid classifier FFT or mel-frequency bounds")
    if family == "mfcc" and coefficients > bands:
        raise ValueError("n_mfcc cannot exceed n_mels")
    if family == "mel" and "n_mfcc" in values:
        raise ValueError("n_mfcc applies only to the MFCC frontend")
    minimum = 8 * hop if family == "mfcc" else 1
    if window_samples is not None and window_samples < minimum:
        raise ValueError("MFCC delta interpolation requires at least nine frames")

    window = torch.from_numpy(scipy.signal.get_window("hann", n_fft).astype(np.float32))
    bank = torch.from_numpy(
        librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=bands,
            fmin=fmin,
            fmax=fmax,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )
    )

    class ClassifierFrontendGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", window)
            self.register_buffer("mel_filterbank", bank)
            if family == "mfcc":
                # Freeze the orthonormal DCT and Savitzky-Golay endpoint maps.
                dct = scipy.fft.dct(np.eye(bands), type=2, norm="ortho", axis=0)
                self.register_buffer("dct", torch.from_numpy(dct[:coefficients]))
                for order in (1, 2):
                    edges = scipy.signal.savgol_filter(
                        np.eye(9), 9, order, deriv=order, axis=-1, mode="interp"
                    )
                    kernel = scipy.signal.savgol_coeffs(
                        9, order, deriv=order, use="dot"
                    )
                    self.register_buffer(
                        f"delta_{order}_edges", torch.from_numpy(edges)
                    )
                    self.register_buffer(
                        f"delta_{order}_kernel", torch.from_numpy(kernel[None, None])
                    )

        def delta(self, features: Any, order: int) -> Any:
            values64 = features.to(torch.float64)
            edges = getattr(self, f"delta_{order}_edges")
            kernel = getattr(self, f"delta_{order}_kernel")
            # ONNX CPU providers do not all implement binary64 Conv. Nine
            # explicit taps preserve SciPy's double-precision accumulation.
            interior = values64[..., :-8] * kernel[0, 0, 0]
            for index in range(1, 8):
                interior = (
                    interior + values64[..., index : index - 8] * kernel[0, 0, index]
                )
            interior = interior + values64[..., 8:] * kernel[0, 0, 8]
            left = torch.matmul(values64[..., :9], edges[:, :4])
            right = torch.matmul(values64[..., -9:], edges[:, -4:])
            return torch.cat((left, interior, right), dim=-1).to(torch.float32)

        def forward(self, waveform: Any) -> Any:
            if window_samples is not None:
                waveform = waveform[:, :window_samples]
                waveform = functional.pad(
                    waveform, (0, window_samples - waveform.shape[-1])
                )
            if normalize_peak:
                peak = waveform.abs().amax(dim=-1, keepdim=True)
                waveform = waveform / torch.where(peak > 0, peak, torch.ones_like(peak))
            spectrum = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=self.window,
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=False,
            )
            power = spectrum.square().sum(dim=-1)
            mel = torch.matmul(self.mel_filterbank, power)
            if family == "mel":
                return torch.log(mel + 1e-6).unsqueeze(1)
            decibels = 10.0 * torch.log10(torch.clamp_min(mel, 1e-10))
            decibels = torch.maximum(
                decibels, decibels.amax(dim=(-2, -1), keepdim=True) - 80.0
            )
            mfcc = torch.matmul(self.dct, decibels.to(torch.float64)).to(torch.float32)
            return torch.cat(
                (mfcc, self.delta(mfcc, 1), self.delta(mfcc, 2)), dim=1
            ).unsqueeze(1)

    graph = ClassifierFrontendGraph().eval()

    def reference(waveforms: np.ndarray) -> np.ndarray:
        outputs = []
        for audio in waveforms:
            if window_samples is not None:
                audio = audio[:window_samples]
                audio = np.pad(audio, (0, window_samples - len(audio)))
            if normalize_peak:
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=bands,
                fmin=fmin,
                fmax=fmax,
            )
            if family == "mel":
                features = np.log(mel + 1e-6)
            else:
                mfcc = librosa.feature.mfcc(
                    S=librosa.power_to_db(mel), n_mfcc=coefficients
                )
                features = np.concatenate(
                    (
                        mfcc,
                        librosa.feature.delta(mfcc),
                        librosa.feature.delta(mfcc, order=2),
                    )
                )
            outputs.append(features[None])
        return np.stack(outputs).astype(np.float32)

    resolved = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop,
        "n_mels": bands,
        "n_mfcc": coefficients if family == "mfcc" else None,
        "fmin": fmin,
        "fmax": fmax,
        "window_samples": window_samples,
        "normalize_peak": normalize_peak,
        "window": "scipy.hann_periodic",
        "center": True,
        "pad_mode": "constant",
        "spectrum": "power_2",
        "mel_scale": "slaney",
        "mel_norm": "slaney",
        "log": (
            "natural_add_1e-6"
            if family == "mel"
            else "power_to_db_ref_1_amin_1e-10_top_80"
        ),
        "dct": "type_2_ortho" if family == "mfcc" else None,
        "deltas": "savgol_width_9_order_1_2_interp" if family == "mfcc" else None,
    }
    dimension = 3 * coefficients if family == "mfcc" else bands
    output_contract = {
        "name": "features",
        "dtype": "float32",
        "shape": ["batch", 1, dimension, "frames"],
        "layout": "batch_channel_frequency_time",
        "feature_dim": dimension,
        "frame_count": f"floor({'window_samples' if window_samples else 'samples'}/{hop})+1",
        "normalization_scope": (
            "entire_window_per_batch_item" if normalize_peak else "none"
        ),
        "streaming_safe": False,
        "streaming_blocker": "Centered frames and optional window normalization; MFCC deltas and top_db use complete windows.",
    }
    return _PreparedFrontend(
        model_family=family,
        graph=graph,
        reference=reference,
        preprocessing_config=resolved,
        input_contract={
            "name": "waveform",
            "dtype": "float32",
            "shape": ["batch", "samples"],
            "sample_rate": sample_rate,
            "channels": 1,
            "minimum_samples": 1 if window_samples is not None else minimum,
            "host_preprocessing": [
                "decode",
                "downmix_to_mono",
                f"resample_to_{sample_rate}",
            ],
        },
        output_contract=output_contract,
        source={
            "reference_implementation": f"mir_core.models.classifier.GenreClassifier.preprocess_{'audio' if family == 'mel' else 'mfcc'}",
            "feature_library": "librosa",
            "librosa_version": str(librosa.__version__),
            "scipy_version": str(scipy.__version__),
            "numpy_version": str(np.__version__),
        },
        constants_sha256=_state_sha256(graph),
        export_example_samples=max(n_fft + 17, 12 * hop),
        parity_rtol=2e-5,
        parity_atol=1e-5 if family == "mel" else 1e-4,
    )


def ensure_classifier_frontend_onnx(
    *,
    feature_type: str,
    feature_config: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> NativeBatchFrontendArtifact:
    """Export/reuse an independently validated mel or MFCC audio frontend."""
    if feature_type not in {"mel", "mfcc"}:
        raise ValueError("classifier frontend must be mel or mfcc")
    return ensure_batch_frontend_onnx(
        model_name=feature_type,
        preprocessing_config=feature_config,
        cache_root=cache_root,
    )


def export_classifier_frontend_onnx(
    destination: str | Path,
    *,
    feature_type: str,
    feature_config: Mapping[str, Any] | None = None,
) -> NativeBatchFrontendArtifact:
    """Export a distributable classifier waveform frontend and its manifest."""
    if feature_type not in {"mel", "mfcc"}:
        raise ValueError("classifier frontend must be mel or mfcc")
    return export_batch_frontend_onnx(
        destination,
        model_name=feature_type,
        preprocessing_config=feature_config,
    )
