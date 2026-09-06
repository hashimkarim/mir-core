"""Portable waveform frontends for the batch beat-model families.

The ABI begins after audio decoding, channel mixing, and resampling: callers
provide finite mono float32 waveforms at the sample rate frozen in the
manifest.  The exported ONNX graph reproduces the canonical madmom, librosa,
or torchaudio feature tensor consumed by the corresponding PyTorch model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
import warnings

import numpy as np

NATIVE_BATCH_FRONTEND_SCHEMA = "mir.native-batch-frontend/v1"
NATIVE_BATCH_FRONTEND_ABI_VERSION = 1
BATCH_FRONTEND_ONNX_OPSET_VERSION = 18
SUPPORTED_BATCH_FRONTENDS = frozenset({"beast", "bocktcn", "spectnt", "mel", "mfcc"})
_MODEL_NAME_ALIASES = {
    "bock_tcn": "bocktcn",
    "spec_tnt": "spectnt",
    "tcn": "bocktcn",
}
_EXPORT_LOCK = threading.Lock()


class UnsupportedNativeBatchFrontendError(ValueError):
    """Raised when a preprocessing setting has no faithful native graph."""


@dataclass(frozen=True, slots=True)
class NativeBatchFrontendArtifact:
    """One validated waveform-to-model-feature ONNX artifact."""

    model_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    cache_hit: bool

    @property
    def model_family(self) -> str:
        return str(self.manifest["model_family"])

    @property
    def artifact_key(self) -> str:
        return str(self.manifest["artifact_key"])


@dataclass(frozen=True, slots=True)
class _PreparedFrontend:
    model_family: str
    graph: Any
    reference: Callable[[np.ndarray], np.ndarray]
    preprocessing_config: Mapping[str, Any]
    input_contract: Mapping[str, Any]
    output_contract: Mapping[str, Any]
    source: Mapping[str, Any]
    constants_sha256: str
    export_example_samples: int
    parity_rtol: float
    parity_atol: float


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_model_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _MODEL_NAME_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_BATCH_FRONTENDS:
        supported = ", ".join(sorted(SUPPORTED_BATCH_FRONTENDS))
        raise ValueError(
            f"native batch frontend export does not support {value!r}; "
            f"supported models: {supported}"
        )
    return normalized


def _checked_keys(
    config: Mapping[str, Any] | None,
    *,
    allowed: set[str],
    model_family: str,
) -> dict[str, Any]:
    values = dict(config or {})
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise UnsupportedNativeBatchFrontendError(
            f"native {model_family} frontend has no audited semantics for "
            f"preprocessing keys: {', '.join(unknown)}"
        )
    return values


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preprocessing.{key} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"preprocessing.{key} must be a positive integer")
    return value


def _finite_float(config: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preprocessing.{key} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"preprocessing.{key} must be finite")
    return value


def _state_sha256(graph: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(graph.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bocktcn_frontend(
    preprocessing_config: Mapping[str, Any] | None,
) -> _PreparedFrontend:
    import madmom
    import torch

    from mir_core.preprocessing.madmom_features import PreProcessor

    values = _checked_keys(
        preprocessing_config,
        allowed={"sample_rate", "frame_size", "num_bands", "fps", "log_add"},
        model_family="BockTCN",
    )
    sample_rate = _positive_int(values, "sample_rate", 44_100)
    frame_size = _positive_int(values, "frame_size", 2_048)
    num_bands = _positive_int(values, "num_bands", 12)
    fps = _positive_int(values, "fps", 100)
    log_add = _finite_float(values, "log_add", 1e-6)
    if sample_rate != 44_100:
        raise UnsupportedNativeBatchFrontendError(
            "BockTCN's canonical madmom SignalProcessor requires 44100 Hz; "
            "resample in the host before invoking the native frontend"
        )
    if frame_size % 2:
        raise UnsupportedNativeBatchFrontendError(
            "native BockTCN framing currently requires an even frame_size"
        )
    hop_float = sample_rate / float(fps)
    hop_size = int(round(hop_float))
    if not math.isclose(hop_float, hop_size, rel_tol=0.0, abs_tol=1e-12):
        raise UnsupportedNativeBatchFrontendError(
            "madmom uses fractional-hop rounding for this BockTCN fps; the ONNX "
            "STFT hop is integer-only, so this configuration is not portable"
        )
    if log_add <= 0.0:
        raise ValueError("preprocessing.log_add must be positive")

    processor = PreProcessor(
        frame_size=frame_size,
        num_bands=num_bands,
        log=np.log,
        add=log_add,
        fps=fps,
    )
    signal = processor.processors[0](
        np.zeros(max(frame_size, hop_size * 2), dtype=np.float32)
    )
    frames = processor.processors[1](signal)
    stft = processor.processors[2](frames)
    filtered = processor.processors[3](stft)
    window = torch.from_numpy(np.asarray(stft.fft_window, dtype=np.float32))
    filterbank = torch.from_numpy(np.asarray(filtered.filterbank, dtype=np.float32))
    feature_dim = int(filterbank.shape[1])

    class BockTCNFrontendGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", window)
            self.register_buffer("filterbank", filterbank)

        def forward(self, waveform: Any) -> Any:
            spectrum = torch.stft(
                waveform,
                n_fft=frame_size,
                hop_length=hop_size,
                win_length=frame_size,
                window=self.window,
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=False,
            )
            frame_count = (waveform.shape[-1] + hop_size - 1) // hop_size
            magnitude = (
                spectrum[:, :-1, :frame_count, :]
                .square()
                .sum(dim=-1)
                .sqrt()
                .transpose(1, 2)
            )
            features = torch.log(torch.matmul(magnitude, self.filterbank) + log_add)
            return features.unsqueeze(1)

    graph = BockTCNFrontendGraph().eval()

    def reference(waveforms: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(processor(waveform), dtype=np.float32)
                for waveform in waveforms
            ]
        )[:, np.newaxis, :, :]

    resolved_config = {
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "hop_size": hop_size,
        "fps": fps,
        "num_bands_per_octave": num_bands,
        "fmin": 30.0,
        "fmax": 17_000.0,
        "fref": 440.0,
        "norm_filters": True,
        "unique_filters": True,
        "window": "numpy.hanning_symmetric",
        "centered_frames": True,
        "end": "normal",
        "include_nyquist": False,
        "spectrum": "magnitude",
        "log": "natural",
        "log_add": log_add,
    }
    input_contract = {
        "name": "waveform",
        "dtype": "float32",
        "shape": ["batch", "samples"],
        "sample_rate": sample_rate,
        "channels": 1,
        "minimum_samples": 1,
        "host_preprocessing": ["decode", "downmix_to_mono", "resample_to_44100"],
    }
    output_contract = {
        "name": "features",
        "dtype": "float32",
        "shape": ["batch", 1, "frames", feature_dim],
        "layout": "batch_channel_time_frequency",
        "feature_dim": feature_dim,
        "frame_count": f"ceil(samples/{hop_size})",
        "frame_reference": "centered_on_k_times_hop_with_zero_padding",
    }
    source = {
        "reference_implementation": "mir_core.preprocessing.PreProcessor",
        "feature_library": "madmom",
        "madmom_version": str(getattr(madmom, "__version__", "unknown")),
        "numpy_version": str(np.__version__),
    }
    return _PreparedFrontend(
        model_family="bocktcn",
        graph=graph,
        reference=reference,
        preprocessing_config=resolved_config,
        input_contract=input_contract,
        output_contract=output_contract,
        source=source,
        constants_sha256=_state_sha256(graph),
        export_example_samples=hop_size * 16,
        parity_rtol=2e-5,
        parity_atol=1e-5,
    )


def _beast_frontend(
    preprocessing_config: Mapping[str, Any] | None,
) -> _PreparedFrontend:
    import librosa
    import torch

    from mir_core.preprocessing.mel_features import BeastPreProcessor

    values = _checked_keys(
        preprocessing_config,
        allowed={"sample_rate", "n_fft", "hop_length", "n_mels", "fmin", "fmax"},
        model_family="BEAST",
    )
    sample_rate = _positive_int(values, "sample_rate", 44_100)
    n_fft = _positive_int(values, "n_fft", 4_096)
    hop_length = _positive_int(values, "hop_length", 1_024)
    n_mels = _positive_int(values, "n_mels", 128)
    fmin = _finite_float(values, "fmin", 30.0)
    fmax = _finite_float(values, "fmax", 11_000.0)
    if n_fft % 2 or fmin < 0.0 or fmax <= fmin or fmax > sample_rate / 2.0:
        raise ValueError("invalid BEAST FFT or mel-frequency configuration")

    processor = BeastPreProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_filterbank = torch.from_numpy(
        librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )
    )
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)

    class BEASTFrontendGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", window)
            self.register_buffer("mel_filterbank", mel_filterbank)

        def forward(self, waveform: Any) -> Any:
            spectrum = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
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
            decibels = 10.0 * torch.log10(torch.clamp_min(mel, 1e-10))
            reference = 10.0 * torch.log10(
                torch.clamp_min(mel.amax(dim=(-2, -1), keepdim=True), 1e-10)
            )
            decibels = decibels - reference
            decibels = torch.maximum(
                decibels,
                decibels.amax(dim=(-2, -1), keepdim=True) - 80.0,
            )
            return decibels.transpose(1, 2)

    graph = BEASTFrontendGraph().eval()

    def reference(waveforms: np.ndarray) -> np.ndarray:
        return np.stack(
            [processor(waveform, sample_rate) for waveform in waveforms]
        ).astype(np.float32, copy=False)

    resolved_config = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "win_length": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "window": "scipy.hann_periodic",
        "center": True,
        "pad_mode": "constant",
        "spectrum": "power_2",
        "mel_scale": "slaney",
        "mel_norm": "slaney",
        "power_to_db": {
            "ref": "per_waveform_global_max",
            "amin": 1e-10,
            "top_db": 80.0,
        },
    }
    input_contract = {
        "name": "waveform",
        "dtype": "float32",
        "shape": ["batch", "samples"],
        "sample_rate": sample_rate,
        "channels": 1,
        "minimum_samples": 1,
        "host_preprocessing": [
            "decode",
            "downmix_to_mono",
            f"resample_to_{sample_rate}",
        ],
    }
    output_contract = {
        "name": "features",
        "dtype": "float32",
        "shape": ["batch", "frames", n_mels],
        "layout": "batch_time_frequency",
        "feature_dim": n_mels,
        "frame_count": f"floor(samples/{hop_length})+1",
        "normalization_scope": "entire_waveform_per_batch_item",
        "streaming_safe": False,
        "streaming_blocker": (
            "librosa.power_to_db(ref=np.max) depends on the maximum mel power "
            "over the complete waveform"
        ),
    }
    source = {
        "reference_implementation": "mir_core.preprocessing.BeastPreProcessor",
        "feature_library": "librosa",
        "librosa_version": str(librosa.__version__),
        "numpy_version": str(np.__version__),
    }
    return _PreparedFrontend(
        model_family="beast",
        graph=graph,
        reference=reference,
        preprocessing_config=resolved_config,
        input_contract=input_contract,
        output_contract=output_contract,
        source=source,
        constants_sha256=_state_sha256(graph),
        export_example_samples=n_fft * 4,
        parity_rtol=2e-5,
        parity_atol=3e-5,
    )


def _checkpoint_bw_q(path: Path) -> tuple[float, dict[str, Any]]:
    import torch

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise UnsupportedNativeBatchFrontendError(
            "SpecTNT preprocessing checkpoint does not contain a state mapping"
        )
    candidates = [
        value
        for key, value in state.items()
        if str(key).removeprefix("module.") == "hstft.bw_Q"
    ]
    if len(candidates) != 1:
        raise UnsupportedNativeBatchFrontendError(
            "SpecTNT preprocessing checkpoint must contain exactly one hstft.bw_Q"
        )
    tensor = torch.as_tensor(candidates[0], dtype=torch.float32).reshape(-1)
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("SpecTNT hstft.bw_Q must be one finite scalar")
    return float(tensor.item()), {
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "state_key": "hstft.bw_Q",
    }


def _spectnt_frontend(
    preprocessing_config: Mapping[str, Any] | None,
) -> _PreparedFrontend:
    import librosa
    import torch
    import torchaudio

    from mir_core.preprocessing.harmonic_features import SpecTNTPreProcessor

    values = _checked_keys(
        preprocessing_config,
        allowed={
            "sample_rate",
            "n_fft",
            "hop_length",
            "n_harmonic",
            "semitone_scale",
            "bw_q",
            "checkpoint",
        },
        model_family="SpecTNT",
    )
    sample_rate = _positive_int(values, "sample_rate", 16_000)
    n_fft = _positive_int(values, "n_fft", 512)
    hop_length = _positive_int(values, "hop_length", 256)
    n_harmonic = _positive_int(values, "n_harmonic", 6)
    semitone_scale = _positive_int(values, "semitone_scale", 2)
    checkpoint_record = None
    if values.get("checkpoint") not in {None, ""}:
        if "bw_q" in values:
            raise ValueError(
                "SpecTNT preprocessing must declare either bw_q or checkpoint, not both"
            )
        bw_q, checkpoint_record = _checkpoint_bw_q(Path(str(values["checkpoint"])))
    else:
        bw_q = _finite_float(values, "bw_q", 1.0)
    if n_fft % 2 or bw_q <= 0.0:
        raise ValueError("SpecTNT n_fft must be even and bw_q must be positive")

    processor = SpecTNTPreProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_harmonic=n_harmonic,
        semitone_scale=semitone_scale,
        bw_q=bw_q,
        device="cpu",
    )
    filterbank = processor._harmonic_filterbank(n_fft // 2 + 1).detach().cpu()
    window = processor.spec.window.detach().cpu()
    frequency_bins = int(processor.level)

    class SpecTNTFrontendGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", window)
            self.register_buffer("harmonic_filterbank", filterbank)

        def forward(self, waveform: Any) -> Any:
            spectrum = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=False,
            )
            power = spectrum.square().sum(dim=-1)
            harmonic = torch.matmul(
                power.transpose(1, 2),
                self.harmonic_filterbank,
            ).transpose(1, 2)
            harmonic = harmonic.reshape(
                waveform.shape[0],
                n_harmonic,
                frequency_bins,
                power.shape[-1],
            )
            return 10.0 * torch.log10(torch.clamp_min(harmonic, 1e-10))

    graph = SpecTNTFrontendGraph().eval()

    def reference(waveforms: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.asarray(
                processor.process_tensor(torch.from_numpy(waveforms)).cpu().numpy(),
                dtype=np.float32,
            )

    resolved_config = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "win_length": n_fft,
        "hop_length": hop_length,
        "n_harmonic": n_harmonic,
        "semitone_scale": semitone_scale,
        "frequency_bins": frequency_bins,
        "bw_q": bw_q,
        "bw_alpha": float(processor.bw_alpha),
        "bw_beta": float(processor.bw_beta),
        "window": "torch.hann_periodic",
        "center": True,
        "pad_mode": "reflect",
        "spectrum": "power_2",
        "amplitude_to_db": {
            "multiplier": 10.0,
            "amin": 1e-10,
            "reference": 1.0,
            "top_db": None,
        },
    }
    input_contract = {
        "name": "waveform",
        "dtype": "float32",
        "shape": ["batch", "samples"],
        "sample_rate": sample_rate,
        "channels": 1,
        "minimum_samples": n_fft // 2 + 1,
        "host_preprocessing": [
            "decode",
            "downmix_to_mono",
            f"resample_to_{sample_rate}",
        ],
    }
    output_contract = {
        "name": "features",
        "dtype": "float32",
        "shape": ["batch", n_harmonic, frequency_bins, "frames"],
        "layout": "batch_harmonic_frequency_time",
        "harmonic_channels": n_harmonic,
        "feature_dim": frequency_bins,
        "frame_count": f"floor(samples/{hop_length})+1",
    }
    source: dict[str, Any] = {
        "reference_implementation": "mir_core.preprocessing.SpecTNTPreProcessor",
        "feature_library": "torchaudio",
        "torchaudio_version": str(torchaudio.__version__),
        "librosa_version": str(librosa.__version__),
        "numpy_version": str(np.__version__),
    }
    if checkpoint_record is not None:
        source["preprocessing_checkpoint"] = checkpoint_record
    return _PreparedFrontend(
        model_family="spectnt",
        graph=graph,
        reference=reference,
        preprocessing_config=resolved_config,
        input_contract=input_contract,
        output_contract=output_contract,
        source=source,
        constants_sha256=_state_sha256(graph),
        export_example_samples=max(n_fft, hop_length * 16),
        parity_rtol=2e-5,
        parity_atol=1e-4,
    )


def _prepare_frontend(
    model_name: str,
    preprocessing_config: Mapping[str, Any] | None,
) -> _PreparedFrontend:
    model_family = _normalized_model_name(model_name)
    if model_family in {"mel", "mfcc"}:
        from .classifier_frontend import prepare_classifier_frontend

        return prepare_classifier_frontend(model_family, preprocessing_config)
    if model_family == "bocktcn":
        return _bocktcn_frontend(preprocessing_config)
    if model_family == "beast":
        return _beast_frontend(preprocessing_config)
    return _spectnt_frontend(preprocessing_config)


def _identity(prepared: _PreparedFrontend, torch_version: str) -> dict[str, Any]:
    config_digest = _sha256_bytes(
        _canonical_json(dict(prepared.preprocessing_config)).encode("utf-8")
    )
    source_digest = _sha256_bytes(
        _canonical_json(dict(prepared.source)).encode("utf-8")
    )
    return {
        "abi_version": NATIVE_BATCH_FRONTEND_ABI_VERSION,
        "model_family": prepared.model_family,
        "preprocessing_config_sha256": config_digest,
        "frontend_constants_sha256": prepared.constants_sha256,
        "source_contract_sha256": source_digest,
        "onnx_opset": BATCH_FRONTEND_ONNX_OPSET_VERSION,
        "torch_version": torch_version,
        "export_example_samples": prepared.export_example_samples,
    }


def _artifact_key(identity: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(dict(identity)).encode("utf-8"))[:24]


def _parity_probes(
    prepared: _PreparedFrontend,
) -> tuple[tuple[str, np.ndarray], ...]:
    samples = int(prepared.export_example_samples)
    hop = int(
        prepared.preprocessing_config["hop_size"]
        if prepared.model_family == "bocktcn"
        else prepared.preprocessing_config["hop_length"]
    )
    alternate_samples = samples + hop + 17
    generator = np.random.default_rng(0x42415443484645)
    impulse_generator = np.random.default_rng(0x46524F4E54454E44)
    structured_generator = np.random.default_rng(0x5354525543545552)
    impulse_train = np.zeros(samples, dtype=np.float32)
    impulse_indices = np.arange(0, samples, 137)
    impulse_train[impulse_indices] = impulse_generator.uniform(
        -0.4,
        0.4,
        impulse_indices.size,
    ).astype(np.float32)
    seconds = np.arange(alternate_samples, dtype=np.float64) / float(
        prepared.input_contract["sample_rate"]
    )
    two_tone = (
        0.17 * np.sin(2.0 * np.pi * 173.0 * seconds)
        + 0.09 * np.sin(2.0 * np.pi * 1_237.0 * seconds + 0.31)
        + structured_generator.normal(0.0, 0.02, alternate_samples)
    ).astype(np.float32)
    two_tone[::997] += np.float32(0.07)
    return (
        (
            f"deterministic_impulse_train(samples={samples},batch=1)",
            impulse_train[None, :],
        ),
        (
            f"normal(seed=47786851898437,samples={samples},batch=2)",
            generator.normal(0.0, 0.05, (2, samples)).astype(np.float32),
        ),
        (
            f"two_tone_impulses(samples={alternate_samples},batch=1)",
            two_tone[None, :],
        ),
    )


def _parity_gate(
    prepared: _PreparedFrontend,
    model_path: Path,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("native batch frontend parity requires onnxruntime") from exc

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )
    errors: dict[str, dict[str, float]] = {}
    maximum_error = 0.0
    for probe_name, waveform in _parity_probes(prepared):
        expected = np.asarray(prepared.reference(waveform), dtype=np.float32)
        actual = np.asarray(
            session.run(("features",), {"waveform": waveform})[0],
            dtype=np.float32,
        )
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"native {prepared.model_family} frontend returned {actual.shape}, "
                f"expected {expected.shape}, for {probe_name}"
            )
        difference = np.abs(actual - expected)
        max_error = float(np.max(difference, initial=0.0))
        mean_error = float(np.mean(difference))
        errors[probe_name] = {
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
        }
        maximum_error = max(maximum_error, max_error)
        if not np.allclose(
            actual,
            expected,
            rtol=prepared.parity_rtol,
            atol=prepared.parity_atol,
        ):
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"native {prepared.model_family} frontend parity failed for "
                f"{probe_name}: maximum absolute error {max_error:.9g}"
            )
    return {
        "reference": str(prepared.source["reference_implementation"]),
        "provider": "CPUExecutionProvider",
        "rtol": prepared.parity_rtol,
        "atol": prepared.parity_atol,
        "max_abs_error": maximum_error,
        "per_probe_error": errors,
        "probes": list(errors),
    }


def _export_prepared(
    prepared: _PreparedFrontend,
    destination: Path,
) -> NativeBatchFrontendArtifact:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.linspace(
        -0.5,
        0.5,
        steps=prepared.export_example_samples,
        dtype=torch.float32,
    ).reshape(1, -1)
    output_time_axis = {
        "bocktcn": 2,
        "beast": 1,
        "spectnt": 3,
        "mel": 3,
        "mfcc": 3,
    }[prepared.model_family]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message="stft with return_complex=False is deprecated.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Constant folding - Only steps=1 can be constant folded.*",
                category=UserWarning,
            )
            torch.onnx.export(
                prepared.graph,
                example,
                str(destination),
                input_names=("waveform",),
                output_names=("features",),
                dynamic_axes={
                    "waveform": {0: "batch", 1: "samples"},
                    "features": {0: "batch", output_time_axis: "frames"},
                },
                opset_version=BATCH_FRONTEND_ONNX_OPSET_VERSION,
                dynamo=False,
                do_constant_folding=True,
            )
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "native batch frontend export requires the optional onnx package"
            ) from exc
        onnx.checker.check_model(onnx.load(str(destination)))
        parity = _parity_gate(prepared, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    identity = _identity(prepared, str(torch.__version__))
    config_digest = str(identity["preprocessing_config_sha256"])
    manifest = {
        "schema": NATIVE_BATCH_FRONTEND_SCHEMA,
        "abi_version": NATIVE_BATCH_FRONTEND_ABI_VERSION,
        "artifact_key": _artifact_key(identity),
        "artifact_kind": "batch_audio_frontend",
        "model_family": prepared.model_family,
        "graph_contract": "canonical-rate-mono-waveform-to-model-ready-features",
        "identity": identity,
        "preprocessing_config": dict(prepared.preprocessing_config),
        "preprocessing_config_sha256": config_digest,
        "frontend_constants_sha256": prepared.constants_sha256,
        "source": dict(prepared.source),
        "input_contract": dict(prepared.input_contract),
        "output_contract": dict(prepared.output_contract),
        "inputs": ["waveform"],
        "outputs": ["features"],
        "onnx": {
            "filename": destination.name,
            "opset": BATCH_FRONTEND_ONNX_OPSET_VERSION,
            "sha256": _sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        },
        "exporter": {
            "name": "torch.onnx.legacy",
            "torch_version": str(torch.__version__),
        },
        "parity_validation": parity,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NativeBatchFrontendArtifact(
        model_path=destination,
        manifest_path=manifest_path,
        manifest=manifest,
        cache_hit=False,
    )


def export_batch_frontend_onnx(
    destination: str | Path,
    *,
    model_name: str,
    preprocessing_config: Mapping[str, Any] | None = None,
) -> NativeBatchFrontendArtifact:
    """Export and parity-gate one canonical batch-model audio frontend."""

    prepared = _prepare_frontend(model_name, preprocessing_config)
    return _export_prepared(prepared, Path(destination).expanduser().resolve())


def _cache_root() -> Path:
    configured = os.environ.get("MIR_NATIVE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve() / "batch-frontend"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (
            Path(xdg_cache).expanduser().resolve()
            / "mir"
            / "native-models"
            / "batch-frontend"
        )
    return Path.home() / ".cache" / "mir" / "native-models" / "batch-frontend"


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    resolved = Path(path).expanduser().resolve()
    if resolved.name.endswith(".onnx.json"):
        return resolved.with_suffix(""), resolved
    return resolved, resolved.with_suffix(resolved.suffix + ".json")


def _validated_loaded_artifact(
    model_path: Path,
    manifest_path: Path,
    *,
    expected_key: str | None = None,
) -> NativeBatchFrontendArtifact:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be an object")
        identity = payload["identity"]
        onnx_record = payload["onnx"]
        config = payload["preprocessing_config"]
        source = payload["source"]
        key = _artifact_key(identity)
        config_digest = _sha256_bytes(_canonical_json(config).encode("utf-8"))
        source_digest = _sha256_bytes(_canonical_json(source).encode("utf-8"))
        valid = (
            payload["schema"] == NATIVE_BATCH_FRONTEND_SCHEMA
            and int(payload["abi_version"]) == NATIVE_BATCH_FRONTEND_ABI_VERSION
            and payload["model_family"] in SUPPORTED_BATCH_FRONTENDS
            and identity["model_family"] == payload["model_family"]
            and payload["artifact_key"] == key
            and (expected_key is None or key == expected_key)
            and payload["preprocessing_config_sha256"] == config_digest
            and identity["preprocessing_config_sha256"] == config_digest
            and identity["frontend_constants_sha256"]
            == payload["frontend_constants_sha256"]
            and identity["source_contract_sha256"] == source_digest
            and int(identity["onnx_opset"]) == BATCH_FRONTEND_ONNX_OPSET_VERSION
            and onnx_record["filename"] == model_path.name
            and int(onnx_record["opset"]) == BATCH_FRONTEND_ONNX_OPSET_VERSION
            and int(onnx_record["size_bytes"]) == model_path.stat().st_size
            and onnx_record["sha256"] == _sha256_file(model_path)
        )
        if not valid:
            raise ValueError("manifest identity or model hash is invalid")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid native batch frontend artifact: {exc}") from exc
    return NativeBatchFrontendArtifact(
        model_path=model_path,
        manifest_path=manifest_path,
        manifest=payload,
        cache_hit=True,
    )


def load_batch_frontend_artifact(
    path: str | Path,
) -> NativeBatchFrontendArtifact:
    """Load a distributed frontend without importing its reference library."""

    model_path, manifest_path = _artifact_paths(path)
    return _validated_loaded_artifact(model_path, manifest_path)


def ensure_batch_frontend_onnx(
    *,
    model_name: str,
    preprocessing_config: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> NativeBatchFrontendArtifact:
    """Return a verified cached frontend, exporting it when absent or stale."""

    import torch

    prepared = _prepare_frontend(model_name, preprocessing_config)
    identity = _identity(prepared, str(torch.__version__))
    key = _artifact_key(identity)
    root = (
        _cache_root() if cache_root is None else Path(cache_root).expanduser().resolve()
    )
    directory = root / prepared.model_family / key
    model_path = directory / "frontend.onnx"
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")
    with _EXPORT_LOCK:
        if model_path.is_file() and manifest_path.is_file():
            try:
                return _validated_loaded_artifact(
                    model_path,
                    manifest_path,
                    expected_key=key,
                )
            except ValueError:
                pass
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="export-", dir=directory) as temporary:
            temporary_model = Path(temporary) / model_path.name
            exported = _export_prepared(prepared, temporary_model)
            os.replace(temporary_model, model_path)
            os.replace(exported.manifest_path, manifest_path)
        return NativeBatchFrontendArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=exported.manifest,
            cache_hit=False,
        )


class OnnxBatchFrontendSession:
    """Validated ONNX Runtime host for canonical-rate mono waveforms."""

    def __init__(
        self,
        artifact: NativeBatchFrontendArtifact,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "native batch frontend inference requires onnxruntime"
            ) from exc
        if intra_op_threads < 1:
            raise ValueError("intra_op_threads must be at least one")
        if artifact.manifest["onnx"]["sha256"] != _sha256_file(artifact.model_path):
            raise ValueError("native batch frontend ONNX hash is invalid")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = int(intra_op_threads)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(artifact.model_path),
            sess_options=options,
            providers=("CPUExecutionProvider",),
        )
        if tuple(value.name for value in self._session.get_inputs()) != ("waveform",):
            raise ValueError("native batch frontend graph has unexpected inputs")
        if tuple(value.name for value in self._session.get_outputs()) != ("features",):
            raise ValueError("native batch frontend graph has unexpected outputs")
        self.artifact = artifact
        self.provider = "CPUExecutionProvider"

    def _expected_shape(self, batch: int, samples: int) -> tuple[int, ...]:
        config = self.artifact.manifest["preprocessing_config"]
        family = self.artifact.model_family
        if family == "bocktcn":
            frames = (samples + int(config["hop_size"]) - 1) // int(config["hop_size"])
            return (
                batch,
                1,
                frames,
                int(self.artifact.manifest["output_contract"]["feature_dim"]),
            )
        frames = samples // int(config["hop_length"]) + 1
        if family in {"mel", "mfcc"}:
            frames = (
                int(config["window_samples"] or samples) // int(config["hop_length"])
                + 1
            )
            return (
                batch,
                1,
                int(self.artifact.manifest["output_contract"]["feature_dim"]),
                frames,
            )
        if family == "beast":
            return (
                batch,
                frames,
                int(self.artifact.manifest["output_contract"]["feature_dim"]),
            )
        output = self.artifact.manifest["output_contract"]
        return (
            batch,
            int(output["harmonic_channels"]),
            int(output["feature_dim"]),
            frames,
        )

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        """Extract a model-ready feature batch from equal-length waveforms."""

        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        minimum = int(self.artifact.manifest["input_contract"]["minimum_samples"])
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < minimum:
            raise ValueError(
                "native batch frontend input must be [batch, samples] with at "
                f"least {minimum} samples, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("native batch frontend waveform must be finite")
        contiguous = np.ascontiguousarray(values)
        features = np.asarray(
            self._session.run(("features",), {"waveform": contiguous})[0],
            dtype=np.float32,
        )
        expected = self._expected_shape(values.shape[0], values.shape[1])
        if features.shape != expected or not np.isfinite(features).all():
            raise RuntimeError(
                f"native batch frontend returned {features.shape}, expected {expected}"
            )
        return features


def benchmark_batch_frontend(
    session: OnnxBatchFrontendSession,
    *,
    waveform: np.ndarray | None = None,
    warmup: int = 5,
    iterations: int = 50,
) -> dict[str, Any]:
    """Benchmark model-only frontend inference on one waveform tensor."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    manifest = session.artifact.manifest
    if waveform is None:
        samples = int(manifest["identity"]["export_example_samples"])
        sample_rate = float(manifest["input_contract"]["sample_rate"])
        seconds = np.arange(samples, dtype=np.float64) / sample_rate
        waveform = (0.1 * np.sin(2.0 * np.pi * 440.0 * seconds)).astype(np.float32)
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim not in {1, 2}:
        raise ValueError("benchmark waveform must have rank one or two")
    for _ in range(warmup):
        session.infer(values)
    durations = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        started = time.perf_counter()
        session.infer(values)
        durations[index] = time.perf_counter() - started
    mean_seconds = float(np.mean(durations))
    sample_count = int(values.shape[-1])
    audio_seconds = sample_count / float(manifest["input_contract"]["sample_rate"])
    return {
        "schema": "mir.native-batch-frontend-benchmark/v1",
        "artifact_key": session.artifact.artifact_key,
        "model_family": session.artifact.model_family,
        "runtime": "onnxruntime-cpu",
        "warmup": int(warmup),
        "iterations": int(iterations),
        "mean_ms": mean_seconds * 1000.0,
        "p50_ms": float(np.percentile(durations, 50)) * 1000.0,
        "p95_ms": float(np.percentile(durations, 95)) * 1000.0,
        "max_ms": float(np.max(durations)) * 1000.0,
        "audio_seconds": audio_seconds,
        "compute_realtime_factor": mean_seconds / audio_seconds,
        "timing_scope": "frontend_graph_only_excludes_decode_downmix_and_resample",
    }


__all__ = [
    "BATCH_FRONTEND_ONNX_OPSET_VERSION",
    "NATIVE_BATCH_FRONTEND_ABI_VERSION",
    "NATIVE_BATCH_FRONTEND_SCHEMA",
    "NativeBatchFrontendArtifact",
    "OnnxBatchFrontendSession",
    "SUPPORTED_BATCH_FRONTENDS",
    "UnsupportedNativeBatchFrontendError",
    "benchmark_batch_frontend",
    "ensure_batch_frontend_onnx",
    "export_batch_frontend_onnx",
    "load_batch_frontend_artifact",
]
