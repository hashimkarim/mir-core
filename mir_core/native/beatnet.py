"""ONNX export and native recurrent inference for BeatNet-family models.

The ONNX graph owns one neural-network frame only.  LSTM hidden and cell state
are explicit graph inputs/outputs, which keeps the artifact usable by native
C++ and Rust hosts without relying on Python-side mutable model state.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NATIVE_STREAMING_SCHEMA = "mir.native-streaming-model/v1"
NATIVE_STREAMING_ABI_VERSION = 1
NATIVE_STREAMING_PARITY_SCHEMA = "mir.native-streaming-parity-validation/v1"
NATIVE_STREAMING_PARITY_VERSION = 1
ONNX_OPSET_VERSION = 18
EXPORTER_NAME = "torch.onnx.legacy.fp64-ordered-fp32-state-v1"

SUPPORTED_STREAMING_MODELS = frozenset(
    {"beatnet", "beatnet_plus", "dance_beatnet", "multihead_beatnet"}
)
_MODEL_NAME_ALIASES = {
    "beatnet_dance": "dance_beatnet",
    "dancebeatnet": "dance_beatnet",
    "multihead": "multihead_beatnet",
    "multi_head_beatnet": "multihead_beatnet",
}
_STANDARD_OUTPUT_NAMES = ("activations", "next_hidden", "next_cell")
_DANCE_OUTPUT_NAMES = (
    "activations",
    "beats",
    "downbeats",
    "dancebeats",
    "next_hidden",
    "next_cell",
)
_PARITY_RTOL = 2.0e-5
_PARITY_ATOL = 2.0e-6
_PARITY_SEQUENCE_LENGTH = 12
_PARITY_FEATURE_SEED = 0x53545245414D
_EXPORT_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class NativeModelArtifact:
    """One validated native graph and its machine-readable deployment contract."""

    model_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    cache_hit: bool

    @property
    def model_family(self) -> str:
        return str(self.manifest["model_family"])

    @property
    def input_dim(self) -> int:
        return int(self.manifest["streaming_state"]["input_dim"])

    @property
    def hidden_dim(self) -> int:
        return int(self.manifest["streaming_state"]["hidden_dim"])

    @property
    def num_layers(self) -> int:
        return int(self.manifest["streaming_state"]["num_layers"])


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


def _model_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _MODEL_NAME_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_STREAMING_MODELS:
        supported = ", ".join(sorted(SUPPORTED_STREAMING_MODELS))
        raise ValueError(
            f"native streaming export does not support {value!r}; "
            f"supported models: {supported}"
        )
    return normalized


def _deployment_model_config(
    model: Any,
    model_name: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return identity inputs including graph behavior not stored in weights."""

    config = dict(model_config)
    if model_name == "multihead_beatnet":
        labels = tuple(str(label) for label in getattr(model, "genre_labels", ()))
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(
                "MultiHeadBeatNet must declare non-empty, unique genre labels"
            )
        requested = config.get("genre_label", config.get("genre"))
        if requested is None:
            raise ValueError(
                "multihead native export requires model_config.genre_label"
            )
        genre_label = str(requested)
        if genre_label not in labels:
            raise ValueError(
                f"unknown MultiHeadBeatNet genre {genre_label!r}; "
                f"available labels: {', '.join(labels)}"
            )
        config.pop("genre", None)
        config["genre_label"] = genre_label
        return config
    if model_name != "dance_beatnet":
        return config

    tracking_target = getattr(model, "tracking_target", None)
    tracking_target_value = getattr(tracking_target, "value", tracking_target)
    if tracking_target_value not in {"beat", "dance"}:
        raise ValueError(
            "DanceBeatNet tracking_target must be either 'beat' or 'dance'"
        )
    configured_target = config.get("tracking_target")
    if configured_target is not None:
        configured_value = getattr(configured_target, "value", configured_target)
        if str(configured_value) != str(tracking_target_value):
            raise ValueError(
                "model_config tracking_target does not match the loaded "
                "DanceBeatNet model"
            )
    # ``tracking_target`` changes the primary graph output but is not present in
    # a PyTorch state dict. Include the observed value even when an older caller
    # omits it so two semantically different graphs cannot share a cache entry.
    config["tracking_target"] = str(tracking_target_value)
    return config


def _cache_root() -> Path:
    configured = os.environ.get("MIR_NATIVE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "mir" / "native-models"
    return Path.home() / ".cache" / "mir" / "native-models"


def onnxruntime_available() -> bool:
    """Return whether the optional native ONNX Runtime binding is importable."""

    return importlib.util.find_spec("onnxruntime") is not None


def _native_export_available() -> bool:
    return onnxruntime_available() and importlib.util.find_spec("onnx") is not None


def resolve_streaming_backend(
    requested: str,
    *,
    device: str = "auto",
) -> str:
    """Resolve ``auto|torch|onnxruntime`` without surprising explicit CUDA users.

    The first native target is CPU ONNX Runtime.  ``auto`` and explicit ``cpu``
    prefer it when installed; explicit ``cuda`` remains on PyTorch until the
    CUDA execution-provider artifact path has its own parity and jitter gate.
    """

    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "torch", "onnxruntime"}:
        raise ValueError("streaming backend must be auto, torch, or onnxruntime")
    device_name = str(device).strip().lower()
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("streaming device must be auto, cpu, or cuda")
    if normalized == "torch":
        return "torch"
    if normalized == "onnxruntime":
        if not _native_export_available():
            raise RuntimeError(
                "onnx/onnxruntime was requested but is not installed; "
                "install mir-core[native] or select the torch backend"
            )
        if device_name == "cuda":
            raise ValueError(
                "the initial ONNX streaming backend is CPU-only; use device "
                "auto/cpu or select the torch backend for CUDA"
            )
        return "onnxruntime"
    if device_name != "cuda" and _native_export_available():
        return "onnxruntime"
    return "torch"


def _artifact_identity(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    torch_version: str,
) -> tuple[str, str]:
    config_sha256 = _sha256_bytes(_canonical_json(dict(model_config)).encode())
    payload = {
        "abi_version": NATIVE_STREAMING_ABI_VERSION,
        "checkpoint_sha256": str(checkpoint_sha256),
        "exporter": EXPORTER_NAME,
        "model_config_sha256": config_sha256,
        "model_family": model_name,
        "opset": ONNX_OPSET_VERSION,
        "torch_version": torch_version,
    }
    return _sha256_bytes(_canonical_json(payload).encode())[:24], config_sha256


def _streaming_shape(model: Any, model_name: str) -> tuple[int, int, int]:
    input_dim = int(model.input_dim)
    lstm = getattr(model, "lstm", None)
    if lstm is None:
        raise TypeError(f"{model_name} has no LSTM module")
    hidden_dim = int(lstm.hidden_size)
    num_layers = int(lstm.num_layers)
    if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
        raise ValueError("native streaming dimensions must all be positive")
    return input_dim, hidden_dim, num_layers


def _deployment_model(
    model: Any,
    model_name: str,
    model_config: Mapping[str, Any],
) -> Any:
    """Select one explicit recurrent head from a multi-head training model."""

    if model_name != "multihead_beatnet":
        return model

    import torch

    genre_label = str(model_config["genre_label"])
    try:
        head = model.heads[genre_label]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"MultiHeadBeatNet has no deployable head {genre_label!r}"
        ) from exc

    class SelectedMultiHeadBeatNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = int(model.input_dim)
            self.conv1 = model.conv1
            self.linear0 = head["linear0"]
            self.lstm = head["lstm"]
            self.linear = head["linear"]

    return SelectedMultiHeadBeatNet().eval()


def _export_graph(model: Any, model_name: str) -> tuple[Any, tuple[str, ...]]:
    import torch
    from .precise_streaming import PreciseRecurrentStep

    if model_name in {"beatnet", "multihead_beatnet"}:
        output_layer_name = "linear"
    elif model_name == "beatnet_plus":
        output_layer_name = "output_linear"
    else:
        output_layer_name = "dance_head"
    output_layer = getattr(model, output_layer_name, None)
    for name in ("conv1", "linear0", "lstm"):
        if getattr(model, name, None) is None:
            raise TypeError(f"{model_name} has no {name} module")
    if output_layer is None:
        raise TypeError(f"{model_name} has no {output_layer_name} module")
    if int(getattr(output_layer, "out_features", -1)) != 3:
        raise TypeError(f"{model_name} output layer must have exactly three heads")

    class StreamingBeatNetGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = PreciseRecurrentStep(model, output_layer)

        def forward(self, features: Any, hidden: Any, cell: Any) -> tuple[Any, Any, Any]:
            logits, next_hidden, next_cell = self.step(features, hidden, cell)
            probabilities = self.step.math.softmax(logits)
            activations = torch.stack(
                (probabilities[..., 0] + probabilities[..., 1], probabilities[..., 1]),
                dim=-1,
            ).to(torch.float32)
            return activations, next_hidden, next_cell

    if model_name != "dance_beatnet":
        return StreamingBeatNetGraph().eval(), _STANDARD_OUTPUT_NAMES

    tracking_target = getattr(model, "tracking_target", None)
    tracking_target_value = getattr(tracking_target, "value", tracking_target)
    accent_index = 1 if tracking_target_value == "beat" else 2

    class StreamingDanceBeatNetGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = PreciseRecurrentStep(model, output_layer)

        def forward(self, features: Any, hidden: Any, cell: Any) -> tuple[Any, ...]:
            logits, next_hidden, next_cell = self.step(features, hidden, cell)
            probabilities = self.step.math.sigmoid(logits).to(torch.float32)
            beats = probabilities[..., 0:1]
            downbeats = probabilities[..., 1:2]
            dancebeats = probabilities[..., 2:3]
            activations = torch.cat((beats, probabilities[..., accent_index:accent_index+1]), dim=-1)
            return activations, beats, downbeats, dancebeats, next_hidden, next_cell

    return StreamingDanceBeatNetGraph().eval(), _DANCE_OUTPUT_NAMES


def build_streaming_step_graph(model: Any, model_name: str = "beatnet") -> Any:
    """Build the shared one-frame deployment computation without writing files.

    Android and desktop exporters use this same precision policy. Multihead
    training models must first select a specific recurrent head.
    """
    graph, _ = _export_graph(model, _model_name(model_name))
    return graph


def _parity_protocol() -> dict[str, Any]:
    """Return the exact replay protocol that makes cached parity current."""

    return {
        "name": "explicit-lstm-state-sequence",
        "sequence_length": _PARITY_SEQUENCE_LENGTH,
        "feature_generator": "numpy.default_rng.normal(mean=0,std=0.25)",
        "feature_seed": _PARITY_FEATURE_SEED,
        "initial_state": "zeros",
    }


def _parity_features(input_dim: int) -> np.ndarray:
    generator = np.random.default_rng(_PARITY_FEATURE_SEED)
    features = generator.normal(
        0.0,
        0.25,
        size=(_PARITY_SEQUENCE_LENGTH, input_dim),
    ).astype(np.float32)
    # Deterministic impulses ensure the sequence is not only low-amplitude
    # stationary noise and exercise a different convolution region each frame.
    frame_indices = np.arange(_PARITY_SEQUENCE_LENGTH)
    features[frame_indices, (frame_indices * 37 + 11) % input_dim] += np.float32(0.5)
    return np.ascontiguousarray(features)


def _parity_gate(
    graph: Any,
    model_path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compare every output while independently advancing PyTorch/ORT state."""

    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "native streaming export parity validation requires onnxruntime"
        ) from exc

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
    actual_input_names = tuple(value.name for value in session.get_inputs())
    actual_output_names = tuple(value.name for value in session.get_outputs())
    if actual_input_names != ("features", "hidden", "cell"):
        raise RuntimeError(
            "native streaming parity found unexpected graph inputs: "
            f"{actual_input_names!r}"
        )
    if actual_output_names != output_names:
        raise RuntimeError(
            "native streaming parity found unexpected graph outputs: "
            f"{actual_output_names!r}"
        )

    reference_hidden = torch.zeros(
        (num_layers, 1, hidden_dim),
        dtype=torch.float32,
    )
    reference_cell = torch.zeros_like(reference_hidden)
    runtime_hidden = np.zeros(reference_hidden.shape, dtype=np.float32)
    runtime_cell = np.zeros(reference_cell.shape, dtype=np.float32)
    per_output = {name: 0.0 for name in output_names}
    per_frame: list[dict[str, Any]] = []
    maximum_error = 0.0

    for frame_index, feature in enumerate(_parity_features(input_dim)):
        feature_values = feature.reshape(1, 1, input_dim)
        feature_tensor = torch.from_numpy(feature_values)
        with torch.no_grad():
            reference = graph(feature_tensor, reference_hidden, reference_cell)
        if not isinstance(reference, tuple) or len(reference) != len(output_names):
            raise RuntimeError(
                "native streaming PyTorch reference returned an unexpected "
                f"output contract at frame {frame_index}"
            )
        actual = session.run(
            output_names,
            {
                "features": feature_values,
                "hidden": runtime_hidden,
                "cell": runtime_cell,
            },
        )
        frame_errors: dict[str, float] = {}
        for name, expected_value, actual_value in zip(
            output_names,
            reference,
            actual,
            strict=True,
        ):
            expected = expected_value.detach().cpu().numpy()
            observed = np.asarray(actual_value)
            if observed.shape != expected.shape:
                raise RuntimeError(
                    f"native streaming {name} parity shape mismatch at frame "
                    f"{frame_index}: expected {expected.shape}, got {observed.shape}"
                )
            if observed.dtype != expected.dtype:
                raise RuntimeError(
                    f"native streaming {name} parity dtype mismatch at frame "
                    f"{frame_index}: expected {expected.dtype}, got {observed.dtype}"
                )
            if not np.isfinite(expected).all() or not np.isfinite(observed).all():
                raise RuntimeError(
                    f"native streaming {name} parity produced non-finite values "
                    f"at frame {frame_index}"
                )
            error = float(np.max(np.abs(observed - expected), initial=0.0))
            frame_errors[name] = error
            per_output[name] = max(per_output[name], error)
            maximum_error = max(maximum_error, error)
            if not np.allclose(
                observed,
                expected,
                rtol=_PARITY_RTOL,
                atol=_PARITY_ATOL,
            ):
                raise RuntimeError(
                    f"native streaming {name} parity failed at frame {frame_index} "
                    f"with maximum absolute error {error:.9g}"
                )
        per_frame.append(
            {
                "name": f"sequence_frame_{frame_index:02d}",
                "frame_index": frame_index,
                "per_output_max_abs_error": frame_errors,
            }
        )
        reference_hidden = reference[-2].detach()
        reference_cell = reference[-1].detach()
        runtime_hidden = np.ascontiguousarray(actual[-2], dtype=np.float32)
        runtime_cell = np.ascontiguousarray(actual[-1], dtype=np.float32)

    return {
        "schema": NATIVE_STREAMING_PARITY_SCHEMA,
        "version": NATIVE_STREAMING_PARITY_VERSION,
        "passed": True,
        "reference": "pytorch-eval-cpu",
        "provider": "CPUExecutionProvider",
        "rtol": _PARITY_RTOL,
        "atol": _PARITY_ATOL,
        "protocol": _parity_protocol(),
        "validated_outputs": list(output_names),
        "max_abs_error": maximum_error,
        "per_output_max_abs_error": per_output,
        "probes": per_frame,
        "onnx_sha256": _sha256_file(model_path),
    }


def export_streaming_beatnet_onnx(
    model: Any,
    destination: Path | str,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
) -> NativeModelArtifact:
    """Export a loaded online BeatNet-family model with explicit LSTM state."""

    import torch

    normalized_name = _model_name(model_name)
    deployment_config = _deployment_model_config(
        model,
        normalized_name,
        model_config,
    )
    deployment_model = _deployment_model(
        model,
        normalized_name,
        deployment_config,
    )
    input_dim, hidden_dim, num_layers = _streaming_shape(
        deployment_model,
        normalized_name,
    )
    try:
        parameter = next(deployment_model.parameters())
    except StopIteration as exc:
        raise ValueError("cannot export a model without parameters") from exc
    if parameter.device.type != "cpu":
        raise ValueError(
            "native CPU export requires a CPU model; move the online model to "
            "CPU before exporting"
        )
    if parameter.dtype != torch.float32:
        raise ValueError("native streaming export currently requires float32 weights")

    checkpoint_digest = str(checkpoint_sha256).strip().lower()
    if len(checkpoint_digest) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_digest
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_path.with_suffix(destination_path.suffix + ".json")
    graph, output_names = _export_graph(deployment_model, normalized_name)
    features = torch.zeros((1, 1, input_dim), dtype=torch.float32)
    hidden = torch.zeros((num_layers, 1, hidden_dim), dtype=torch.float32)
    cell = torch.zeros_like(hidden)

    try:
        # The legacy exporter is intentional for this first ABI: it emits a
        # primitive float64 graph and does not require onnxscript at deployment
        # time. Its float32 input/state/output ABI remains unchanged.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Exporting a model to ONNX with a batch_size.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"torch\.onnx\..*",
            )
            warnings.filterwarnings(
                "ignore",
                category=torch.jit.TracerWarning,
                module=r"torch\.nn\.modules\.rnn",
            )
            torch.onnx.export(
                graph,
                (features, hidden, cell),
                str(destination_path),
                input_names=("features", "hidden", "cell"),
                output_names=output_names,
                opset_version=ONNX_OPSET_VERSION,
                dynamo=False,
                do_constant_folding=True,
            )

        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "ONNX export validation requires the optional 'onnx' package"
            ) from exc
        serialized = onnx.load(str(destination_path))
        onnx.checker.check_model(serialized)
        vendor_math = {"Conv", "LSTM", "Gemm", "MatMul", "Exp", "Tanh", "Sigmoid", "ReduceSum"}
        unexpected = {node.op_type for node in serialized.graph.node} & vendor_math
        if unexpected:
            raise RuntimeError(f"Fixed-operation export contains backend-dependent math: {sorted(unexpected)}")
        parity = _parity_gate(
            graph,
            destination_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_names=output_names,
        )
    except Exception:
        destination_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise

    artifact_key, config_sha256 = _artifact_identity(
        model_name=normalized_name,
        model_config=deployment_config,
        checkpoint_sha256=checkpoint_digest,
        torch_version=str(torch.__version__),
    )
    if normalized_name == "dance_beatnet":
        tracking_target = str(deployment_config["tracking_target"])
        accent_output = "downbeats" if tracking_target == "beat" else "dancebeats"
        activation_definition = f"all_beats,{accent_output}"
        output_contract = {
            "primary_output": "activations",
            "primary_channels": ["all_beats", accent_output],
            "tracking_target": tracking_target,
            "independent_sigmoid_heads": {
                "all_beats": "beats",
                "downbeats": "downbeats",
                "dancebeats": "dancebeats",
            },
        }
    else:
        activation_definition = "all_beats,downbeats"
        output_contract = {
            "primary_output": "activations",
            "primary_channels": ["all_beats", "downbeats"],
            "tracking_target": "beat",
        }

    manifest = {
        "schema": NATIVE_STREAMING_SCHEMA,
        "abi_version": NATIVE_STREAMING_ABI_VERSION,
        "artifact_key": artifact_key,
        "model_family": normalized_name,
        "task": "beat_tracking",
        "graph_contract": "one-causal-feature-frame-with-explicit-lstm-state",
        "checkpoint_sha256": checkpoint_digest,
        "model_config_sha256": config_sha256,
        "onnx": {
            "filename": destination_path.name,
            "opset": ONNX_OPSET_VERSION,
            "sha256": _sha256_file(destination_path),
            "size_bytes": destination_path.stat().st_size,
        },
        "exporter": {
            "name": EXPORTER_NAME,
            "torch_version": str(torch.__version__),
        },
        "streaming_state": {
            "batch_size": 1,
            "time_steps": 1,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dtype": "float32",
        },
        "inputs": ["features", "hidden", "cell"],
        "outputs": list(output_names),
        "activation_definition": activation_definition,
        "output_contract": output_contract,
        "parity_validation": parity,
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        destination_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return NativeModelArtifact(
        model_path=destination_path,
        manifest_path=manifest_path,
        manifest=manifest,
        cache_hit=False,
    )


def _is_nonnegative_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) >= 0.0
    )


def _has_current_successful_parity(
    payload: Mapping[str, Any],
    onnx_record: Mapping[str, Any],
) -> bool:
    """Return whether cached parity proves the current graph and protocol."""

    parity = payload.get("parity_validation")
    outputs = payload.get("outputs")
    if not isinstance(parity, dict) or not isinstance(outputs, list):
        return False
    expected_parity_fields = {
        "schema",
        "version",
        "passed",
        "reference",
        "provider",
        "rtol",
        "atol",
        "protocol",
        "validated_outputs",
        "max_abs_error",
        "per_output_max_abs_error",
        "probes",
        "onnx_sha256",
    }
    if (
        set(parity) != expected_parity_fields
        or parity.get("schema") != NATIVE_STREAMING_PARITY_SCHEMA
        or not isinstance(parity.get("version"), int)
        or isinstance(parity.get("version"), bool)
        or parity.get("version") != NATIVE_STREAMING_PARITY_VERSION
        or parity.get("passed") is not True
        or parity.get("reference") != "pytorch-eval-cpu"
        or parity.get("provider") != "CPUExecutionProvider"
        or parity.get("rtol") != _PARITY_RTOL
        or parity.get("atol") != _PARITY_ATOL
        or parity.get("protocol") != _parity_protocol()
        or parity.get("validated_outputs") != outputs
        or parity.get("onnx_sha256") != onnx_record.get("sha256")
    ):
        return False
    if (
        not outputs
        or any(not isinstance(name, str) for name in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        return False

    per_output = parity.get("per_output_max_abs_error")
    frames = parity.get("probes")
    if not isinstance(per_output, dict) or set(per_output) != set(outputs):
        return False
    if not isinstance(frames, list) or len(frames) != _PARITY_SEQUENCE_LENGTH:
        return False
    observed_per_output = {name: 0.0 for name in outputs}
    for expected_index, frame in enumerate(frames):
        if (
            not isinstance(frame, dict)
            or set(frame) != {"name", "frame_index", "per_output_max_abs_error"}
            or frame.get("name") != f"sequence_frame_{expected_index:02d}"
            or not isinstance(frame.get("frame_index"), int)
            or isinstance(frame.get("frame_index"), bool)
            or frame.get("frame_index") != expected_index
        ):
            return False
        errors = frame.get("per_output_max_abs_error")
        if not isinstance(errors, dict) or set(errors) != set(outputs):
            return False
        for name in outputs:
            value = errors[name]
            if not _is_nonnegative_finite_number(value):
                return False
            observed_per_output[name] = max(
                observed_per_output[name],
                float(value),
            )

    for name in outputs:
        if (
            not _is_nonnegative_finite_number(per_output[name])
            or float(per_output[name]) != observed_per_output[name]
        ):
            return False
    maximum_error = parity.get("max_abs_error")
    recorded_maximum = max(observed_per_output.values(), default=0.0)
    # With zero initial state, an LSTM cell is bounded by the sequence length;
    # this is a conservative upper bound for every validated tensor.
    maximum_allowed = _PARITY_ATOL + _PARITY_RTOL * _PARITY_SEQUENCE_LENGTH
    return bool(
        _is_nonnegative_finite_number(maximum_error)
        and float(maximum_error) == recorded_maximum
        and recorded_maximum <= maximum_allowed
    )


def _load_cached_artifact(
    model_path: Path,
    manifest_path: Path,
    *,
    artifact_key: str,
) -> NativeModelArtifact | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        onnx_record = payload.get("onnx")
        exporter_record = payload.get("exporter")
        if not isinstance(onnx_record, dict) or not isinstance(exporter_record, dict):
            return None
        if (
            payload.get("schema") != NATIVE_STREAMING_SCHEMA
            or payload.get("abi_version") != NATIVE_STREAMING_ABI_VERSION
            or payload.get("artifact_key") != artifact_key
            or exporter_record.get("name") != EXPORTER_NAME
            or onnx_record.get("filename") != model_path.name
            or int(onnx_record.get("size_bytes", -1)) != model_path.stat().st_size
            or str(onnx_record.get("sha256")) != _sha256_file(model_path)
            or not _has_current_successful_parity(payload, onnx_record)
        ):
            return None
        return NativeModelArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=payload,
            cache_hit=True,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def ensure_streaming_beatnet_onnx(
    model: Any,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    cache_root: Path | str | None = None,
) -> NativeModelArtifact:
    """Return a validated cached graph, exporting it once when absent/stale."""

    import torch

    normalized_name = _model_name(model_name)
    deployment_config = _deployment_model_config(
        model,
        normalized_name,
        model_config,
    )
    artifact_key, _ = _artifact_identity(
        model_name=normalized_name,
        model_config=deployment_config,
        checkpoint_sha256=checkpoint_sha256,
        torch_version=str(torch.__version__),
    )
    root = (
        _cache_root() if cache_root is None else Path(cache_root).expanduser().resolve()
    )
    artifact_directory = root / normalized_name / artifact_key
    model_path = artifact_directory / "streaming.onnx"
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")

    with _EXPORT_LOCK:
        if model_path.is_file() and manifest_path.is_file():
            cached = _load_cached_artifact(
                model_path,
                manifest_path,
                artifact_key=artifact_key,
            )
            if cached is not None:
                return cached

        artifact_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="export-",
            dir=artifact_directory,
        ) as temporary_directory:
            temporary_model = Path(temporary_directory) / model_path.name
            exported = export_streaming_beatnet_onnx(
                model,
                temporary_model,
                model_name=normalized_name,
                model_config=deployment_config,
                checkpoint_sha256=checkpoint_sha256,
            )
            temporary_manifest = exported.manifest_path
            os.replace(temporary_model, model_path)
            os.replace(temporary_manifest, manifest_path)
        return NativeModelArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=exported.manifest,
            cache_hit=False,
        )


class OnnxBeatNetStreamingSession:
    """Stateful one-frame native runner for a validated BeatNet ONNX artifact."""

    def __init__(
        self,
        artifact: NativeModelArtifact,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "native streaming inference requires the optional onnxruntime package"
            ) from exc
        if intra_op_threads < 1:
            raise ValueError("intra_op_threads must be at least 1")

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
        input_names = tuple(value.name for value in self._session.get_inputs())
        output_names = tuple(value.name for value in self._session.get_outputs())
        if input_names != ("features", "hidden", "cell"):
            raise ValueError(f"unexpected native model inputs: {input_names!r}")
        manifest_output_names = tuple(
            str(name) for name in artifact.manifest["outputs"]
        )
        if output_names != manifest_output_names:
            raise ValueError(f"unexpected native model outputs: {output_names!r}")
        expected_output_names = (
            _DANCE_OUTPUT_NAMES
            if artifact.model_family == "dance_beatnet"
            else _STANDARD_OUTPUT_NAMES
        )
        if output_names != expected_output_names:
            raise ValueError(
                f"unsupported {artifact.model_family} output contract: {output_names!r}"
            )

        self.artifact = artifact
        self.input_dim = artifact.input_dim
        self.hidden_dim = artifact.hidden_dim
        self.num_layers = artifact.num_layers
        self.provider = "CPUExecutionProvider"
        self.output_names = output_names
        self.activation_output_names = output_names[:-2]
        self._features = np.empty((1, 1, self.input_dim), dtype=np.float32)
        self.reset_hidden()

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: object | None = None,
    ) -> None:
        del device
        if batch_size != 1:
            raise ValueError("native streaming ABI currently fixes batch_size=1")
        shape = (self.num_layers, 1, self.hidden_dim)
        self.hidden = np.zeros(shape, dtype=np.float32)
        self.cell = np.zeros(shape, dtype=np.float32)

    def infer_outputs(self, feature: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one frame and return every named non-state graph output."""

        values = np.asarray(feature, dtype=np.float32)
        if values.size != self.input_dim:
            raise ValueError(
                f"native feature frame must have {self.input_dim} values, "
                f"got {values.size}"
            )
        np.copyto(self._features.reshape(-1), values.reshape(-1))
        raw_outputs = self._session.run(
            self.output_names,
            {
                "features": self._features,
                "hidden": self.hidden,
                "cell": self.cell,
            },
        )
        output_values = dict(zip(self.output_names, raw_outputs, strict=True))
        next_hidden = output_values.pop("next_hidden")
        next_cell = output_values.pop("next_cell")
        self.hidden = np.asarray(next_hidden, dtype=np.float32)
        self.cell = np.asarray(next_cell, dtype=np.float32)
        results = {
            name: np.asarray(output_values[name], dtype=np.float32).reshape(-1)
            for name in self.activation_output_names
        }
        result = results["activations"].reshape(-1, 2)
        if result.shape != (1, 2):
            raise RuntimeError(
                f"native runtime returned unexpected activation shape {result.shape}"
            )
        results["activations"] = result[0]
        if self.artifact.model_family == "dance_beatnet":
            for head_name in ("beats", "downbeats", "dancebeats"):
                if results[head_name].shape != (1,):
                    raise RuntimeError(
                        f"native runtime returned unexpected {head_name} shape "
                        f"{results[head_name].shape}"
                    )
        return results

    def infer(self, feature: np.ndarray) -> np.ndarray:
        """Advance one frame and return its canonical two-channel projection."""

        return self.infer_outputs(feature)["activations"]
