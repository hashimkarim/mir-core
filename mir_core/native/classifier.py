"""Parity-gated ONNX artifacts for genre-classifier model heads.

Artifacts consume the exact model-ready feature tensors used during training.
External EfficientAT and YAMNet audio frontends are deliberately not folded
into these graphs because their executable weights are not packaged by
``mir-core``; promoted downstream heads remain fully portable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

NATIVE_CLASSIFIER_SCHEMA = "mir.native-classifier-model/v1"
NATIVE_CLASSIFIER_ABI_VERSION = 1
CLASSIFIER_ONNX_OPSET_VERSION = 18
SUPPORTED_CLASSIFIER_ARCHITECTURES = frozenset(
    {
        "mel_cnn",
        "mfcc_cnn",
        "mel_cnn_attention",
        "beatnet_log_spect_cnn",
        "embedding_stats_mlp",
        "framewise_embedding_mlp",
        "beatnet_conv",
    }
)
UNSUPPORTED_CLASSIFIER_FRONTENDS = {
    "efficientat_embedding": (
        "downstream head only: EfficientAT code and backbone weights are not "
        "packaged in mir-core, so model-ready embeddings are required"
    ),
    "yamnet_embedding": (
        "downstream head only: the TensorFlow Hub YAMNet graph is not packaged "
        "in mir-core, so model-ready embeddings are required"
    ),
}
_CLASS_NAME_TO_ARCHITECTURE = {
    "MelCNN": "mel_cnn",
    "MFCCCNN": "mfcc_cnn",
    "MelCNNAttention": "mel_cnn_attention",
    "BeatNetLogSpectCNN": "beatnet_log_spect_cnn",
    "EmbeddingStatsMLP": "embedding_stats_mlp",
    "FramewiseEmbeddingMLP": "framewise_embedding_mlp",
    "BeatNetConvClassifier": "beatnet_conv",
}
_EXPORT_LOCK = threading.Lock()
_PARITY_RTOL = 2e-5
_PARITY_ATOL = 2e-6
_OUTPUT_NAMES = ("logits", "probabilities")


@dataclass(frozen=True, slots=True)
class NativeClassifierArtifact:
    """A validated classifier graph and its deployment manifest."""

    model_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    cache_hit: bool

    @property
    def architecture(self) -> str:
        return str(self.manifest["architecture"])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(str(label) for label in self.manifest["labels"])

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.manifest["outputs"])


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


def _checkpoint_digest(value: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    return digest


def _input_shape(value: Sequence[int]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_shape must contain positive integers") from exc
    if not shape or any(dimension < 1 for dimension in shape):
        raise ValueError("input_shape must contain positive integers")
    return shape


def _architecture(model: Any) -> tuple[str, Any, bool]:
    architecture = getattr(model, "arch_name", None)
    underlying = getattr(model, "model", None)
    is_genre_wrapper = isinstance(architecture, str) and underlying is not None
    if is_genre_wrapper:
        normalized = architecture.strip().lower().replace("-", "_")
    else:
        underlying = model
        normalized = _CLASS_NAME_TO_ARCHITECTURE.get(type(model).__name__, "")
    if normalized not in SUPPORTED_CLASSIFIER_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_CLASSIFIER_ARCHITECTURES))
        raise ValueError(
            f"native classifier export does not support {type(model).__name__!r}; "
            f"supported architectures: {supported}"
        )
    return normalized, underlying, is_genre_wrapper


def _num_classes(underlying: Any) -> int:
    import torch

    linear_layers = [
        module for module in underlying.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise TypeError("classifier has no linear output layer")
    count = int(linear_layers[-1].out_features)
    if count < 1:
        raise TypeError("classifier output class count must be positive")
    return count


def _labels(
    model: Any,
    num_classes: int,
    class_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    model_labels = getattr(model, "genre_labels", None)
    source = model_labels if model_labels is not None else class_labels
    if source is None:
        labels = tuple(f"class_{index}" for index in range(num_classes))
    else:
        labels = tuple(str(label) for label in source)
    if len(labels) != num_classes:
        raise ValueError(
            f"class_labels must contain {num_classes} entries, got {len(labels)}"
        )
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("class_labels must be non-empty and unique")
    if model_labels is not None and class_labels is not None:
        supplied = tuple(str(label) for label in class_labels)
        if supplied != labels:
            raise ValueError("class_labels do not match GenreClassifier.genre_labels")
    return labels


def _temperature(
    model: Any,
    model_config: Mapping[str, Any],
    is_genre_wrapper: bool,
) -> float:
    if is_genre_wrapper:
        value = getattr(model, "calibration_temperature")
        configured = model_config.get("calibration_temperature")
        if configured is not None and float(configured) != float(value):
            raise ValueError(
                "model_config calibration_temperature does not match GenreClassifier"
            )
    else:
        value = model_config.get("calibration_temperature", 1.0)
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration_temperature must be positive and finite")
    return temperature


def _shape_contract(
    architecture: str,
    underlying: Any,
    example_shape: tuple[int, ...],
) -> dict[str, Any]:
    if architecture == "beatnet_conv":
        if len(example_shape) != 3:
            raise ValueError(
                "BeatNetConvClassifier input_shape must be [batch, time, features]"
            )
        first_linear = next(iter(underlying.classifier))
        expected_features = int(first_linear.in_features)
        if example_shape[2] != expected_features:
            raise ValueError(
                f"BeatNetConvClassifier requires {expected_features} features"
            )
        return {
            "mode": "dynamic_batch_and_time_fixed_features",
            "input": {
                "name": "features",
                "dtype": "float32",
                "shape": ["batch", "time", expected_features],
                "constraints": {"time_min": 1},
            },
        }

    if len(example_shape) != 4:
        raise ValueError(
            f"{architecture} input_shape must be [batch, channel, features, time]"
        )
    if example_shape[1] != 1:
        raise ValueError(f"{architecture} requires a singleton channel axis")
    expected_features = example_shape[2]
    declared_features = getattr(underlying, "feature_dim", None)
    if declared_features is None:
        declared_features = getattr(underlying, "embedding_dim", None)
    if declared_features is not None and expected_features != int(declared_features):
        raise ValueError(f"{architecture} requires {int(declared_features)} features")
    minimum_time = (
        8
        if architecture
        in {
            "mel_cnn",
            "mfcc_cnn",
            "mel_cnn_attention",
            "beatnet_log_spect_cnn",
        }
        else 1
    )
    if example_shape[3] < minimum_time:
        raise ValueError(f"{architecture} requires at least {minimum_time} time frames")
    return {
        "mode": "dynamic_batch_and_time_fixed_features",
        "input": {
            "name": "features",
            "dtype": "float32",
            "shape": ["batch", 1, expected_features, "time"],
            "constraints": {"time_min": minimum_time},
        },
    }


def _frontend_contract(feature_config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(feature_config or {})
    feature_type = str(config.get("type", "model_ready_features"))
    feature_config_sha256 = _sha256_bytes(_canonical_json(config).encode())
    unsupported_reason = UNSUPPORTED_CLASSIFIER_FRONTENDS.get(feature_type)
    if unsupported_reason is None:
        return {
            "included": False,
            "input_representation": feature_type,
            "status": "model_ready_feature_tensor_required",
            "feature_config_sha256": feature_config_sha256,
        }
    return {
        "included": False,
        "input_representation": feature_type,
        "status": "downstream_head_only",
        "feature_config_sha256": feature_config_sha256,
        "unsupported_full_frontend_reason": unsupported_reason,
    }


def _output_contract(
    labels: tuple[str, ...],
    temperature: float,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "logits",
            "semantics": "raw_class_logits",
            "classes": list(labels),
            "shape": ["batch", len(labels)],
        },
        {
            "name": "probabilities",
            "semantics": "temperature_scaled_softmax_probability",
            "classes": list(labels),
            "temperature": temperature,
            "shape": ["batch", len(labels)],
        },
    ]


def _artifact_identity(
    *,
    architecture: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    shape_contract: Mapping[str, Any],
    output_contract: Sequence[Mapping[str, Any]],
    frontend_contract: Mapping[str, Any],
    wrapper: str,
    torch_version: str,
) -> tuple[str, str]:
    config_sha256 = _sha256_bytes(_canonical_json(dict(model_config)).encode())
    payload = {
        "abi_version": NATIVE_CLASSIFIER_ABI_VERSION,
        "architecture": architecture,
        "checkpoint_sha256": checkpoint_sha256,
        "exporter": "torch.onnx.legacy",
        "frontend_contract": dict(frontend_contract),
        "model_config_sha256": config_sha256,
        "opset": CLASSIFIER_ONNX_OPSET_VERSION,
        "output_contract": list(output_contract),
        "shape_contract": dict(shape_contract),
        "torch_version": torch_version,
        "wrapper": wrapper,
    }
    return _sha256_bytes(_canonical_json(payload).encode())[:24], config_sha256


def _deployment_graph(model: Any, temperature: float) -> Any:
    import torch

    deployment_model = copy.deepcopy(model).cpu().eval()

    class ClassifierDeploymentGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = deployment_model
            self.temperature = temperature

        def forward(self, features: Any) -> tuple[Any, Any]:
            logits = self.model(features)
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            return logits, probabilities

    return ClassifierDeploymentGraph().eval()


def _cache_root() -> Path:
    configured = os.environ.get("MIR_NATIVE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve() / "classifier"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (
            Path(xdg_cache).expanduser().resolve()
            / "mir"
            / "native-models"
            / "classifier"
        )
    return Path.home() / ".cache" / "mir" / "native-models" / "classifier"


def _parity_gate(graph: Any, model_path: Path, features: Any) -> dict[str, Any]:
    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(
            "classifier export parity validation requires onnxruntime"
        ) from exc

    with torch.no_grad():
        reference = graph(features)
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
    actual = session.run(_OUTPUT_NAMES, {"features": features.numpy()})
    maximum_error = 0.0
    per_output: dict[str, float] = {}
    for name, expected_value, actual_value in zip(
        _OUTPUT_NAMES,
        reference,
        actual,
        strict=True,
    ):
        expected = expected_value.detach().cpu().numpy()
        observed = np.asarray(actual_value)
        if observed.shape != expected.shape:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"native classifier {name} shape mismatch: "
                f"expected {expected.shape}, got {observed.shape}"
            )
        error = float(np.max(np.abs(observed - expected), initial=0.0))
        per_output[name] = error
        maximum_error = max(maximum_error, error)
        if not np.allclose(
            observed,
            expected,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        ):
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"native classifier {name} parity failed with maximum "
                f"absolute error {error:.9g}"
            )
    return {
        "provider": "CPUExecutionProvider",
        "rtol": _PARITY_RTOL,
        "atol": _PARITY_ATOL,
        "max_abs_error": maximum_error,
        "per_output_max_abs_error": per_output,
        "input_pattern": "linspace[-1,1]",
    }


def _export_contracts(
    model: Any,
    *,
    model_config: Mapping[str, Any],
    input_shape: Sequence[int],
    class_labels: Sequence[str] | None,
    feature_config: Mapping[str, Any] | None,
) -> tuple[
    str,
    tuple[int, ...],
    tuple[str, ...],
    float,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    str,
]:
    architecture, underlying, is_genre_wrapper = _architecture(model)
    configured_architecture = model_config.get("arch")
    if (
        configured_architecture is not None
        and str(configured_architecture) != architecture
    ):
        raise ValueError("model_config arch does not match the loaded classifier")
    example_shape = _input_shape(input_shape)
    num_classes = _num_classes(underlying)
    labels = _labels(model, num_classes, class_labels)
    temperature = _temperature(model, model_config, is_genre_wrapper)
    shape_contract = _shape_contract(architecture, underlying, example_shape)
    output_contract = _output_contract(labels, temperature)
    frontend_contract = _frontend_contract(feature_config)
    wrapper = "genre_classifier" if is_genre_wrapper else "bare_architecture"
    return (
        architecture,
        example_shape,
        labels,
        temperature,
        shape_contract,
        output_contract,
        frontend_contract,
        wrapper,
    )


def export_classifier_onnx(
    model: Any,
    destination: Path | str,
    *,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    input_shape: Sequence[int],
    class_labels: Sequence[str] | None = None,
    feature_config: Mapping[str, Any] | None = None,
) -> NativeClassifierArtifact:
    """Export one classifier head and reject it unless ORT matches PyTorch."""

    import torch

    digest = _checkpoint_digest(checkpoint_sha256)
    (
        architecture,
        example_shape,
        labels,
        temperature,
        shape_contract,
        output_contract,
        frontend_contract,
        wrapper,
    ) = _export_contracts(
        model,
        model_config=model_config,
        input_shape=input_shape,
        class_labels=class_labels,
        feature_config=feature_config,
    )
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("cannot export a classifier without parameters") from exc
    if parameter.device.type != "cpu":
        raise ValueError("native classifier export requires a CPU model")
    if parameter.dtype != torch.float32:
        raise ValueError("native classifier export currently requires float32 weights")

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    graph = _deployment_graph(model, temperature)
    feature_count = int(np.prod(example_shape, dtype=np.int64))
    features = torch.linspace(
        -1.0,
        1.0,
        steps=feature_count,
        dtype=torch.float32,
    ).reshape(example_shape)
    input_time_axis = 1 if architecture == "beatnet_conv" else 3
    dynamic_axes = {
        "features": {0: "batch", input_time_axis: "time"},
        "logits": {0: "batch"},
        "probabilities": {0: "batch"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"torch\.onnx\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=torch.jit.TracerWarning,
        )
        torch.onnx.export(
            graph,
            features,
            str(destination_path),
            input_names=("features",),
            output_names=_OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=CLASSIFIER_ONNX_OPSET_VERSION,
            dynamo=False,
            do_constant_folding=True,
        )

    try:
        import onnx
    except ImportError as exc:
        destination_path.unlink(missing_ok=True)
        raise RuntimeError(
            "classifier export requires the optional onnx package"
        ) from exc
    onnx.checker.check_model(onnx.load(str(destination_path)))
    parity = _parity_gate(graph, destination_path, features)
    artifact_key, config_sha256 = _artifact_identity(
        architecture=architecture,
        model_config=model_config,
        checkpoint_sha256=digest,
        shape_contract=shape_contract,
        output_contract=output_contract,
        frontend_contract=frontend_contract,
        wrapper=wrapper,
        torch_version=str(torch.__version__),
    )
    manifest_path = destination_path.with_suffix(destination_path.suffix + ".json")
    manifest = {
        "schema": NATIVE_CLASSIFIER_SCHEMA,
        "abi_version": NATIVE_CLASSIFIER_ABI_VERSION,
        "artifact_key": artifact_key,
        "model_family": "classifier",
        "architecture": architecture,
        "wrapper": wrapper,
        "task": "classification",
        "graph_contract": "model-ready-features-to-calibrated-class-probabilities",
        "checkpoint_sha256": digest,
        "model_config_sha256": config_sha256,
        "onnx": {
            "filename": destination_path.name,
            "opset": CLASSIFIER_ONNX_OPSET_VERSION,
            "sha256": _sha256_file(destination_path),
            "size_bytes": destination_path.stat().st_size,
        },
        "exporter": {
            "name": "torch.onnx.legacy",
            "torch_version": str(torch.__version__),
        },
        "inputs": ["features"],
        "outputs": list(_OUTPUT_NAMES),
        "labels": list(labels),
        "calibration_temperature": temperature,
        "shape_contract": shape_contract,
        "output_contract": output_contract,
        "feature_frontend": frontend_contract,
        "export_example_shape": list(example_shape),
        "parity_validation": parity,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NativeClassifierArtifact(
        model_path=destination_path,
        manifest_path=manifest_path,
        manifest=manifest,
        cache_hit=False,
    )


def _load_cached_artifact(
    model_path: Path,
    manifest_path: Path,
    *,
    artifact_key: str,
) -> NativeClassifierArtifact | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        onnx_record = payload.get("onnx")
        if not isinstance(onnx_record, dict):
            return None
        if (
            payload.get("schema") != NATIVE_CLASSIFIER_SCHEMA
            or payload.get("abi_version") != NATIVE_CLASSIFIER_ABI_VERSION
            or payload.get("artifact_key") != artifact_key
            or onnx_record.get("filename") != model_path.name
            or int(onnx_record.get("size_bytes", -1)) != model_path.stat().st_size
            or str(onnx_record.get("sha256")) != _sha256_file(model_path)
        ):
            return None
        return NativeClassifierArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=payload,
            cache_hit=True,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def ensure_classifier_onnx(
    model: Any,
    *,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    input_shape: Sequence[int],
    class_labels: Sequence[str] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    cache_root: Path | str | None = None,
) -> NativeClassifierArtifact:
    """Return a verified cached classifier artifact or export it atomically."""

    import torch

    digest = _checkpoint_digest(checkpoint_sha256)
    (
        architecture,
        example_shape,
        _,
        _,
        shape_contract,
        output_contract,
        frontend_contract,
        wrapper,
    ) = _export_contracts(
        model,
        model_config=model_config,
        input_shape=input_shape,
        class_labels=class_labels,
        feature_config=feature_config,
    )
    artifact_key, _ = _artifact_identity(
        architecture=architecture,
        model_config=model_config,
        checkpoint_sha256=digest,
        shape_contract=shape_contract,
        output_contract=output_contract,
        frontend_contract=frontend_contract,
        wrapper=wrapper,
        torch_version=str(torch.__version__),
    )
    root = (
        _cache_root() if cache_root is None else Path(cache_root).expanduser().resolve()
    )
    artifact_directory = root / architecture / artifact_key
    model_path = artifact_directory / "classifier.onnx"
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
            exported = export_classifier_onnx(
                model,
                temporary_model,
                model_config=model_config,
                checkpoint_sha256=digest,
                input_shape=example_shape,
                class_labels=class_labels,
                feature_config=feature_config,
            )
            os.replace(temporary_model, model_path)
            os.replace(exported.manifest_path, manifest_path)
        return NativeClassifierArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=exported.manifest,
            cache_hit=False,
        )


class OnnxClassifierSession:
    """CPU ONNX Runtime host for a validated classifier artifact."""

    def __init__(
        self,
        artifact: NativeClassifierArtifact,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "native classifier inference requires onnxruntime"
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
        if input_names != ("features",):
            raise ValueError(f"unexpected native classifier inputs: {input_names!r}")
        if output_names != artifact.output_names:
            raise ValueError(f"unexpected native classifier outputs: {output_names!r}")

        self.artifact = artifact
        self.output_names = output_names
        self.labels = artifact.labels
        self.provider = "CPUExecutionProvider"
        self._input_contract = artifact.manifest["shape_contract"]["input"]

    def infer(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Return raw logits and calibrated probabilities for one batch."""

        values = np.ascontiguousarray(features, dtype=np.float32)
        expected_shape = self._input_contract["shape"]
        if values.ndim != len(expected_shape):
            raise ValueError(
                f"native classifier input must have rank {len(expected_shape)}, "
                f"got rank {values.ndim}"
            )
        for axis, (actual, expected) in enumerate(
            zip(values.shape, expected_shape, strict=True)
        ):
            if isinstance(expected, int) and actual != expected:
                raise ValueError(
                    f"native classifier input axis {axis} must be {expected}, "
                    f"got {actual}"
                )
        minimum_time = int(self._input_contract["constraints"]["time_min"])
        time_axis = 1 if self.artifact.architecture == "beatnet_conv" else 3
        if values.shape[time_axis] < minimum_time:
            raise ValueError(
                f"native classifier input requires at least {minimum_time} time frames"
            )
        outputs = self._session.run(
            self.output_names,
            {"features": values},
        )
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in zip(self.output_names, outputs, strict=True)
        }


__all__ = [
    "CLASSIFIER_ONNX_OPSET_VERSION",
    "NATIVE_CLASSIFIER_ABI_VERSION",
    "NATIVE_CLASSIFIER_SCHEMA",
    "NativeClassifierArtifact",
    "OnnxClassifierSession",
    "SUPPORTED_CLASSIFIER_ARCHITECTURES",
    "UNSUPPORTED_CLASSIFIER_FRONTENDS",
    "ensure_classifier_onnx",
    "export_classifier_onnx",
]
