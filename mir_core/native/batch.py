"""Portable ONNX artifacts for non-streaming MIR model families.

The batch ABI is deliberately separate from the recurrent streaming ABI. It
captures fixed versus dynamic axes explicitly and only accepts model families
whose exported graph passes an ONNX Runtime numerical parity gate.
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

NATIVE_BATCH_SCHEMA = "mir.native-batch-model/v1"
NATIVE_BATCH_ABI_VERSION = 1
BATCH_ONNX_OPSET_VERSION = 18
SUPPORTED_BATCH_MODELS = frozenset({"beast", "bocktcn", "spectnt"})
UNSUPPORTED_BATCH_MODELS: dict[str, str] = {}
_MODEL_NAME_ALIASES = {
    "bock_tcn": "bocktcn",
    "spec_tnt": "spectnt",
}
_EXPORT_LOCK = threading.Lock()
_PARITY_RTOL = 2e-5
_PARITY_ATOL = 2e-6


class UnsupportedNativeBatchModelError(ValueError):
    """Raised when a known family has not passed the native parity gate."""


@dataclass(frozen=True, slots=True)
class NativeBatchModelArtifact:
    """A validated batch graph and its machine-readable deployment contract."""

    model_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    cache_hit: bool

    @property
    def model_family(self) -> str:
        return str(self.manifest["model_family"])

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


def _model_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _MODEL_NAME_ALIASES.get(normalized, normalized)
    if normalized in UNSUPPORTED_BATCH_MODELS:
        raise UnsupportedNativeBatchModelError(
            f"native batch export does not support {value!r}: "
            f"{UNSUPPORTED_BATCH_MODELS[normalized]}"
        )
    if normalized not in SUPPORTED_BATCH_MODELS:
        supported = ", ".join(sorted(SUPPORTED_BATCH_MODELS))
        raise ValueError(
            f"native batch export does not support {value!r}; "
            f"supported models: {supported}"
        )
    return normalized


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


def _cache_root() -> Path:
    configured = os.environ.get("MIR_NATIVE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve() / "batch"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (
            Path(xdg_cache).expanduser().resolve() / "mir" / "native-models" / "batch"
        )
    return Path.home() / ".cache" / "mir" / "native-models" / "batch"


def _bocktcn_contract(
    model: Any,
    example_shape: tuple[int, ...],
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    if len(example_shape) != 4:
        raise ValueError(
            "BockTCN input_shape must be [batch, channel, time, frequency]"
        )
    if example_shape[1] != 1 or example_shape[3] != 81:
        raise ValueError("BockTCN requires one channel and 81 frequency bands")
    if example_shape[2] < 5:
        raise ValueError("BockTCN requires at least five input frames")

    include_downbeats = bool(getattr(model, "include_downbeats", False))
    include_tempo = bool(getattr(model, "include_tempo", False))
    output_names = ["activations", "beats"]
    output_specs = [
        {
            "name": "activations",
            "semantics": "event_activation",
            "channels": ["all_beats", "downbeats"],
            "shape": ["batch", "input_time-4", 2],
            "downbeats_available": include_downbeats,
        },
        {
            "name": "beats",
            "semantics": "all_beats_probability",
            "shape": ["batch", "input_time-4", 1],
        },
    ]
    if include_downbeats:
        output_names.append("downbeats")
        output_specs.append(
            {
                "name": "downbeats",
                "semantics": "downbeat_probability",
                "shape": ["batch", "input_time-4", 1],
            }
        )
    if include_tempo:
        tempo_dense = getattr(model, "tempo_dense", None)
        tempo_classes = int(getattr(tempo_dense, "out_features", -1))
        if tempo_classes < 1:
            raise TypeError("BockTCN tempo head has no valid output dimension")
        output_names.append("tempo")
        output_specs.append(
            {
                "name": "tempo",
                "semantics": "tempo_class_probability",
                "shape": ["batch", tempo_classes],
            }
        )

    shape_contract = {
        "mode": "dynamic_batch_and_time",
        "input": {
            "name": "features",
            "dtype": "float32",
            "shape": ["batch", 1, "input_time", 81],
            "constraints": {"input_time_min": 5},
        },
        "time_relation": "output_time=input_time-4",
    }
    return shape_contract, tuple(output_names), output_specs


def _spectnt_contract(
    model: Any,
    example_shape: tuple[int, ...],
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    if len(example_shape) != 4:
        raise ValueError(
            "SpecTNT input_shape must be [batch, channel, frequency, time]"
        )
    input_bn = getattr(getattr(model, "fe_model", None), "input_bn", None)
    expected_channels = int(getattr(input_bn, "num_features", example_shape[1]))
    if example_shape[1] != expected_channels:
        raise ValueError(f"SpecTNT requires {expected_channels} input channels")
    if int(getattr(model, "n_classes", -1)) != 3:
        raise TypeError("SpecTNT batch ABI requires exactly three frame classes")

    output_time = 1 if bool(getattr(model, "use_tct", False)) else int(model.n_times)
    output_names = (
        "logits",
        "frame_class_activations",
        "event_activations",
        "beats",
        "downbeats",
    )
    output_specs = [
        {
            "name": "logits",
            "semantics": "frame_class_logits",
            "channels": ["beat_only", "downbeat", "non_beat"],
            "shape": ["batch", output_time, 3],
        },
        {
            "name": "frame_class_activations",
            "semantics": "frame_class_probability",
            "channels": ["beat_only", "downbeat", "non_beat"],
            "shape": ["batch", output_time, 3],
        },
        {
            "name": "event_activations",
            "semantics": "event_activation",
            "channels": ["all_beats", "downbeats"],
            "shape": ["batch", output_time, 2],
        },
        {
            "name": "beats",
            "semantics": "all_beats_probability",
            "shape": ["batch", output_time, 1],
        },
        {
            "name": "downbeats",
            "semantics": "downbeat_probability",
            "shape": ["batch", output_time, 1],
        },
    ]
    shape_contract = {
        "mode": "dynamic_batch_fixed_features",
        "input": {
            "name": "features",
            "dtype": "float32",
            "shape": ["batch", *example_shape[1:]],
        },
        "configured_frontend_output": {
            "frequency": int(model.n_frequencies),
            "time": int(model.n_times),
        },
    }
    return shape_contract, output_names, output_specs


def _beast_contract(
    model: Any,
    example_shape: tuple[int, ...],
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    if len(example_shape) != 3:
        raise ValueError("BEAST input_shape must be [batch, time, features]")
    if example_shape[2] != 128:
        raise ValueError("BEAST requires 128 input features")
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise TypeError("BEAST has no contextual-block encoder")
    block_size = int(encoder.block_size)
    left_size = int(encoder.left)
    center_size = int(encoder.center)
    look_ahead = int(encoder.look_ahead)
    input_time = example_shape[1]
    if block_size <= 0 or center_size <= 0:
        raise ValueError("BEAST native export requires finite contextual blocks")
    if not bool(encoder.init_average):
        raise ValueError("BEAST native export requires average context initialization")
    if input_time <= block_size:
        raise ValueError(
            f"BEAST native export requires more than {block_size} frames to use "
            "the validated contextual-block path"
        )
    if len(encoder.encoders) < 1:
        raise ValueError("BEAST native export requires at least one encoder layer")
    block_count = math.ceil((input_time - left_size - look_ahead) / center_size)
    if block_count < 2:
        raise ValueError("BEAST native export requires at least two context blocks")

    beat_classes = int(getattr(getattr(model, "out_linear", None), "out_features", -1))
    tempo_classes = int(
        getattr(getattr(model, "out_linear_t", None), "out_features", -1)
    )
    if beat_classes != 2:
        raise TypeError("BEAST batch ABI requires exactly two beat classes")
    if tempo_classes < 1:
        raise TypeError("BEAST output heads have invalid dimensions")
    output_names = ("logits", "tempo_logits")
    output_specs = [
        {
            "name": "logits",
            "semantics": "raw_beat_downbeat_logits",
            "channels": ["beat", "downbeat"],
            "shape": ["batch", input_time, beat_classes],
        },
        {
            "name": "tempo_logits",
            "semantics": "raw_tempo_class_logits",
            "shape": ["batch", tempo_classes],
        },
    ]
    shape_contract = {
        "mode": "dynamic_batch_fixed_context_and_features",
        "input": {
            "name": "features",
            "dtype": "float32",
            "shape": ["batch", input_time, 128],
        },
        "contextual_blocks": {
            "left_size": left_size,
            "center_size": center_size,
            "look_ahead": look_ahead,
            "block_size": block_size,
            "block_count": block_count,
        },
    }
    return shape_contract, output_names, output_specs


def _contracts(
    model: Any,
    model_name: str,
    example_shape: tuple[int, ...],
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    if model_name == "beast":
        return _beast_contract(model, example_shape)
    if model_name == "bocktcn":
        return _bocktcn_contract(model, example_shape)
    return _spectnt_contract(model, example_shape)


def _artifact_identity(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    shape_contract: Mapping[str, Any],
    output_contract: Sequence[Mapping[str, Any]],
    torch_version: str,
) -> tuple[str, str]:
    config_sha256 = _sha256_bytes(_canonical_json(dict(model_config)).encode())
    payload = {
        "abi_version": NATIVE_BATCH_ABI_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "exporter": "torch.onnx.legacy",
        "model_config_sha256": config_sha256,
        "model_family": model_name,
        "opset": BATCH_ONNX_OPSET_VERSION,
        "output_contract": list(output_contract),
        "shape_contract": dict(shape_contract),
        "torch_version": torch_version,
    }
    return _sha256_bytes(_canonical_json(payload).encode())[:24], config_sha256


def _replace_same_conv1d_padding(module: Any) -> None:
    """Replace export-hostile SAME Conv1d layers with equivalent numeric padding."""

    import torch

    for child_name, child in tuple(module.named_children()):
        if not isinstance(child, torch.nn.Conv1d) or child.padding != "same":
            _replace_same_conv1d_padding(child)
            continue
        if child.stride != (1,):
            raise ValueError("numeric SAME conversion only supports Conv1d stride 1")
        total_padding = child.dilation[0] * (child.kernel_size[0] - 1)
        if total_padding % 2:
            raise ValueError(
                "numeric SAME conversion requires symmetric effective padding"
            )
        replacement = torch.nn.Conv1d(
            in_channels=child.in_channels,
            out_channels=child.out_channels,
            kernel_size=child.kernel_size,
            stride=child.stride,
            padding=(total_padding // 2,),
            dilation=child.dilation,
            groups=child.groups,
            bias=child.bias is not None,
            padding_mode=child.padding_mode,
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        replacement.load_state_dict(child.state_dict())
        replacement.train(child.training)
        for new_parameter, old_parameter in zip(
            replacement.parameters(),
            child.parameters(),
            strict=True,
        ):
            new_parameter.requires_grad_(old_parameter.requires_grad)
        setattr(module, child_name, replacement)


def _beast_deployment_graph(model: Any, input_time: int) -> Any:
    """Build a mutation-free equivalent of BEAST's contextual block path.

    The source implementation fills block and output containers in-place. Those
    writes are valid in eager PyTorch but are not represented faithfully by the
    legacy ONNX tracer. This graph expresses the same fixed-context computation
    with concatenation and stacking, including the layer-to-layer context shift.
    """

    import torch

    encoder = model.encoder
    block_size = int(encoder.block_size)
    center_size = int(encoder.center)
    look_ahead = int(encoder.look_ahead)
    sequence_size = block_size + 2
    block_count = math.ceil((input_time - int(encoder.left) - look_ahead) / center_size)
    feature_size = int(encoder.output_size())
    last_block_size = input_time - (block_count - 1) * center_size
    output_offset = block_size - look_ahead - center_size + 1

    with torch.no_grad():
        positional_input = torch.zeros(1, sequence_size, feature_size)
        _, position_embedding = encoder.encoders[0].pos_enc(positional_input)
    attention_mask = torch.zeros(1, sequence_size, sequence_size)
    attention_mask[:, 1:, :-1] = 1.0

    class BeastDeploymentGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer(
                "position_embedding",
                position_embedding.detach().clone(),
            )
            self.register_buffer("attention_mask", attention_mask)

        def frontend(self, features: Any) -> Any:
            values = features.unsqueeze(1)
            values = self.model.conv1(values)
            values = values[:, :, :-2, :]
            values = self.model.maxpool1(values)
            values = torch.relu(values)
            values = self.model.dropout1(values)

            values = self.model.conv2(values)
            values = self.model.maxpool2(values)
            values = torch.relu(values)
            values = self.model.dropout2(values)

            values = self.model.conv3(values)
            values = values[:, :, :-1, :]
            values = self.model.maxpool3(values)
            values = torch.relu(values)
            values = self.model.dropout3(values)
            return values.transpose(1, 3).squeeze(1).contiguous()

        def assemble_blocks(self, values: Any) -> Any:
            context_vectors = []
            for block_index in range(block_count):
                start = block_index * center_size
                stop = min(start + block_size, input_time)
                context_vectors.append(values[:, start:stop].mean(dim=1))

            blocks = []
            for block_index in range(block_count):
                start = block_index * center_size
                valid_size = min(block_size, input_time - start)
                block_values = values[:, start : start + valid_size]
                if valid_size < block_size:
                    padding = torch.zeros_like(values[:, : block_size - valid_size])
                    block_values = torch.cat((block_values, padding), dim=1)
                left_context = context_vectors[max(0, block_index - 1)]
                right_context = context_vectors[block_index]
                blocks.append(
                    torch.cat(
                        (
                            left_context.unsqueeze(1),
                            block_values,
                            right_context.unsqueeze(1),
                        ),
                        dim=1,
                    )
                )
            return torch.stack(blocks, dim=1)

        @staticmethod
        def relative_attention(
            attention: Any,
            query: Any,
            key: Any,
            value: Any,
            position_embedding_value: Any,
            mask: Any,
        ) -> Any:
            batch_size = query.size(0)
            query_projection = attention.linear_q(query).reshape(
                batch_size,
                -1,
                attention.h,
                attention.d_k,
            )
            key_projection = (
                attention.linear_k(key)
                .reshape(
                    batch_size,
                    -1,
                    attention.h,
                    attention.d_k,
                )
                .transpose(1, 2)
            )
            value_projection = (
                attention.linear_v(value)
                .reshape(
                    batch_size,
                    -1,
                    attention.h,
                    attention.d_k,
                )
                .transpose(1, 2)
            )
            position_projection = (
                attention.linear_pos(position_embedding_value)
                .reshape(
                    position_embedding_value.size(0),
                    -1,
                    attention.h,
                    attention.d_k,
                )
                .transpose(1, 2)
            )

            content_query = (query_projection + attention.pos_bias_u).transpose(1, 2)
            position_query = (query_projection + attention.pos_bias_v).transpose(1, 2)
            content_scores = torch.matmul(
                content_query,
                key_projection.transpose(-2, -1),
            )
            position_scores = torch.matmul(
                position_query,
                position_projection.transpose(-2, -1),
            )
            position_scores = attention.rel_shift(position_scores)
            scores = (content_scores + position_scores) / math.sqrt(attention.d_k)

            blocked = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).masked_fill(blocked, 0.0)
            attended = torch.matmul(attention.dropout(weights), value_projection)
            return (
                attended.transpose(1, 2)
                .contiguous()
                .reshape(
                    batch_size,
                    -1,
                    attention.h * attention.d_k,
                )
            )

        def encode_blocks(self, blocks: Any) -> tuple[Any, Any]:
            batch_size = blocks.size(0)
            mask = self.attention_mask.expand(
                batch_size * block_count,
                -1,
                -1,
            )
            values = blocks
            tempo_layers = []
            for layer_index, layer in enumerate(self.model.encoder.encoders):
                if layer_index:
                    shifted_context = torch.cat(
                        (values[:, :1, -1], values[:, :-1, -1]),
                        dim=1,
                    )
                    values = torch.cat(
                        (shifted_context.unsqueeze(2), values[:, :, 1:]),
                        dim=2,
                    )

                flat_values = values.reshape(
                    batch_size * block_count,
                    sequence_size,
                    feature_size,
                )
                if layer_index == 0:
                    flat_values = flat_values * math.sqrt(feature_size)

                residual = flat_values
                normalized = layer.norm1(flat_values)
                attention_output = self.relative_attention(
                    layer.self_attn,
                    normalized,
                    normalized,
                    normalized,
                    self.position_embedding,
                    mask,
                )
                tempo_layers.append(
                    attention_output.reshape(
                        batch_size,
                        block_count,
                        sequence_size,
                        feature_size,
                    )
                )
                flat_values = residual + layer.dropout(attention_output)
                residual = flat_values
                flat_values = residual + layer.dropout(
                    layer.feed_forward(layer.norm2(flat_values))
                )
                values = flat_values.reshape(
                    batch_size,
                    block_count,
                    sequence_size,
                    feature_size,
                )

            tempo = torch.stack(tempo_layers, dim=-1).sum(dim=-1)
            return values, tempo

        @staticmethod
        def restore_frames(blocks: Any) -> Any:
            first_size = block_size - look_ahead
            segments = [blocks[:, 0, 1 : first_size + 1]]
            block_index = 1
            left_index = center_size
            while left_index + block_size < input_time and block_index < block_count:
                segments.append(
                    blocks[
                        :,
                        block_index,
                        output_offset : output_offset + center_size,
                    ]
                )
                left_index += center_size
                block_index += 1
            segments.append(blocks[:, block_index, output_offset : last_block_size + 1])
            return torch.cat(segments, dim=1)

        def encode(self, values: Any) -> tuple[Any, Any]:
            encoded_blocks, tempo_blocks = self.encode_blocks(
                self.assemble_blocks(values)
            )
            encoded = self.restore_frames(encoded_blocks)
            tempo = self.restore_frames(tempo_blocks)
            if self.model.encoder.normalize_before:
                encoded = self.model.encoder.after_norm(encoded)
                tempo = self.model.encoder.after_norm_t(tempo)
            return encoded, tempo

        def forward(self, features: Any) -> tuple[Any, Any]:
            encoded, tempo = self.encode(self.frontend(features))
            logits = self.model.out_linear(torch.relu(encoded))
            tempo_logits = self.model.out_linear_t(
                self.model.dropout_t(torch.relu(tempo)).mean(dim=1)
            )
            return logits, tempo_logits

    return BeastDeploymentGraph().eval()


def _deployment_graph(
    model: Any,
    model_name: str,
    example_shape: tuple[int, ...],
) -> Any:
    import torch

    deployment_model = copy.deepcopy(model).cpu().eval()
    if model_name == "beast":
        return _beast_deployment_graph(deployment_model, example_shape[1])
    if model_name == "bocktcn":
        _replace_same_conv1d_padding(deployment_model)
        include_downbeats = bool(deployment_model.include_downbeats)
        include_tempo = bool(deployment_model.include_tempo)

        class BockTCNDeploymentGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = deployment_model

            def forward(self, features: Any) -> tuple[Any, ...]:
                output = self.model(features)
                values = [output["event_activations"], output["beats"]]
                if include_downbeats:
                    values.append(output["downbeats"])
                if include_tempo:
                    values.append(output["tempo"])
                return tuple(values)

        return BockTCNDeploymentGraph().eval()

    class SpecTNTDeploymentGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = deployment_model

        def forward(self, features: Any) -> tuple[Any, ...]:
            output = self.model(features)
            return (
                output["logits"],
                output["frame_class_activations"],
                output["event_activations"],
                output["beats"],
                output["downbeats"],
            )

    return SpecTNTDeploymentGraph().eval()


def _dynamic_axes(
    model_name: str,
    output_specs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {"features": {0: "batch"}}
    if model_name == "bocktcn":
        axes["features"][2] = "input_time"
    for output in output_specs:
        name = str(output["name"])
        axes[name] = {0: "batch"}
        shape = output["shape"]
        if model_name == "bocktcn" and len(shape) > 1 and shape[1] == "input_time-4":
            axes[name][1] = "output_time"
    return axes


def _parity_gate(
    graph: Any,
    model_path: Path,
    features: Any,
    output_names: tuple[str, ...],
    *,
    extra_probes: Sequence[tuple[str, Any]] = (),
) -> dict[str, Any]:
    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(
            "batch export parity validation requires the optional onnxruntime package"
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
    maximum_error = 0.0
    per_output = {name: 0.0 for name in output_names}
    probe_errors: dict[str, dict[str, float]] = {}
    probes = (("linspace[-1,1]", features), *extra_probes)
    for probe_name, probe_features in probes:
        with torch.no_grad():
            reference = graph(probe_features)
        if not isinstance(reference, tuple):
            reference = (reference,)
        actual = session.run(
            output_names,
            {"features": probe_features.detach().cpu().numpy()},
        )
        current_errors: dict[str, float] = {}
        for name, expected_value, actual_value in zip(
            output_names,
            reference,
            actual,
            strict=True,
        ):
            expected = expected_value.detach().cpu().numpy()
            observed = np.asarray(actual_value)
            if observed.shape != expected.shape:
                model_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"native {name} parity shape mismatch on {probe_name}: "
                    f"expected {expected.shape}, got {observed.shape}"
                )
            error = float(np.max(np.abs(observed - expected), initial=0.0))
            current_errors[name] = error
            per_output[name] = max(per_output[name], error)
            maximum_error = max(maximum_error, error)
            if not np.allclose(
                observed,
                expected,
                rtol=_PARITY_RTOL,
                atol=_PARITY_ATOL,
            ):
                model_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"native {name} parity failed on {probe_name} with "
                    f"maximum absolute error {error:.9g}"
                )
        probe_errors[probe_name] = current_errors
    return {
        "provider": "CPUExecutionProvider",
        "rtol": _PARITY_RTOL,
        "atol": _PARITY_ATOL,
        "max_abs_error": maximum_error,
        "per_output_max_abs_error": per_output,
        "input_pattern": "linspace[-1,1]",
        "probes": probe_errors,
    }


def export_batch_model_onnx(
    model: Any,
    destination: Path | str,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    input_shape: Sequence[int],
) -> NativeBatchModelArtifact:
    """Export and parity-validate one supported batch model."""

    import torch

    normalized_name = _model_name(model_name)
    example_shape = _input_shape(input_shape)
    digest = _checkpoint_digest(checkpoint_sha256)
    shape_contract, output_names, output_specs = _contracts(
        model,
        normalized_name,
        example_shape,
    )
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("cannot export a model without parameters") from exc
    if parameter.device.type != "cpu":
        raise ValueError("native CPU export requires a CPU model")
    if parameter.dtype != torch.float32:
        raise ValueError("native batch export currently requires float32 weights")

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    graph = _deployment_graph(model, normalized_name, example_shape)
    feature_count = int(np.prod(example_shape, dtype=np.int64))
    features = torch.linspace(
        -1.0,
        1.0,
        steps=feature_count,
        dtype=torch.float32,
    ).reshape(example_shape)
    dynamic_axes = _dynamic_axes(normalized_name, output_specs)

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
        warnings.filterwarnings(
            "ignore",
            message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            graph,
            features,
            str(destination_path),
            input_names=("features",),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=BATCH_ONNX_OPSET_VERSION,
            dynamo=False,
            do_constant_folding=True,
        )

    try:
        import onnx
    except ImportError as exc:
        destination_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ONNX export validation requires the optional 'onnx' package"
        ) from exc
    onnx.checker.check_model(onnx.load(str(destination_path)))
    extra_probes: tuple[tuple[str, Any], ...] = ()
    if normalized_name == "beast":
        extra_probes = (
            (
                "normal(seed=780887,batch=1)",
                torch.randn(
                    example_shape,
                    generator=torch.Generator().manual_seed(780887),
                ),
            ),
            (
                "normal(seed=780888,batch=2)",
                torch.randn(
                    (2, *example_shape[1:]),
                    generator=torch.Generator().manual_seed(780888),
                ),
            ),
        )
    parity = _parity_gate(
        graph,
        destination_path,
        features,
        output_names,
        extra_probes=extra_probes,
    )

    artifact_key, config_sha256 = _artifact_identity(
        model_name=normalized_name,
        model_config=model_config,
        checkpoint_sha256=digest,
        shape_contract=shape_contract,
        output_contract=output_specs,
        torch_version=str(torch.__version__),
    )
    manifest_path = destination_path.with_suffix(destination_path.suffix + ".json")
    manifest = {
        "schema": NATIVE_BATCH_SCHEMA,
        "abi_version": NATIVE_BATCH_ABI_VERSION,
        "artifact_key": artifact_key,
        "model_family": normalized_name,
        "task": "beat_tracking",
        "graph_contract": "batch-feature-tensor-to-named-model-outputs",
        "checkpoint_sha256": digest,
        "model_config_sha256": config_sha256,
        "onnx": {
            "filename": destination_path.name,
            "opset": BATCH_ONNX_OPSET_VERSION,
            "sha256": _sha256_file(destination_path),
            "size_bytes": destination_path.stat().st_size,
        },
        "exporter": {
            "name": "torch.onnx.legacy",
            "torch_version": str(torch.__version__),
        },
        "inputs": ["features"],
        "outputs": list(output_names),
        "shape_contract": shape_contract,
        "output_contract": output_specs,
        "export_example_shape": list(example_shape),
        "deployment_transforms": {
            "beast": ["contextual_blocks_to_functional_tensor_graph"],
            "bocktcn": ["conv1d_same_to_explicit_symmetric_padding"],
            "spectnt": [],
        }[normalized_name],
        "parity_validation": parity,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NativeBatchModelArtifact(
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
) -> NativeBatchModelArtifact | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        onnx_record = payload.get("onnx")
        if not isinstance(onnx_record, dict):
            return None
        if (
            payload.get("schema") != NATIVE_BATCH_SCHEMA
            or payload.get("abi_version") != NATIVE_BATCH_ABI_VERSION
            or payload.get("artifact_key") != artifact_key
            or onnx_record.get("filename") != model_path.name
            or int(onnx_record.get("size_bytes", -1)) != model_path.stat().st_size
            or str(onnx_record.get("sha256")) != _sha256_file(model_path)
        ):
            return None
        return NativeBatchModelArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=payload,
            cache_hit=True,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def ensure_batch_model_onnx(
    model: Any,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    checkpoint_sha256: str,
    input_shape: Sequence[int],
    cache_root: Path | str | None = None,
) -> NativeBatchModelArtifact:
    """Return a verified cached batch graph, exporting it when absent or stale."""

    import torch

    normalized_name = _model_name(model_name)
    example_shape = _input_shape(input_shape)
    digest = _checkpoint_digest(checkpoint_sha256)
    shape_contract, _, output_specs = _contracts(
        model,
        normalized_name,
        example_shape,
    )
    artifact_key, _ = _artifact_identity(
        model_name=normalized_name,
        model_config=model_config,
        checkpoint_sha256=digest,
        shape_contract=shape_contract,
        output_contract=output_specs,
        torch_version=str(torch.__version__),
    )
    root = (
        _cache_root() if cache_root is None else Path(cache_root).expanduser().resolve()
    )
    artifact_directory = root / normalized_name / artifact_key
    model_path = artifact_directory / "batch.onnx"
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
            exported = export_batch_model_onnx(
                model,
                temporary_model,
                model_name=normalized_name,
                model_config=model_config,
                checkpoint_sha256=digest,
                input_shape=example_shape,
            )
            os.replace(temporary_model, model_path)
            os.replace(exported.manifest_path, manifest_path)
        return NativeBatchModelArtifact(
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=exported.manifest,
            cache_hit=False,
        )


class OnnxBatchModelSession:
    """CPU ONNX Runtime host for a validated batch artifact."""

    def __init__(
        self,
        artifact: NativeBatchModelArtifact,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "native batch inference requires the optional onnxruntime package"
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
            raise ValueError(f"unexpected native batch inputs: {input_names!r}")
        if output_names != artifact.output_names:
            raise ValueError(f"unexpected native batch outputs: {output_names!r}")

        self.artifact = artifact
        self.output_names = output_names
        self.provider = "CPUExecutionProvider"
        self._input_contract = artifact.manifest["shape_contract"]["input"]

    def infer(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Run one batch while enforcing the artifact's fixed-axis contract."""

        values = np.ascontiguousarray(features, dtype=np.float32)
        expected_shape = self._input_contract["shape"]
        if values.ndim != len(expected_shape):
            raise ValueError(
                f"native batch input must have rank {len(expected_shape)}, "
                f"got rank {values.ndim}"
            )
        for axis, (actual, expected) in enumerate(
            zip(values.shape, expected_shape, strict=True)
        ):
            if isinstance(expected, int) and actual != expected:
                raise ValueError(
                    f"native batch input axis {axis} must be {expected}, got {actual}"
                )
        constraints = self._input_contract.get("constraints", {})
        minimum_time = constraints.get("input_time_min")
        if minimum_time is not None and values.shape[2] < int(minimum_time):
            raise ValueError(
                f"native batch input requires at least {minimum_time} time frames"
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
    "BATCH_ONNX_OPSET_VERSION",
    "NATIVE_BATCH_ABI_VERSION",
    "NATIVE_BATCH_SCHEMA",
    "NativeBatchModelArtifact",
    "OnnxBatchModelSession",
    "SUPPORTED_BATCH_MODELS",
    "UNSUPPORTED_BATCH_MODELS",
    "UnsupportedNativeBatchModelError",
    "ensure_batch_model_onnx",
    "export_batch_model_onnx",
]
