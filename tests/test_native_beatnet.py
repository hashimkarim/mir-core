from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

from mir_core.checkpoints import beatnet_plus_base_checkpoint_path
from mir_core.models.beatnet.beatnet_plus import BeatNetPlusOnline
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.models.beatnet.multihead import MultiHeadBeatNet
from mir_core.native import (
    OnnxBeatNetStreamingSession,
    ensure_streaming_beatnet_onnx,
    export_streaming_beatnet_onnx,
    resolve_streaming_backend,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

_PARITY_SCHEMA = "mir.native-streaming-parity-validation/v1"
_PARITY_SEQUENCE_LENGTH = 12


def _dance_model_class() -> type[torch.nn.Module]:
    """Exercise optional DanceBeat models without depending on their branch."""

    module_name = "mir_core.models.beatnet.dance"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        pytest.skip("Optional DanceBeat model is not present in this checkout")
    return module.DanceBeatNetCRNN


def _checkpoint_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _assert_current_parity_manifest(artifact: object) -> None:
    manifest = artifact.manifest
    parity = manifest["parity_validation"]
    output_names = manifest["outputs"]

    assert parity["schema"] == _PARITY_SCHEMA
    assert parity["version"] == 1
    assert parity["passed"] is True
    assert parity["reference"] == "pytorch-eval-cpu"
    assert parity["provider"] == "CPUExecutionProvider"
    assert parity["validated_outputs"] == output_names
    assert parity["onnx_sha256"] == manifest["onnx"]["sha256"]
    assert set(parity["per_output_max_abs_error"]) == set(output_names)
    assert len(parity["probes"]) == _PARITY_SEQUENCE_LENGTH
    for frame_index, probe in enumerate(parity["probes"]):
        assert probe["name"] == f"sequence_frame_{frame_index:02d}"
        assert probe["frame_index"] == frame_index
        assert set(probe["per_output_max_abs_error"]) == set(output_names)
    assert np.isfinite(parity["max_abs_error"])
    assert parity["max_abs_error"] <= 2.0e-6 + 2.0e-5 * _PARITY_SEQUENCE_LENGTH


def _add_onnx_output_offset(path: Path, output_name: str) -> None:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path))
    original_name = f"{output_name}_before_parity_test_offset"
    producers = [node for node in model.graph.node if output_name in node.output]
    assert len(producers) == 1
    producer = producers[0]
    producer.output[list(producer.output).index(output_name)] = original_name
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == output_name:
                node.input[index] = original_name
    offset_name = f"{output_name}_parity_test_offset"
    model.graph.initializer.append(
        helper.make_tensor(offset_name, TensorProto.FLOAT, [], [0.25])
    )
    model.graph.node.append(
        helper.make_node(
            "Add",
            (original_name, offset_name),
            (output_name,),
            name=f"perturb_{output_name}_for_parity_test",
        )
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _reference_frame(
    model: torch.nn.Module,
    model_name: str,
    feature: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1).transpose(1, 2)
        activations = torch.stack(
            (
                probabilities[..., 0] + probabilities[..., 1],
                probabilities[..., 1],
            ),
            dim=-1,
        )
    return activations[0, 0].numpy()


def _reference_dance_frame(
    model: torch.nn.Module,
    feature: torch.Tensor,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        output = model(feature)
    return {
        "activations": output["event_activations"][0, 0].numpy(),
        "beats": output["beats"][0, 0].numpy(),
        "downbeats": output["downbeats"][0, 0].numpy(),
        "dancebeats": output["dancebeats"][0, 0].numpy(),
    }


@pytest.mark.parametrize(
    ("model_name", "model", "input_dim", "num_layers"),
    [
        ("beatnet", BeatNetCRNN(), 272, 2),
        ("beatnet_plus", BeatNetPlusOnline(), 288, 4),
    ],
)
def test_native_streaming_sequence_matches_pytorch_recurrent_state(
    tmp_path: Path,
    model_name: str,
    model: torch.nn.Module,
    input_dim: int,
    num_layers: int,
) -> None:
    torch.manual_seed(91)
    model.eval()
    model.reset_hidden()
    config = {
        "name": model_name,
        "input_dim": input_dim,
        "hidden_dim": 150,
        "num_layers": num_layers,
    }
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / f"{model_name}.onnx",
        model_name=model_name,
        model_config=config,
        checkpoint_sha256=_checkpoint_digest(model),
    )
    _assert_current_parity_manifest(artifact)
    native = OnnxBeatNetStreamingSession(artifact)
    frames = torch.randn(16, input_dim)

    for frame in frames:
        feature = frame.reshape(1, 1, -1)
        expected = _reference_frame(model, model_name, feature)
        actual = native.infer(frame.numpy())
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    np.testing.assert_allclose(
        native.hidden,
        model.hidden.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        native.cell,
        model.cell.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )


def test_packaged_beatnet_plus_checkpoint_streams_with_native_parity(
    tmp_path: Path,
) -> None:
    checkpoint = beatnet_plus_base_checkpoint_path()
    model = BeatNetPlusOnline().eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    model.reset_hidden()
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / "packaged-beatnet-plus.onnx",
        model_name="beatnet_plus",
        model_config={
            "name": "beatnet_plus",
            "input_dim": 288,
            "hidden_dim": 150,
            "num_layers": 4,
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    native = OnnxBeatNetStreamingSession(artifact)

    for frame in torch.randn(12, 288):
        expected = _reference_frame(
            model,
            "beatnet_plus",
            frame.reshape(1, 1, -1),
        )
        actual = native.infer(frame.numpy())
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_native_multihead_streams_one_explicit_genre_head(
    tmp_path: Path,
) -> None:
    torch.manual_seed(717)
    model = MultiHeadBeatNet(
        genre_labels=["salsa", "other"],
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
    ).eval()
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / "multihead-salsa.onnx",
        model_name="multihead",
        model_config={
            "name": "multihead_beatnet",
            "genre_label": "salsa",
            "input_dim": 32,
            "hidden_dim": 8,
            "num_layers": 1,
        },
        checkpoint_sha256=_checkpoint_digest(model),
    )
    _assert_current_parity_manifest(artifact)
    native = OnnxBeatNetStreamingSession(artifact)
    head = model.heads["salsa"]
    hidden = torch.zeros((1, 1, 8))
    cell = torch.zeros_like(hidden)

    assert artifact.model_family == "multihead_beatnet"
    assert artifact.manifest["output_contract"]["primary_channels"] == [
        "all_beats",
        "downbeats",
    ]
    for frame in torch.randn(12, 32):
        with torch.no_grad():
            convolution = model.forward_conv(frame.reshape(1, 1, -1))
            projected = head["linear0"](convolution).reshape(1, 1, -1)
            recurrent, (hidden, cell) = head["lstm"](
                projected,
                (hidden, cell),
            )
            probabilities = torch.softmax(head["linear"](recurrent), dim=-1)
            expected = torch.stack(
                (
                    probabilities[..., 0] + probabilities[..., 1],
                    probabilities[..., 1],
                ),
                dim=-1,
            )[0, 0].numpy()
        actual = native.infer(frame.numpy())
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    np.testing.assert_allclose(native.hidden, hidden.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(native.cell, cell.numpy(), rtol=2e-5, atol=2e-6)


def test_native_multihead_requires_an_explicit_known_genre(tmp_path: Path) -> None:
    model = MultiHeadBeatNet(
        genre_labels=["salsa", "other"],
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
    ).eval()
    digest = _checkpoint_digest(model)

    with pytest.raises(ValueError, match="genre_label"):
        export_streaming_beatnet_onnx(
            model,
            tmp_path / "missing.onnx",
            model_name="multihead_beatnet",
            model_config={"name": "multihead_beatnet"},
            checkpoint_sha256=digest,
        )
    with pytest.raises(ValueError, match="unknown MultiHeadBeatNet genre"):
        export_streaming_beatnet_onnx(
            model,
            tmp_path / "unknown.onnx",
            model_name="multihead_beatnet",
            model_config={"name": "multihead_beatnet", "genre_label": "waltz"},
            checkpoint_sha256=digest,
        )


@pytest.mark.parametrize(
    ("tracking_target", "accent_output"),
    [("beat", "downbeats"), ("dance", "dancebeats")],
)
def test_native_dance_streaming_matches_every_head_and_recurrent_state(
    tmp_path: Path,
    tracking_target: str,
    accent_output: str,
) -> None:
    torch.manual_seed(812)
    model = _dance_model_class()(
        input_dim=32,
        hidden_dim=8,
        num_layers=2,
        tracking_target=tracking_target,
    ).eval()
    model.reset_hidden()
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / f"dance-{tracking_target}.onnx",
        model_name="beatnet_dance" if tracking_target == "beat" else "DanceBeatNet",
        model_config={
            "name": "dance_beatnet",
            "input_dim": 32,
            "hidden_dim": 8,
            "num_layers": 2,
            "tracking_target": tracking_target,
        },
        checkpoint_sha256=_checkpoint_digest(model),
    )
    _assert_current_parity_manifest(artifact)
    native = OnnxBeatNetStreamingSession(artifact)

    assert artifact.model_family == "dance_beatnet"
    assert artifact.manifest["outputs"] == [
        "activations",
        "beats",
        "downbeats",
        "dancebeats",
        "next_hidden",
        "next_cell",
    ]
    assert artifact.manifest["output_contract"] == {
        "primary_output": "activations",
        "primary_channels": ["all_beats", accent_output],
        "tracking_target": tracking_target,
        "independent_sigmoid_heads": {
            "all_beats": "beats",
            "downbeats": "downbeats",
            "dancebeats": "dancebeats",
        },
    }

    for frame in torch.randn(16, 32):
        feature = frame.reshape(1, 1, -1)
        expected = _reference_dance_frame(model, feature)
        actual = native.infer_outputs(frame.numpy())
        assert actual.keys() == expected.keys()
        for output_name in expected:
            np.testing.assert_allclose(
                actual[output_name],
                expected[output_name],
                rtol=2e-5,
                atol=2e-6,
            )

    np.testing.assert_allclose(
        native.hidden,
        model.hidden.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        native.cell,
        model.cell.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )


def test_native_dance_default_deployment_geometry_matches_pytorch(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    model = _dance_model_class()().eval()
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / "dance-default.onnx",
        model_name="dance_beatnet",
        model_config={
            "name": "dance_beatnet",
            "input_dim": 272,
            "hidden_dim": 150,
            "num_layers": 2,
            "tracking_target": "dance",
        },
        checkpoint_sha256=_checkpoint_digest(model),
    )
    native = OnnxBeatNetStreamingSession(artifact)
    frame = torch.linspace(-1.0, 1.0, 272)
    expected = _reference_dance_frame(model, frame.reshape(1, 1, -1))
    actual = native.infer_outputs(frame.numpy())

    assert artifact.manifest["streaming_state"] == {
        "batch_size": 1,
        "time_steps": 1,
        "input_dim": 272,
        "hidden_dim": 150,
        "num_layers": 2,
        "dtype": "float32",
    }
    for output_name in expected:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=2e-5,
            atol=2e-6,
        )


def test_native_artifact_cache_is_content_addressed_and_verified(
    tmp_path: Path,
) -> None:
    model = BeatNetCRNN().eval()
    config = {"name": "beatnet", "input_dim": 272, "hidden_dim": 150, "num_layers": 2}
    kwargs = {
        "model_name": "beatnet",
        "model_config": config,
        "checkpoint_sha256": _checkpoint_digest(model),
        "cache_root": tmp_path,
    }

    exported = ensure_streaming_beatnet_onnx(model, **kwargs)
    reused = ensure_streaming_beatnet_onnx(model, **kwargs)

    assert exported.cache_hit is False
    assert reused.cache_hit is True
    assert reused.model_path == exported.model_path
    assert (
        reused.manifest["onnx"]["sha256"]
        == hashlib.sha256(reused.model_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "damage",
    ["missing", "failed", "stale_version", "stale_graph", "unknown_field"],
)
def test_native_cache_rejects_missing_failed_or_stale_parity(
    tmp_path: Path,
    damage: str,
) -> None:
    torch.manual_seed(440)
    model = BeatNetCRNN(input_dim=32, hidden_dim=8, num_layers=1).eval()
    kwargs = {
        "model_name": "beatnet",
        "model_config": {
            "name": "beatnet",
            "input_dim": 32,
            "hidden_dim": 8,
            "num_layers": 1,
        },
        "checkpoint_sha256": _checkpoint_digest(model),
        "cache_root": tmp_path,
    }
    exported = ensure_streaming_beatnet_onnx(model, **kwargs)
    payload = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    if damage == "missing":
        payload.pop("parity_validation")
    elif damage == "failed":
        payload["parity_validation"]["passed"] = False
    elif damage == "stale_version":
        payload["parity_validation"]["version"] = 0
    elif damage == "stale_graph":
        payload["parity_validation"]["onnx_sha256"] = "0" * 64
    else:
        payload["parity_validation"]["unverified_claim"] = True
    exported.manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    repaired = ensure_streaming_beatnet_onnx(model, **kwargs)

    assert repaired.cache_hit is False
    _assert_current_parity_manifest(repaired)


def test_native_dance_cache_verifies_hash_and_tracks_projection_semantics(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    dance_model = _dance_model_class()(
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
        tracking_target="dance",
    ).eval()
    shared_config = {
        "name": "dance_beatnet",
        "input_dim": 32,
        "hidden_dim": 8,
        "num_layers": 1,
    }
    checkpoint_sha256 = _checkpoint_digest(dance_model)

    exported = ensure_streaming_beatnet_onnx(
        dance_model,
        model_name="dance_beatnet",
        model_config=shared_config,
        checkpoint_sha256=checkpoint_sha256,
        cache_root=tmp_path,
    )
    reused = ensure_streaming_beatnet_onnx(
        dance_model,
        model_name="dance_beatnet",
        model_config=shared_config,
        checkpoint_sha256=checkpoint_sha256,
        cache_root=tmp_path,
    )
    assert reused.cache_hit is True
    assert reused.model_path == exported.model_path

    reused.model_path.write_bytes(reused.model_path.read_bytes() + b"corrupt")
    repaired = ensure_streaming_beatnet_onnx(
        dance_model,
        model_name="dance_beatnet",
        model_config=shared_config,
        checkpoint_sha256=checkpoint_sha256,
        cache_root=tmp_path,
    )
    assert repaired.cache_hit is False
    assert (
        repaired.manifest["onnx"]["sha256"]
        == hashlib.sha256(repaired.model_path.read_bytes()).hexdigest()
    )

    beat_model = _dance_model_class()(
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
        tracking_target="beat",
    ).eval()
    beat_model.load_state_dict(dance_model.state_dict())
    beat_artifact = ensure_streaming_beatnet_onnx(
        beat_model,
        model_name="dance_beatnet",
        model_config=shared_config,
        checkpoint_sha256=checkpoint_sha256,
        cache_root=tmp_path,
    )
    assert beat_artifact.model_path != repaired.model_path
    assert beat_artifact.manifest["output_contract"]["tracking_target"] == "beat"


def test_native_dance_rejects_mismatched_tracking_target_config(
    tmp_path: Path,
) -> None:
    model = _dance_model_class()(
        input_dim=32,
        hidden_dim=8,
        num_layers=1,
        tracking_target="dance",
    ).eval()

    with pytest.raises(ValueError, match="tracking_target does not match"):
        export_streaming_beatnet_onnx(
            model,
            tmp_path / "mismatched-target.onnx",
            model_name="dance_beatnet",
            model_config={"tracking_target": "beat"},
            checkpoint_sha256=_checkpoint_digest(model),
        )


@pytest.mark.parametrize(
    ("model_name", "perturbed_output"),
    [("beatnet", "next_cell"), ("dance_beatnet", "dancebeats")],
)
def test_streaming_export_rejects_and_removes_perturbed_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    perturbed_output: str,
) -> None:
    torch.manual_seed(991)
    if model_name == "dance_beatnet":
        model: torch.nn.Module = _dance_model_class()(
            input_dim=32,
            hidden_dim=8,
            num_layers=1,
            tracking_target="dance",
        ).eval()
    else:
        model = BeatNetCRNN(input_dim=32, hidden_dim=8, num_layers=1).eval()
    real_export = torch.onnx.export

    def export_then_perturb(*args: object, **kwargs: object) -> None:
        real_export(*args, **kwargs)
        _add_onnx_output_offset(Path(str(args[2])), perturbed_output)

    monkeypatch.setattr(torch.onnx, "export", export_then_perturb)
    destination = tmp_path / f"perturbed-{model_name}.onnx"

    with pytest.raises(
        RuntimeError,
        match=rf"{perturbed_output} parity failed at frame 0",
    ):
        export_streaming_beatnet_onnx(
            model,
            destination,
            model_name=model_name,
            model_config={
                "name": model_name,
                "input_dim": 32,
                "hidden_dim": 8,
                "num_layers": 1,
                "tracking_target": "dance",
            },
            checkpoint_sha256=_checkpoint_digest(model),
        )

    assert not destination.exists()
    assert not destination.with_suffix(".onnx.json").exists()


def test_native_session_reset_restores_initial_recurrent_result(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = BeatNetCRNN().eval()
    frame = np.linspace(-1.0, 1.0, 272, dtype=np.float32)
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / "beatnet.onnx",
        model_name="beatnet",
        model_config={"name": "beatnet", "input_dim": 272},
        checkpoint_sha256=_checkpoint_digest(model),
    )
    native = OnnxBeatNetStreamingSession(artifact)

    first = native.infer(frame).copy()
    native.infer(frame)
    native.reset_hidden()

    np.testing.assert_array_equal(native.infer(frame), first)


def test_native_dance_session_reset_restores_every_head(tmp_path: Path) -> None:
    torch.manual_seed(73)
    model = _dance_model_class()(input_dim=32, hidden_dim=8, num_layers=1).eval()
    frame = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    artifact = export_streaming_beatnet_onnx(
        model,
        tmp_path / "dance-reset.onnx",
        model_name="dance_beatnet",
        model_config={"name": "dance_beatnet", "input_dim": 32},
        checkpoint_sha256=_checkpoint_digest(model),
    )
    native = OnnxBeatNetStreamingSession(artifact)

    first = native.infer_outputs(frame)
    native.infer_outputs(frame)
    native.reset_hidden()
    reset = native.infer_outputs(frame)

    for output_name in first:
        np.testing.assert_array_equal(reset[output_name], first[output_name])


def test_auto_backend_prefers_native_cpu_but_preserves_explicit_cuda() -> None:
    assert resolve_streaming_backend("auto", device="auto") == "onnxruntime"
    assert resolve_streaming_backend("auto", device="cpu") == "onnxruntime"
    assert resolve_streaming_backend("auto", device="cuda") == "torch"
    assert resolve_streaming_backend("torch", device="auto") == "torch"


def test_explicit_native_cuda_is_rejected_until_cuda_parity_gate_exists() -> None:
    with pytest.raises(ValueError, match="CPU-only"):
        resolve_streaming_backend("onnxruntime", device="cuda")
