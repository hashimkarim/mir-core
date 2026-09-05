from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mir_core.checkpoints import beast_baseline_checkpoint_path
from mir_core.checkpoints.beast import BASE_CHECKPOINT_SHA256 as BEAST_BASE_SHA256
from mir_core.models import BEAST, SpecTNT
from mir_core.models.bock_tcn.tcn import BockTCN
from mir_core.models.spectnt import ResFrontEnd
from mir_core.native import (
    NATIVE_BATCH_SCHEMA,
    OnnxBatchModelSession,
    ensure_batch_model_onnx,
    export_batch_model_onnx,
)
from mir_core.native.batch import _beast_deployment_graph

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_spectnt() -> SpecTNT:
    return SpecTNT(
        n_channels=8,
        n_frequencies=2,
        n_times=4,
        spectral_dmodel=4,
        spectral_nheads=1,
        spectral_dimff=4,
        temporal_dmodel=4,
        temporal_nheads=1,
        temporal_dimff=4,
        embed_dim=4,
        n_blocks=1,
        dropout=0.0,
        fe_model=ResFrontEnd(
            in_channels=6,
            out_channels=8,
            freq_pooling=(2, 2, 2),
            time_pooling=(2, 2, 1),
        ),
    ).eval()


def test_native_bocktcn_dynamic_batch_and_time_matches_all_heads(
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)
    model = BockTCN(
        n_filters=4,
        n_dilations=3,
        kernel_size=5,
        dropout_rate=0.0,
        include_downbeats=True,
        include_tempo=True,
    ).eval()
    artifact = export_batch_model_onnx(
        model,
        tmp_path / "bocktcn.onnx",
        model_name="bock_tcn",
        model_config={
            "n_filters": 4,
            "n_dilations": 3,
            "kernel_size": 5,
            "include_downbeats": True,
            "include_tempo": True,
        },
        checkpoint_sha256=_state_digest(model),
        input_shape=(1, 1, 48, 81),
    )
    native = OnnxBatchModelSession(artifact)

    assert artifact.model_family == "bocktcn"
    assert artifact.manifest["schema"] == NATIVE_BATCH_SCHEMA
    assert artifact.output_names == (
        "activations",
        "beats",
        "downbeats",
        "tempo",
    )
    assert artifact.manifest["shape_contract"]["mode"] == "dynamic_batch_and_time"
    assert artifact.manifest["deployment_transforms"] == [
        "conv1d_same_to_explicit_symmetric_padding"
    ]
    assert artifact.manifest["parity_validation"]["max_abs_error"] < 2e-6

    features = torch.randn(2, 1, 63, 81)
    with torch.no_grad():
        reference = model(features)
    actual = native.infer(features.numpy())
    expected = {
        "activations": reference["event_activations"].numpy(),
        "beats": reference["beats"].numpy(),
        "downbeats": reference["downbeats"].numpy(),
        "tempo": reference["tempo"].numpy(),
    }
    assert actual.keys() == expected.keys()
    for output_name in expected:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=2e-5,
            atol=2e-6,
        )
    assert actual["activations"].shape == (2, 59, 2)


def test_native_spectnt_dynamic_batch_with_fixed_feature_shape(
    tmp_path: Path,
) -> None:
    torch.manual_seed(202)
    model = _small_spectnt()
    artifact = export_batch_model_onnx(
        model,
        tmp_path / "spectnt.onnx",
        model_name="spec_tnt",
        model_config={
            "n_channels": 8,
            "n_frequencies": 2,
            "n_times": 4,
            "n_blocks": 1,
        },
        checkpoint_sha256=_state_digest(model),
        input_shape=(1, 6, 16, 16),
    )
    native = OnnxBatchModelSession(artifact)

    assert artifact.model_family == "spectnt"
    assert artifact.output_names == (
        "logits",
        "frame_class_activations",
        "event_activations",
        "beats",
        "downbeats",
    )
    assert artifact.manifest["shape_contract"]["mode"] == "dynamic_batch_fixed_features"
    assert artifact.manifest["shape_contract"]["input"]["shape"] == [
        "batch",
        6,
        16,
        16,
    ]
    assert artifact.manifest["parity_validation"]["max_abs_error"] < 2e-6

    features = torch.randn(2, 6, 16, 16)
    with torch.no_grad():
        reference = model(features)
    actual = native.infer(features.numpy())
    expected = {
        "logits": reference["logits"].numpy(),
        "frame_class_activations": reference["frame_class_activations"].numpy(),
        "event_activations": reference["event_activations"].numpy(),
        "beats": reference["beats"].numpy(),
        "downbeats": reference["downbeats"].numpy(),
    }
    for output_name in expected:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=2e-5,
            atol=2e-6,
        )

    with pytest.raises(ValueError, match="axis 3 must be 16"):
        native.infer(np.zeros((1, 6, 16, 20), dtype=np.float32))


def test_native_batch_cache_is_content_addressed_and_hash_verified(
    tmp_path: Path,
) -> None:
    torch.manual_seed(303)
    model = BockTCN(
        n_filters=2,
        n_dilations=1,
        dropout_rate=0.0,
    ).eval()
    kwargs = {
        "model_name": "bocktcn",
        "model_config": {
            "n_filters": 2,
            "n_dilations": 1,
            "include_downbeats": False,
        },
        "checkpoint_sha256": _state_digest(model),
        "input_shape": (1, 1, 32, 81),
        "cache_root": tmp_path,
    }

    exported = ensure_batch_model_onnx(model, **kwargs)
    reused = ensure_batch_model_onnx(model, **kwargs)
    assert exported.cache_hit is False
    assert reused.cache_hit is True
    assert reused.model_path == exported.model_path
    assert reused.manifest["onnx"]["sha256"] == _file_digest(reused.model_path)

    reused.model_path.write_bytes(reused.model_path.read_bytes() + b"corrupt")
    repaired = ensure_batch_model_onnx(model, **kwargs)
    assert repaired.cache_hit is False
    assert repaired.manifest["onnx"]["sha256"] == _file_digest(repaired.model_path)


def test_beast_functional_context_graph_matches_internal_boundaries() -> None:
    torch.manual_seed(404)
    model = BEAST(
        dmodel=32,
        nhead=4,
        d_hid=64,
        nlayers=3,
        dropout=0.0,
        left_size=16,
        center_size=8,
        right_size=4,
    ).eval()
    graph = _beast_deployment_graph(model, 40)
    features = torch.randn(2, 40, 128)
    captured: dict[str, torch.Tensor] = {}

    def capture_encoder_input(
        _module: torch.nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured["frontend"] = arguments[0].detach().clone()

    def capture_first_layer_input(
        _module: torch.nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured["blocks"] = arguments[0].detach().clone()
        captured["mask"] = arguments[1].detach().clone()

    def capture_encoder_output(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        captured["encoded"] = output[0].detach().clone()
        captured["tempo"] = output[2].detach().clone()

    handles = (
        model.encoder.register_forward_pre_hook(capture_encoder_input),
        model.encoder.encoders[0].register_forward_pre_hook(capture_first_layer_input),
        model.encoder.register_forward_hook(capture_encoder_output),
    )
    try:
        with torch.no_grad():
            reference = model(features)
    finally:
        for handle in handles:
            handle.remove()

    with torch.no_grad():
        frontend = graph.frontend(features)
        blocks = graph.assemble_blocks(frontend)
        encoded, tempo = graph.encode(frontend)
        actual = graph(features)

    torch.testing.assert_close(frontend, captured["frontend"], rtol=0, atol=0)
    torch.testing.assert_close(blocks, captured["blocks"], rtol=0, atol=0)
    expected_mask = graph.attention_mask.unsqueeze(1).expand(2, 3, -1, -1)
    torch.testing.assert_close(expected_mask, captured["mask"], rtol=0, atol=0)
    torch.testing.assert_close(encoded, captured["encoded"], rtol=0, atol=0)
    torch.testing.assert_close(tempo, captured["tempo"], rtol=0, atol=0)
    for actual_value, reference_value in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_value, reference_value, rtol=0, atol=0)


def test_packaged_beast_checkpoint_exports_with_dynamic_batch_parity(
    tmp_path: Path,
) -> None:
    checkpoint_path = beast_baseline_checkpoint_path()
    assert _file_digest(checkpoint_path) == BEAST_BASE_SHA256
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = BEAST(nhead=8, d_hid=1024).eval()
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    artifact = export_batch_model_onnx(
        model,
        tmp_path / "beast.onnx",
        model_name="beast",
        model_config={"nhead": 8, "d_hid": 1024, "nlayers": 9},
        checkpoint_sha256=BEAST_BASE_SHA256,
        input_shape=(1, 320, 128),
    )
    native = OnnxBatchModelSession(artifact)

    assert artifact.model_family == "beast"
    assert artifact.output_names == ("logits", "tempo_logits")
    assert artifact.manifest["shape_contract"]["mode"] == (
        "dynamic_batch_fixed_context_and_features"
    )
    assert artifact.manifest["shape_contract"]["input"]["shape"] == [
        "batch",
        320,
        128,
    ]
    assert artifact.manifest["deployment_transforms"] == [
        "contextual_blocks_to_functional_tensor_graph"
    ]
    assert set(artifact.manifest["parity_validation"]["probes"]) == {
        "linspace[-1,1]",
        "normal(seed=780887,batch=1)",
        "normal(seed=780888,batch=2)",
    }
    assert artifact.manifest["parity_validation"]["max_abs_error"] < 2e-5

    for seed, batch_size in ((505, 1), (506, 2)):
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(batch_size, 320, 128, generator=generator)
        with torch.no_grad():
            reference_logits, reference_tempo = model(features)
        actual = native.infer(features.numpy())
        np.testing.assert_allclose(
            actual["logits"],
            reference_logits.numpy(),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            actual["tempo_logits"],
            reference_tempo.numpy(),
            rtol=2e-5,
            atol=2e-6,
        )

    with pytest.raises(ValueError, match="axis 1 must be 320"):
        native.infer(np.zeros((1, 321, 128), dtype=np.float32))


def test_beast_native_export_rejects_broken_short_sequence_path(
    tmp_path: Path,
) -> None:
    model = BEAST(
        dmodel=32,
        nhead=4,
        d_hid=64,
        nlayers=2,
        dropout=0.0,
        left_size=16,
        center_size=8,
        right_size=4,
    ).eval()
    with pytest.raises(ValueError, match="more than 28 frames"):
        export_batch_model_onnx(
            model,
            tmp_path / "short-beast.onnx",
            model_name="beast",
            model_config={"dmodel": 32, "nlayers": 2},
            checkpoint_sha256=_state_digest(model),
            input_shape=(1, 28, 128),
        )
