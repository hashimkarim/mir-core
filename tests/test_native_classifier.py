from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mir_core.checkpoints import load_trained_model_bundle
from mir_core.models import (
    BeatNetConvClassifier,
    BeatNetLogSpectCNN,
    EmbeddingStatsMLP,
    FramewiseEmbeddingMLP,
    GenreClassifier,
    MFCCCNN,
    MelCNN,
    MelCNNAttention,
)
from mir_core.native import (
    NATIVE_CLASSIFIER_SCHEMA,
    OnnxClassifierSession,
    ensure_classifier_onnx,
    export_classifier_onnx,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

LABELS = ("candombe", "brid", "salsa")


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _architecture_case(
    architecture: str,
) -> tuple[torch.nn.Module, tuple[int, ...], dict[str, object]]:
    if architecture == "mel_cnn":
        return (
            MelCNN(num_classes=3, n_mels=32, dropout=0.0).eval(),
            (1, 1, 32, 24),
            {"arch": architecture, "n_mels": 32, "dropout": 0.0},
        )
    if architecture == "mfcc_cnn":
        return (
            MFCCCNN(num_classes=3, n_features=24, dropout=0.0).eval(),
            (1, 1, 24, 24),
            {"arch": architecture, "n_features": 24, "dropout": 0.0},
        )
    if architecture == "mel_cnn_attention":
        return (
            MelCNNAttention(num_classes=3, n_mels=32, dropout=0.0).eval(),
            (1, 1, 32, 24),
            {"arch": architecture, "n_mels": 32, "dropout": 0.0},
        )
    if architecture == "beatnet_log_spect_cnn":
        return (
            BeatNetLogSpectCNN(
                num_classes=3,
                feature_dim=32,
                dropout=0.0,
            ).eval(),
            (1, 1, 32, 24),
            {"arch": architecture, "feature_dim": 32, "dropout": 0.0},
        )
    if architecture == "embedding_stats_mlp":
        return (
            EmbeddingStatsMLP(
                num_classes=3,
                embedding_dim=12,
                hidden_dim=8,
                dropout=0.0,
            ).eval(),
            (1, 1, 12, 5),
            {
                "arch": architecture,
                "embedding_dim": 12,
                "hidden_dim": 8,
                "dropout": 0.0,
            },
        )
    if architecture == "framewise_embedding_mlp":
        return (
            FramewiseEmbeddingMLP(
                num_classes=3,
                embedding_dim=12,
                hidden_dim=8,
                dropout=0.0,
            ).eval(),
            (1, 1, 12, 5),
            {
                "arch": architecture,
                "embedding_dim": 12,
                "hidden_dim": 8,
                "dropout": 0.0,
            },
        )
    assert architecture == "beatnet_conv"
    return (
        BeatNetConvClassifier(
            num_classes=3,
            input_dim=12,
            dropout=0.0,
        ).eval(),
        (1, 5, 12),
        {"arch": architecture, "input_dim": 12, "dropout": 0.0},
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "mel_cnn",
        "mfcc_cnn",
        "mel_cnn_attention",
        "beatnet_log_spect_cnn",
        "embedding_stats_mlp",
        "framewise_embedding_mlp",
        "beatnet_conv",
    ],
)
def test_native_classifier_architecture_matches_pytorch_with_dynamic_batch_and_time(
    tmp_path: Path,
    architecture: str,
) -> None:
    torch.manual_seed(404)
    model, example_shape, model_config = _architecture_case(architecture)
    artifact = export_classifier_onnx(
        model,
        tmp_path / f"{architecture}.onnx",
        model_config=model_config,
        checkpoint_sha256=_state_digest(model),
        input_shape=example_shape,
        class_labels=LABELS,
    )
    native = OnnxClassifierSession(artifact)

    assert artifact.manifest["schema"] == NATIVE_CLASSIFIER_SCHEMA
    assert artifact.architecture == architecture
    assert artifact.labels == LABELS
    assert artifact.output_names == ("logits", "probabilities")
    assert artifact.manifest["wrapper"] == "bare_architecture"
    assert artifact.manifest["shape_contract"]["mode"] == (
        "dynamic_batch_and_time_fixed_features"
    )
    assert artifact.manifest["parity_validation"]["max_abs_error"] < 2e-6

    dynamic_shape = list(example_shape)
    dynamic_shape[0] = 2
    dynamic_shape[1 if architecture == "beatnet_conv" else 3] += 3
    features = torch.randn(*dynamic_shape)
    with torch.no_grad():
        expected_logits = model(features).numpy()
    actual = native.infer(features.numpy())
    np.testing.assert_allclose(
        actual["logits"],
        expected_logits,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual["probabilities"],
        torch.softmax(torch.from_numpy(expected_logits), dim=-1).numpy(),
        rtol=2e-5,
        atol=2e-6,
    )


def test_native_genre_classifier_preserves_labels_and_temperature(
    tmp_path: Path,
) -> None:
    torch.manual_seed(505)
    labels = ("candombe", "brid", "salsa", "other")
    model = GenreClassifier(
        arch="beatnet_conv",
        num_classes=4,
        genre_labels=list(labels),
        calibration_temperature=1.75,
        input_dim=12,
        dropout=0.0,
    ).eval()
    artifact = export_classifier_onnx(
        model,
        tmp_path / "genre-classifier.onnx",
        model_config={
            "arch": "beatnet_conv",
            "input_dim": 12,
            "dropout": 0.0,
            "calibration_temperature": 1.75,
        },
        checkpoint_sha256=_state_digest(model),
        input_shape=(1, 5, 12),
    )
    native = OnnxClassifierSession(artifact)
    features = torch.randn(3, 8, 12)
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits / 1.75, dim=-1)
    actual = native.infer(features.numpy())

    assert artifact.manifest["wrapper"] == "genre_classifier"
    assert artifact.labels == labels
    assert artifact.manifest["calibration_temperature"] == 1.75
    np.testing.assert_allclose(actual["logits"], logits.numpy(), atol=2e-6)
    np.testing.assert_allclose(
        actual["probabilities"],
        probabilities.numpy(),
        atol=2e-6,
    )


def test_native_classifier_cache_verifies_hash_and_label_contract(
    tmp_path: Path,
) -> None:
    torch.manual_seed(606)
    model = BeatNetConvClassifier(
        num_classes=2,
        input_dim=8,
        dropout=0.0,
    ).eval()
    kwargs = {
        "model_config": {"arch": "beatnet_conv", "input_dim": 8},
        "checkpoint_sha256": _state_digest(model),
        "input_shape": (1, 4, 8),
        "class_labels": ("target", "other"),
        "cache_root": tmp_path,
    }
    exported = ensure_classifier_onnx(model, **kwargs)
    reused = ensure_classifier_onnx(model, **kwargs)
    assert exported.cache_hit is False
    assert reused.cache_hit is True
    assert reused.model_path == exported.model_path
    assert reused.manifest["onnx"]["sha256"] == _file_digest(reused.model_path)

    reused.model_path.write_bytes(reused.model_path.read_bytes() + b"corrupt")
    repaired = ensure_classifier_onnx(model, **kwargs)
    assert repaired.cache_hit is False
    assert repaired.manifest["onnx"]["sha256"] == _file_digest(repaired.model_path)

    relabelled = ensure_classifier_onnx(
        model,
        **{**kwargs, "class_labels": ("other", "target")},
    )
    assert relabelled.model_path != repaired.model_path


@pytest.mark.parametrize(
    ("condition", "expected_architecture", "embedding_dim", "feature_type"),
    [
        ("efficientat", "embedding_stats_mlp", 600, "efficientat_embedding"),
        ("yamnet", "framewise_embedding_mlp", 1024, "yamnet_embedding"),
    ],
)
def test_promoted_embedding_classifier_head_exports_with_frontend_boundary(
    tmp_path: Path,
    condition: str,
    expected_architecture: str,
    embedding_dim: int,
    feature_type: str,
) -> None:
    bundle = load_trained_model_bundle(
        "classifier",
        "latin_router",
        condition,
    )
    checkpoint_path = bundle.checkpoint_path(0)
    checkpoint_record = bundle.checkpoints[0]
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = dict(checkpoint["model_config"])
    architecture = str(model_config.pop("arch"))
    labels = tuple(str(label) for label in checkpoint["labels"])
    temperature = float(checkpoint["calibration"]["temperature"])
    model = GenreClassifier(
        arch=architecture,
        num_classes=len(labels),
        genre_labels=list(labels),
        calibration_temperature=temperature,
        **model_config,
    ).eval()
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    segment_frames = int(checkpoint["runtime_window_contract"]["segment_frames"])
    export_config = dict(checkpoint["model_config"])
    export_config["calibration_temperature"] = temperature
    artifact = export_classifier_onnx(
        model,
        tmp_path / f"{condition}.onnx",
        model_config=export_config,
        checkpoint_sha256=checkpoint_record.sha256,
        input_shape=(1, 1, embedding_dim, segment_frames),
        feature_config=checkpoint["feature_config"],
    )
    native = OnnxClassifierSession(artifact)

    assert _file_digest(checkpoint_path) == checkpoint_record.sha256
    assert artifact.architecture == expected_architecture
    assert artifact.labels == labels
    assert artifact.manifest["checkpoint_sha256"] == checkpoint_record.sha256
    frontend = artifact.manifest["feature_frontend"]
    assert frontend["included"] is False
    assert frontend["input_representation"] == feature_type
    assert frontend["status"] == "downstream_head_only"
    assert "not packaged" in frontend["unsupported_full_frontend_reason"]
    assert len(frontend["feature_config_sha256"]) == 64

    features = torch.randn(2, 1, embedding_dim, segment_frames + 2)
    with torch.no_grad():
        expected_logits = model(features)
        expected_probabilities = torch.softmax(
            expected_logits / temperature,
            dim=-1,
        )
    actual = native.infer(features.numpy())
    np.testing.assert_allclose(
        actual["logits"],
        expected_logits.numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual["probabilities"],
        expected_probabilities.numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
