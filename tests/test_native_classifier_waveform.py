"""Raw-waveform classifier ports must preserve the original feature contract.

Keep this module's basename distinct from mir-train-hpc's frontend tests:
pytest importlib mode resolves both repositories' tests package names together.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from mir_core.models import GenreClassifier, MelCNN, MelCNNAttention, MFCCCNN
from mir_core.native import (
    OnnxBatchFrontendSession,
    OnnxClassifierSession,
    ensure_classifier_frontend_onnx,
    export_classifier_onnx,
)


@pytest.fixture(scope="module", params=["mel", "mfcc"])
def frontend(request, tmp_path_factory):
    return ensure_classifier_frontend_onnx(
        feature_type=request.param,
        cache_root=tmp_path_factory.mktemp("classifier-frontends"),
    )


def original(family, audio, target_samples, *, bands=128, coefficients=20):
    function = (
        GenreClassifier.preprocess_audio
        if family == "mel"
        else GenreClassifier.preprocess_mfcc
    )
    options = {"n_mels": bands} if family == "mel" else {"n_mfcc": coefficients}
    return function(audio, duration=target_samples / 22_050, **options).numpy()[0]


def compare(artifact, actual, expected):
    gate = artifact.manifest["parity_validation"]
    np.testing.assert_allclose(actual, expected, atol=gate["atol"], rtol=gate["rtol"])


def test_dynamic_classifier_frontend_matches_original_and_derivative_edges(frontend):
    session = OnnxBatchFrontendSession(frontend)
    rng = np.random.default_rng(2309)
    for length in [4096, 4097, 8193, 16384]:
        waves = rng.normal(0.0, 0.13, (2, length)).astype(np.float32)
        waves[0, [0, length // 2, -1]] = [0.8, -0.9, 1.0]
        expected = np.stack(
            [original(frontend.model_family, row, length) for row in waves]
        )
        actual = session.infer(waves)
        compare(frontend, actual, expected)
        compare(frontend, session.infer(waves), actual)
    zeros = np.zeros((2, 8192), dtype=np.float32)
    compare(
        frontend,
        session.infer(zeros),
        np.stack([original(frontend.model_family, row, 8192) for row in zeros]),
    )


@pytest.mark.parametrize("family", ["mel", "mfcc"])
@pytest.mark.parametrize("normalize", [False, True])
def test_fixed_window_padding_truncation_and_optional_normalization(
    tmp_path, family, normalize
):
    target = 8192
    artifact = ensure_classifier_frontend_onnx(
        feature_type=family,
        feature_config={"window_samples": target, "normalize_peak": normalize},
        cache_root=tmp_path,
    )
    session = OnnxBatchFrontendSession(artifact)
    rng = np.random.default_rng(983)
    for length in [1, 511, target, target + 513]:
        waves = rng.normal(0, 0.07, (2, length)).astype(np.float32)
        waves[0] = 0
        expected = []
        for wave in waves:
            prepared = np.pad(wave[:target], (0, max(0, target - length)))
            if normalize and np.max(np.abs(prepared)) > 0:
                prepared = prepared / np.max(np.abs(prepared))
            expected.append(original(family, prepared, target))
        compare(artifact, session.infer(waves), np.stack(expected))


def test_frontend_identity_binds_configuration_and_rejects_invalid_input(
    frontend, tmp_path
):
    reused = ensure_classifier_frontend_onnx(
        feature_type=frontend.model_family, cache_root=frontend.model_path.parents[2]
    )
    assert reused.cache_hit and reused.artifact_key == frontend.artifact_key
    changed = ensure_classifier_frontend_onnx(
        feature_type=frontend.model_family,
        feature_config={"normalize_peak": True},
        cache_root=tmp_path,
    )
    assert changed.artifact_key != frontend.artifact_key
    session = OnnxBatchFrontendSession(frontend)
    for values in [
        np.zeros((0, 8192), dtype=np.float32),
        np.full(8192, np.nan),
        np.full(8192, np.inf),
    ]:
        with pytest.raises(ValueError):
            session.infer(values)
    if frontend.model_family == "mfcc":
        with pytest.raises(ValueError, match="at least 4096"):
            session.infer(np.zeros(4095, dtype=np.float32))


@pytest.mark.parametrize("architecture", ["mel_cnn", "mel_cnn_attention", "mfcc_cnn"])
def test_waveform_to_classifier_logits_matches_original_pipeline(
    tmp_path, architecture
):
    torch.manual_seed(937)
    family = "mfcc" if architecture == "mfcc_cnn" else "mel"
    model = {
        "mel_cnn": MelCNN,
        "mel_cnn_attention": MelCNNAttention,
        "mfcc_cnn": MFCCCNN,
    }[architecture](num_classes=3, dropout=0).eval()
    frontend = ensure_classifier_frontend_onnx(
        feature_type=family, cache_root=tmp_path / "features"
    )
    waves = np.random.default_rng(85).normal(0, 0.09, (2, 16384)).astype(np.float32)
    expected_features = np.stack([original(family, row, len(row)) for row in waves])
    actual_features = OnnxBatchFrontendSession(frontend).infer(waves)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    classifier = export_classifier_onnx(
        model,
        tmp_path / "classifier.onnx",
        model_config={"arch": architecture, "dropout": 0},
        checkpoint_sha256=digest.hexdigest(),
        input_shape=(1, *expected_features.shape[1:]),
        class_labels=("candombe", "brid", "salsa"),
        feature_config={"feature_type": family},
    )
    actual = OnnxClassifierSession(classifier).infer(actual_features)
    with torch.no_grad():
        logits = model(torch.from_numpy(expected_features))
    np.testing.assert_allclose(actual["logits"], logits.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        actual["probabilities"], torch.softmax(logits, -1).numpy(), rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "options",
    [
        {"window_samples": 100},
        {"window_samples": True},
        {"normalize_peak": 1},
        {"n_mfcc": 129},
        {"n_fft": 1023},
        {"fmin": 1000, "fmax": 500},
        {"center": False},
        {"hop_length": 512.5},
        {"n_fft": True},
    ],
)
def test_unsupported_or_inconsistent_mfcc_config_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        ensure_classifier_frontend_onnx(
            feature_type="mfcc", feature_config=options, cache_root=tmp_path
        )
