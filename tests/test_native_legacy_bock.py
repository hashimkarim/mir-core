"""Compare every historical member and its ordered ensemble with madmom itself."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from mir_core.checkpoints.bocktcn import (
    baseline_checkpoint_names,
    baseline_checkpoint_path,
)
from mir_core.models.bock_tcn.legacy import LegacyBockTCN
from mir_core.native import OnnxBatchModelSession, ensure_batch_model_onnx
from mir_core.native.batch_frontend import (
    OnnxBatchFrontendSession,
    ensure_batch_frontend_onnx,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _reference(selectors, features):
    from madmom.features.beats import _tcn_beat_processor_pad
    from madmom.ml.nn import average_predictions

    outputs = []
    for selector in selectors:
        # Only immutable, packaged resources already verified by the port loader.
        with baseline_checkpoint_path(selector).open("rb") as handle:
            network = pickle.load(handle, encoding="latin1")
        outputs.append(network(_tcn_beat_processor_pad(features)))
    return average_predictions(outputs)


def _session(model, directory):
    artifact = ensure_batch_model_onnx(
        model,
        model_name="bocktcn_legacy",
        model_config=model.port_config,
        checkpoint_sha256=model.checkpoint_sha256,
        input_shape=(1, 1, 37, 81),
        cache_root=directory,
    )
    assert artifact.output_names == ("beats", "tempo")
    return OnnxBatchModelSession(artifact)


@pytest.mark.parametrize("selector", baseline_checkpoint_names())
def test_every_original_member_matches_madmom_and_onnx(selector, tmp_path: Path):
    model = LegacyBockTCN([selector]).eval()
    session = _session(model, tmp_path)
    for frames in (1, 37, 257):
        features = (
            np.random.default_rng(frames)
            .normal(-3, 1, (2, 1, frames, 81))
            .astype(np.float32)
        )
        actual = session.infer(features)
        with torch.inference_mode():
            lowered = model(torch.from_numpy(features))
        for batch in range(2):
            reference = _reference([selector], features[batch, 0])
            for index, name in enumerate(("beats", "tempo")):
                expected = np.asarray(reference[index]).reshape(
                    actual[name][batch].shape
                )
                np.testing.assert_allclose(
                    actual[name][batch], expected, rtol=2e-5, atol=2e-6
                )
                np.testing.assert_allclose(
                    lowered[index][batch].numpy(), expected, rtol=2e-5, atol=2e-6
                )


def test_original_ensemble_from_waveform_matches_madmom(tmp_path: Path):
    from madmom.audio.signal import Signal
    from madmom.features.beats import TCNBeatProcessor

    model = LegacyBockTCN().eval()
    session = _session(model, tmp_path)
    reference = TCNBeatProcessor(
        nn_files=[
            str(baseline_checkpoint_path(name)) for name in baseline_checkpoint_names()
        ],
        tasks=(0, 1),
        fps=100,
        num_threads=1,
    )
    frontend = OnnxBatchFrontendSession(
        ensure_batch_frontend_onnx(model_name="bocktcn", cache_root=tmp_path)
    )
    assert len(model.port_config["members"]) == 8
    for samples in (1, 16385, 88201):
        wave = (
            np.random.default_rng(samples).normal(0, 0.08, samples).astype(np.float32)
        )
        wave[::22050] = 0.95
        if samples == 1:
            wave[:] = 0
        expected = reference(Signal(wave, sample_rate=44100))
        actual = session.infer(frontend.infer(wave[None]))
        for index, name in enumerate(("beats", "tempo")):
            np.testing.assert_allclose(
                actual[name][0],
                np.asarray(expected[index]).reshape(actual[name][0].shape),
                rtol=2e-5,
                atol=2e-6,
            )


def test_historical_identity_rejects_changed_weights_and_metadata(tmp_path: Path):
    model = LegacyBockTCN(["baseline"])
    with pytest.raises(ValueError, match="distinct"):
        LegacyBockTCN(["baseline", "baseline"])
    with pytest.raises(ValueError, match="metadata must identify"):
        ensure_batch_model_onnx(
            model,
            model_name="bocktcn_legacy",
            model_config={},
            checkpoint_sha256=model.checkpoint_sha256,
            input_shape=(1, 1, 37, 81),
            cache_root=tmp_path,
        )
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="weights were modified"):
        _session(model, tmp_path)
    model = LegacyBockTCN(["baseline"])
    model.port_config["members"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata was modified"):
        _session(model, tmp_path)
