"""Full-geometry and sustained-state probes for the explicit cross-port gate.

These are implementation tests. Seeded parameters are never described as a
trained checkpoint or as evidence of musical accuracy.
"""
from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch

from mir_core.models import SpecTNT
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.models.beatnet.beatnet_plus import BeatNetPlusOnline
from mir_core.native import export_streaming_beatnet_onnx, OnnxBeatNetStreamingSession
from mir_core.native.batch import export_batch_model_onnx, OnnxBatchModelSession


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_full_spectnt_reproduction_geometry_all_outputs(tmp_path: Path):
    torch.manual_seed(20260905)
    model = SpecTNT().eval()
    assert sum(p.numel() for p in model.parameters()) == 5_664_042
    artifact = export_batch_model_onnx(
        model, tmp_path / "spectnt.onnx", model_name="spec_tnt", model_config={},
        checkpoint_sha256=state_digest(model), input_shape=(1, 6, 128, 312),
    )
    native = OnnxBatchModelSession(artifact)
    generator = np.random.default_rng(20260905)
    for values in (np.zeros((1, 6, 128, 312), np.float32), generator.normal(0, 1, (2, 6, 128, 312)).astype(np.float32)):
        with torch.inference_mode():
            expected = model(torch.from_numpy(values))
        actual = native.infer(values)
        for name, output in actual.items():
            np.testing.assert_allclose(output, expected[name].numpy(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("family,cls,features,layers", [
    ("beatnet", BeatNetCRNN, 272, 2),
    ("beatnet_plus", BeatNetPlusOnline, 288, 4),
])
def test_1024_frame_stream_preserves_states_and_reset(tmp_path, family, cls, features, layers):
    torch.manual_seed(713)
    model = cls().eval()
    model.reset_hidden()
    artifact = export_streaming_beatnet_onnx(
        model, tmp_path / f"{family}.onnx", model_name=family,
        model_config={"input_dim": features, "hidden_dim": 150, "num_layers": layers},
        checkpoint_sha256=state_digest(model),
    )
    native = OnnxBeatNetStreamingSession(artifact)
    generator = np.random.default_rng(713)
    frames = generator.normal(0, 0.8, (1024, features)).astype(np.float32)
    frames[256:384] = 0
    frames[512:640] = frames[0:128]
    first = None
    for index, frame in enumerate(frames):
        if index == 768:
            model.reset_hidden()
            native.reset_hidden()
        with torch.inference_mode():
            logits = model(torch.from_numpy(frame).reshape(1, 1, -1))
            # Both canonical online classes return [batch, classes, time].
            probabilities = torch.softmax(logits, dim=1).reshape(3).numpy()
        expected = np.array([probabilities[0] + probabilities[1], probabilities[1]])
        actual = native.infer(frame)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(native.hidden, model.hidden.numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(native.cell, model.cell.numpy(), rtol=2e-5, atol=2e-6)
        if first is None:
            first = actual.copy()
    native.reset_hidden()
    np.testing.assert_array_equal(native.infer(frames[0]), first)
