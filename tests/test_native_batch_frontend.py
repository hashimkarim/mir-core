from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from mir_core.native import (
    NATIVE_BATCH_FRONTEND_SCHEMA,
    OnnxBatchFrontendSession,
    UnsupportedNativeBatchFrontendError,
    benchmark_batch_frontend,
    ensure_batch_frontend_onnx,
    load_batch_frontend_artifact,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bock_artifact(tmp_path_factory: pytest.TempPathFactory):
    return ensure_batch_frontend_onnx(
        model_name="bock_tcn",
        preprocessing_config={"fps": 100, "frame_size": 2048, "num_bands": 12},
        cache_root=tmp_path_factory.mktemp("native-bock-frontend"),
    )


@pytest.fixture(scope="module")
def beast_artifact(tmp_path_factory: pytest.TempPathFactory):
    return ensure_batch_frontend_onnx(
        model_name="beast",
        preprocessing_config={
            "sample_rate": 44_100,
            "n_fft": 4_096,
            "hop_length": 1_024,
            "n_mels": 128,
            "fmin": 30.0,
            "fmax": 11_000.0,
        },
        cache_root=tmp_path_factory.mktemp("native-beast-frontend"),
    )


@pytest.fixture(scope="module")
def spectnt_artifact(tmp_path_factory: pytest.TempPathFactory):
    return ensure_batch_frontend_onnx(
        model_name="spec_tnt",
        preprocessing_config={},
        cache_root=tmp_path_factory.mktemp("native-spectnt-frontend"),
    )


def test_bock_frontend_matches_real_madmom_preprocessor_and_frame_contract(
    bock_artifact,
) -> None:
    from mir_core.preprocessing.madmom_features import PreProcessor

    artifact = bock_artifact
    native = OnnxBatchFrontendSession(artifact)
    processor = PreProcessor(frame_size=2048, num_bands=12, fps=100)
    generator = np.random.default_rng(101)
    waveforms = generator.normal(0.0, 0.05, (2, 441 * 7)).astype(np.float32)
    expected = np.stack([processor(value) for value in waveforms])[:, None]
    actual = native.infer(waveforms)

    assert artifact.manifest["schema"] == NATIVE_BATCH_FRONTEND_SCHEMA
    assert artifact.model_family == "bocktcn"
    assert artifact.manifest["inputs"] == ["waveform"]
    assert artifact.manifest["outputs"] == ["features"]
    assert actual.shape == (2, 1, 7, 81)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)

    one_more_sample = np.pad(waveforms[:1], ((0, 0), (0, 1)))
    assert native.infer(one_more_sample).shape == (1, 1, 8, 81)


def test_beast_frontend_matches_librosa_and_records_noncausal_boundary(
    beast_artifact,
) -> None:
    from mir_core.preprocessing import BeastPreProcessor

    artifact = beast_artifact
    native = OnnxBatchFrontendSession(artifact)
    processor = BeastPreProcessor()
    generator = np.random.default_rng(202)
    waveforms = generator.normal(0.0, 0.04, (2, 8_209)).astype(np.float32)
    expected = np.stack([processor(value, 44_100) for value in waveforms])
    actual = native.infer(waveforms)

    assert actual.shape == (2, 9, 128)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)
    output_contract = artifact.manifest["output_contract"]
    assert output_contract["streaming_safe"] is False
    assert "complete waveform" in output_contract["streaming_blocker"]
    assert output_contract["normalization_scope"] == ("entire_waveform_per_batch_item")


def test_spectnt_frontend_matches_torchaudio_harmonic_stft(
    spectnt_artifact,
) -> None:
    import torch

    from mir_core.preprocessing import SpecTNTPreProcessor

    artifact = spectnt_artifact
    native = OnnxBatchFrontendSession(artifact)
    processor = SpecTNTPreProcessor(device="cpu")
    generator = np.random.default_rng(303)
    waveforms = generator.normal(0.0, 0.06, (2, 2_065)).astype(np.float32)
    with torch.no_grad():
        expected = processor.process_tensor(torch.from_numpy(waveforms)).numpy()
    actual = native.infer(waveforms)

    assert actual.shape == (2, 6, 128, 9)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-4)
    assert artifact.manifest["preprocessing_config"]["bw_q"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 257 samples"):
        native.infer(np.zeros(256, dtype=np.float32))


def test_frontend_cache_and_distributed_loader_verify_content_hash(
    tmp_path: Path,
    bock_artifact,
) -> None:
    artifact = bock_artifact
    reused = ensure_batch_frontend_onnx(
        model_name="tcn",
        preprocessing_config={"fps": 100, "frame_size": 2048, "num_bands": 12},
        cache_root=artifact.model_path.parents[2],
    )
    assert reused.cache_hit is True
    assert reused.artifact_key == artifact.artifact_key
    assert artifact.manifest["onnx"]["sha256"] == _file_digest(artifact.model_path)

    distributed = tmp_path / "frontend.onnx"
    distributed_manifest = tmp_path / "frontend.onnx.json"
    shutil.copyfile(artifact.model_path, distributed)
    shutil.copyfile(artifact.manifest_path, distributed_manifest)
    loaded = load_batch_frontend_artifact(distributed_manifest)
    assert loaded.artifact_key == artifact.artifact_key

    distributed.write_bytes(distributed.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="invalid native batch frontend artifact"):
        load_batch_frontend_artifact(distributed)


def test_spectnt_learned_bandwidth_checkpoint_is_content_addressed(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "renamed.pt"
    torch.save({"hstft.bw_Q": torch.tensor([0.73])}, first_checkpoint)
    shutil.copyfile(first_checkpoint, second_checkpoint)

    first = ensure_batch_frontend_onnx(
        model_name="spectnt",
        preprocessing_config={"checkpoint": str(first_checkpoint)},
        cache_root=tmp_path / "cache",
    )
    second = ensure_batch_frontend_onnx(
        model_name="spec_tnt",
        preprocessing_config={"checkpoint": str(second_checkpoint)},
        cache_root=tmp_path / "cache",
    )

    assert first.artifact_key == second.artifact_key
    assert second.cache_hit is True
    assert first.manifest["preprocessing_config"]["bw_q"] == pytest.approx(0.73)
    checkpoint_record = first.manifest["source"]["preprocessing_checkpoint"]
    assert checkpoint_record["sha256"] == _file_digest(first_checkpoint)
    assert "path" not in checkpoint_record
    assert "path_name" not in checkpoint_record


def test_unsupported_frontend_semantics_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        UnsupportedNativeBatchFrontendError,
        match="fractional-hop rounding",
    ):
        ensure_batch_frontend_onnx(
            model_name="bocktcn",
            preprocessing_config={"fps": 99},
            cache_root=tmp_path,
        )
    with pytest.raises(
        UnsupportedNativeBatchFrontendError,
        match="no audited semantics",
    ):
        ensure_batch_frontend_onnx(
            model_name="beast",
            preprocessing_config={"resample_type": "soxr_hq"},
            cache_root=tmp_path,
        )
    with pytest.raises(ValueError, match="either bw_q or checkpoint"):
        ensure_batch_frontend_onnx(
            model_name="spectnt",
            preprocessing_config={"bw_q": 1.0, "checkpoint": "unused.pt"},
            cache_root=tmp_path,
        )


def test_frontend_benchmark_reports_graph_only_scope(bock_artifact) -> None:
    report = benchmark_batch_frontend(
        OnnxBatchFrontendSession(bock_artifact),
        warmup=0,
        iterations=2,
    )
    assert report["schema"] == "mir.native-batch-frontend-benchmark/v1"
    assert report["model_family"] == "bocktcn"
    assert report["mean_ms"] > 0.0
    assert report["compute_realtime_factor"] > 0.0
    assert report["timing_scope"].endswith("decode_downmix_and_resample")


def test_manifest_is_stable_json_and_records_multi_probe_parity(
    bock_artifact,
    beast_artifact,
    spectnt_artifact,
) -> None:
    for artifact in (bock_artifact, beast_artifact, spectnt_artifact):
        payload = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        assert payload == artifact.manifest
        parity = payload["parity_validation"]
        assert len(parity["probes"]) == 3
        assert parity["max_abs_error"] > 0.0
        assert payload["identity"]["model_family"] == payload["model_family"]
        assert len(payload["preprocessing_config_sha256"]) == 64
        assert len(payload["frontend_constants_sha256"]) == 64
