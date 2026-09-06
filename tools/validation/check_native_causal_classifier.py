#!/usr/bin/env python3
"""Compare the standalone causal adapter with unmodified ClassifierLab sources.

The deterministic classifier and LOG_SPECT policy probes test implementation
parity, not trained accuracy or policy selection. The policy state-machine
parameters come from the independently frozen promoted-policy fixture; its
window timing is explicitly adapted to the LOG_SPECT test frontend.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from classifierlab.causal_streaming import BeatNetCausalFrontend
from classifierlab.live_runtime import LiveCausalFeatureFrontend, LiveClassifierRouter
from classifierlab.window_routing import runtime_policy_from_dict
from mir_core.models import BeatNetLogSpectCNN
from mir_core.native import export_classifier_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root, output = args.workspace, args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    binary = root / "mir-desktop-app/native-runtime/target/debug/examples/causal_classifier_replay"
    fixture_path = root / "mir-desktop-app/native-runtime/tests/fixtures/python_router_policies.json"
    policy_case = json.loads(fixture_path.read_text())["policies"][0]
    policy_value = json.loads(policy_case["policy_json"])
    policy = runtime_policy_from_dict(policy_value)
    policy = dataclasses.replace(policy, window_contract={
        **policy.window_contract, "feature_fps": 50.0, "segment_frames": 50,
        "hop_frames": 25, "context_seconds": 1.0, "hop_seconds": 0.5,
        "feature_receptive_field_seconds": 1408 / 22050,
        "frame_availability_offset_seconds": 704 / 22050,
    })
    policy_value = policy.as_dict()
    policy_path = output / "policy.json"
    policy_path.write_text(json.dumps(policy_value))
    policy_file_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    segment = int(policy.window_contract["segment_frames"])
    resampler = root / "mir-core/native-dsp/build/libmir_dsp.so"
    reports = []
    for mode, rate in [("none", 22050), ("running_peak", 22050),
                       ("fixed_gain", 22050), ("running_peak", 44100)]:
        name = f"logspect-{mode}-{rate}"
        cfg = {"type": "beatnet_log_spect", "causal_normalization": mode,
               "causal_fixed_gain": 1.7}
        torch.manual_seed(991)
        model = BeatNetLogSpectCNN(num_classes=len(policy_case["labels"])).eval()
        model_sha = hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        artifact = export_classifier_onnx(
            model, output / name / "classifier.onnx",
            model_config={"arch": "beatnet_log_spect_cnn", "feature_dim": 272,
                          "num_classes": len(policy_case["labels"])},
            checkpoint_sha256=model_sha, input_shape=(1, 1, 272, segment),
            class_labels=policy_case["labels"], feature_config=cfg)
        reference_frontend = LiveCausalFeatureFrontend(BeatNetCausalFrontend(cfg), source_sample_rate=rate)
        reference_classifier = LiveClassifierRouter(
            model=model, labels=policy_case["labels"], policy=policy,
            frontend=LiveCausalFeatureFrontend(BeatNetCausalFrontend(cfg), source_sample_rate=rate),
            device=torch.device("cpu"))
        count = int((segment + 4 * policy.window_contract["hop_frames"] + 30) / 50 * rate)
        audio = np.random.default_rng(811).normal(0, 0.02, count).astype(np.float32)
        audio[: int(rate * 0.12)] = 0
        audio[count // 2] = -0.9  # Later peak must not normalize earlier features.
        chunks, cursor = [[], []], 0
        sizes = [1, 703, 13, 441, 1701, 88, 997, 4096]
        while cursor < len(audio):
            size = sizes[len(chunks) % len(sizes)]
            chunks.append(audio[cursor:cursor+size].tolist())
            cursor += size
        chunks.append([])
        frontend_config = {"kind": "beatnet-log-spect", "feature_config": cfg}
        classifier_config = {
            "artifact": str(artifact.manifest_path), "frontend": frontend_config,
            "policy": str(policy_path), "policy_sha256": policy_file_hash,
            "resampler_library": str(resampler) if rate != 22050 else None,
        }
        replay = {"source_rate": rate, "frontend": frontend_config,
                  "resampler_library": classifier_config["resampler_library"],
                  "classifier": classifier_config, "chunks": chunks}
        path = output / f"{name}.input.json"
        path.write_text(json.dumps(replay))
        frames, decisions = [], []
        for chunk in chunks:
            values = np.asarray(chunk, dtype=np.float32)
            frames.extend(reference_frontend.process_audio(values))
            decisions.extend(reference_classifier.process_audio(values))
        result = subprocess.run([str(binary), str(path)], check=True, text=True, capture_output=True)
        actual = json.loads(result.stdout)
        (output / f"{name}.actual.json").write_text(result.stdout)
        observed = actual["result"]["frames"]
        assert actual["reset_exact"] and len(observed) == len(frames)
        expected_features = np.stack([f.value for f in frames])
        observed_features = np.asarray([f["values"] for f in observed], dtype=np.float32)
        np.testing.assert_allclose(observed_features, expected_features, rtol=2e-5, atol=1e-5)
        for got, want in zip(observed, frames, strict=True):
            assert got["frame_index"] == want.frame_index
            for key in ["source_start_seconds", "source_end_seconds", "availability_seconds"]:
                assert got[key] == getattr(want, key), (name, key, got[key], getattr(want, key))
        observed_decisions = actual["result"]["decisions"]
        assert len(observed_decisions) == len(decisions) >= 4
        for got, want in zip(observed_decisions, decisions, strict=True):
            for key in ["feature_start_frame", "feature_end_frame", "source_start_seconds",
                        "source_end_seconds", "availability_seconds"]:
                assert got[key] == getattr(want, key)
            expected_route = dataclasses.asdict(want.routing)
            for key, value in got["routing"].items():
                if key in {"confidence", "probabilities", "ema_probabilities"}:
                    np.testing.assert_allclose(value, expected_route[key], rtol=2e-5, atol=2e-6)
                else:
                    assert value == expected_route[key], (name, key, value, expected_route[key])
        # Independently bound identity must reject a changed waveform normalization.
        replay["classifier"]["frontend"]["feature_config"] = dict(cfg, causal_fixed_gain=2.1)
        bad = output / f"{name}.tampered.json"
        bad.write_text(json.dumps(replay))
        invalid = subprocess.run([str(binary), str(bad)], text=True, capture_output=True)
        assert invalid.returncode != 0 and "configuration hashes differ" in invalid.stderr
        reports.append({"case": name, "frames": len(frames), "decisions": len(decisions),
                        "max_feature_error": float(np.max(np.abs(observed_features-expected_features))),
                        "status": "pass", "reset_exact": True, "tampered_config_rejected": True})
        print(json.dumps(reports[-1]), flush=True)
    sources = [root / "mir-train-hpc/classifierlab" / f for f in
               ["live_runtime.py", "causal_streaming.py", "window_routing.py", "yamnet_contract.py"]]
    report = {"schema": "mir.causal-classifier-source-parity/v1", "cases": reports,
              "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "policy_fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              "policy_kind": "synthetic LOG_SPECT timing probe using frozen router parameters"}
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
