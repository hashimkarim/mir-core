"""Replay successful Python native-reference tests through C++ and Android.

Use with pytest -p port_validation_plugin and MIR_PORT_FIXTURES=<output dir>.
MIR_CPP_REPLAY additionally checks the C++ executable. The original tests and
their PyTorch/madmom/librosa assertions remain active. Exported goldens are the
Python ONNX outputs, bound to those reference tests and exporter parity gates;
they do not certify training accuracy or original-paper authenticity.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

RECORDS: dict[str, dict] = {}
ROOT: Path
CURRENT_TEST = ""
TEST_RESULTS: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tensor(stream, values):
    values = np.ascontiguousarray(values, dtype="<f4")
    stream.write(struct.pack("<I", values.ndim))
    stream.write(struct.pack("<" + "Q" * values.ndim, *values.shape))
    stream.write(values.tobytes())


def write_outputs(stream, outputs):
    stream.write(struct.pack("<I", len(outputs)))
    for name, values in sorted(outputs.items()):
        encoded = name.encode()
        stream.write(struct.pack("<I", len(encoded)))
        stream.write(encoded)
        write_tensor(stream, values)


def read_u32(stream):
    return struct.unpack("<I", stream.read(4))[0]


def read_outputs(stream):
    outputs = {}
    for _ in range(read_u32(stream)):
        name = stream.read(read_u32(stream)).decode()
        rank = read_u32(stream)
        shape = struct.unpack("<" + "Q" * rank, stream.read(8 * rank))
        count = int(np.prod(shape))
        outputs[name] = np.frombuffer(stream.read(4 * count), dtype="<f4").reshape(shape)
    return outputs


def record(session, values, outputs, reset):
    artifact = session.artifact
    manifest = artifact.manifest
    graph_record = manifest.get("onnx", manifest.get("tflite"))
    key = manifest["artifact_key"] + "-" + graph_record["sha256"][:12]
    # Interleaved live sessions own separate state and therefore separate
    # replays, even when their graph and weights have the same identity.
    if manifest["schema"] == "mir.native-streaming-model/v1":
        if not hasattr(session, "_port_fixture_key"):
            session._port_fixture_key = key + f"-session-{len(RECORDS)}"
        key = session._port_fixture_key
    folder = ROOT / key
    if key not in RECORDS:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.model_path, folder / artifact.model_path.name)
        shutil.copy2(artifact.manifest_path, folder / artifact.manifest_path.name)
        for filename in ("inputs.replay", "expected.replay"):
            (folder / filename).write_bytes(struct.pack("<I", 0))
        parity = manifest["parity_validation"]
        RECORDS[key] = {
            "id": key,
            "schema": manifest["schema"],
            "family": manifest.get("architecture", manifest.get("model_family", manifest.get("frontend"))),
            "manifest": artifact.manifest_path.name,
            "model": artifact.model_path.name,
            "calls": 0, "tests": [],
            "rtol": parity["rtol"], "atol": parity["atol"],
        }
    entry = RECORDS[key]
    if CURRENT_TEST not in entry["tests"]:
        entry["tests"].append(CURRENT_TEST)
    with (folder / "inputs.replay").open("ab") as stream:
        stream.write(struct.pack("<I", int(reset)))
        write_tensor(stream, values)
    with (folder / "expected.replay").open("ab") as stream:
        write_outputs(stream, outputs)
    entry["calls"] += 1


def wrap_features(cls, name=None):
    original = cls.infer

    @functools.wraps(original)
    def infer(self, values):
        expected = original(self, values)
        tensor = np.ascontiguousarray(values, dtype=np.float32)
        if name in {"features", "embeddings", "embedding"} and tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        record(self, tensor, {name: expected} if name else expected, True)
        return expected

    cls.infer = infer


def pytest_configure(config):
    global ROOT
    ROOT = Path(os.environ["MIR_PORT_FIXTURES"]).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    from mir_core.native.batch import OnnxBatchModelSession
    from mir_core.native.classifier import OnnxClassifierSession
    from mir_core.native.batch_frontend import OnnxBatchFrontendSession
    from mir_core.native.beatnet import OnnxBeatNetStreamingSession
    from classifierlab.native_frontend import OnnxFeatureFrontendSession
    from classifierlab.native_yamnet_frontend import TFLiteYAMNetFrontendSession

    wrap_features(OnnxBatchModelSession)
    wrap_features(OnnxClassifierSession)
    wrap_features(OnnxBatchFrontendSession, "features")
    wrap_features(OnnxFeatureFrontendSession, "embeddings")
    wrap_features(TFLiteYAMNetFrontendSession, "embedding")
    original = OnnxBeatNetStreamingSession.infer_outputs
    reset = OnnxBeatNetStreamingSession.reset_hidden

    @functools.wraps(original)
    def infer_outputs(self, values):
        expected = original(self, values)
        outputs = {name: np.asarray(value).reshape(1, 1, -1) for name, value in expected.items()}
        outputs.update(next_hidden=self.hidden, next_cell=self.cell)
        record(self, np.asarray(values).reshape(1, 1, -1), outputs, getattr(self, "_port_reset", True))
        self._port_reset = False
        return expected

    @functools.wraps(reset)
    def reset_hidden(self, *args, **kwargs):
        result = reset(self, *args, **kwargs)
        self._port_reset = True
        return result

    OnnxBeatNetStreamingSession.infer_outputs = infer_outputs
    OnnxBeatNetStreamingSession.reset_hidden = reset_hidden


def pytest_runtest_setup(item):
    global CURRENT_TEST
    CURRENT_TEST = item.nodeid


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.when == "call":
        TEST_RESULTS[item.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    failures = []
    for entry in RECORDS.values():
        folder = ROOT / entry["id"]
        for filename in ("inputs.replay", "expected.replay"):
            with (folder / filename).open("r+b") as stream:
                stream.write(struct.pack("<I", entry["calls"]))
        entry["reference_tests_passed"] = all(TEST_RESULTS.get(test) == "passed" for test in entry["tests"])
        entry["sha256"] = {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != "cpp.replay"}
        if executable := os.environ.get("MIR_CPP_REPLAY"):
            try:
                subprocess.run([executable, str(folder / entry["manifest"]), str(folder / "inputs.replay"), str(folder / "cpp.replay")], check=True, capture_output=True, text=True, timeout=300)
                with (folder / "expected.replay").open("rb") as expected_file, (folder / "cpp.replay").open("rb") as actual_file:
                    assert read_u32(expected_file) == read_u32(actual_file) == entry["calls"]
                    maximum = {}
                    for _ in range(entry["calls"]):
                        expected, actual = read_outputs(expected_file), read_outputs(actual_file)
                        assert expected.keys() == actual.keys()
                        for name in expected:
                            assert actual[name].shape == expected[name].shape
                            error = float(np.max(np.abs(actual[name] - expected[name])))
                            maximum[name] = max(error, maximum.get(name, 0.0))
                            np.testing.assert_allclose(actual[name], expected[name], rtol=entry["rtol"], atol=entry["atol"], err_msg=entry["id"] + ":" + name)
                    assert not expected_file.read(1) and not actual_file.read(1)
                entry["cpp"] = {"status": "pass", "max_abs_by_output": maximum}
            except Exception as error:
                message = getattr(error, "stderr", None) or str(error)
                entry["cpp"] = {"status": "fail", "error": message}
                failures.append(entry["id"] + ": " + message)
    report = {"schema": "mir.port-replay-suite/v1", "reference_test_exit_status": int(exitstatus), "reference_tests": TEST_RESULTS, "artifacts": list(RECORDS.values())}
    (ROOT / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        session.exitstatus = 1
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal:
            terminal.write_sep("=", "C++ port failures")
            for failure in failures:
                terminal.write_line(failure)
