"""Replay all promoted classifier heads across caller input-buffer alignments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runner = ROOT / "mir-desktop-app/native-runtime/target/debug/examples/feature_alignment_replay"
    records = []
    for family in ("efficientat", "yamnet"):
        for fold in range(5):
            name = f"{family}-fold-{fold}"
            directory = args.promoted_routing / name
            classifier = json.loads((directory / "input.json").read_text())["classifier"]
            result = json.loads((directory / "actual.json").read_text())["result"]
            decision = result["decisions"][0]
            frames = result["frames"][decision["feature_start_frame"]:decision["feature_end_frame"]]
            dimension = len(frames[0]["values"])
            manifest = json.loads(Path(classifier["artifact"]).read_text())
            if manifest["architecture"] == "beatnet_conv":
                shape = [1, len(frames), dimension]
                values = [value for frame in frames for value in frame["values"]]
            else:
                shape = [1, 1, dimension, len(frames)]
                values = [frame["values"][f] for f in range(dimension) for frame in frames]
            fixture = args.output / (name + "-input.json")
            fixture.write_text(json.dumps({"artifact": classifier["artifact"], "shape": shape, "values": values}))
            process = subprocess.run([str(runner), str(fixture)], text=True, capture_output=True, timeout=120)
            (args.output / (name + "-result.json")).write_text(process.stdout)
            assert process.returncode == 0, process.stderr
            actual = json.loads(process.stdout)
            assert actual["passed"] and len(actual["cases"]) == 32
            assert sorted(c["address_mod_128"] for c in actual["cases"]) == list(range(0, 128, 4))
            records.append({"name": name, "manifest_sha256": hashlib.sha256(Path(classifier["artifact"]).read_bytes()).hexdigest(), "input_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "caller_alignments": 32, "outputs_exact": True})
    report = {"status": "pass", "cases": records, "comparisons": 32 * len(records),
              "scope": "same graph and exact input across all float offsets in 128 bytes; independent source accuracy remains separately tested"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
