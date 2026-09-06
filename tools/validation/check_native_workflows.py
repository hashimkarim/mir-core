"""Compare native experiment plans, matching, captures and reports to source.

No model accuracy is inferred from synthetic participant events. Expectations
come only from the original Python workflows, Python Random, and mir_eval.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import copy
import csv
import fcntl

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "mir-desktop-app/src"), str(ROOT / "mir-train-hpc")]
from mir_desktop_app.experiment import (
    HeldoutTrack, TrialCapture, build_experiment_plan,
)
from mir_desktop_app.experiment_analysis import (
    event_accuracy_metrics, realtime_event_accuracy_metrics, match_events,
    analyze_trial, aggregate_trial_analyses,
)
from mir_desktop_app.experiment_workspace import ExperimentWorkspace
from mir_desktop_app.pipeline import FoldContext


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), (path, actual, expected)
        for k in expected:
            compare(actual[k], expected[k], f"{path}/{k}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (path, len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}/{i}")
    elif isinstance(expected, float):
        # Only f64 reductions/percentiles: no tolerance for identities or counts.
        assert np.isclose(actual, expected, rtol=2e-14, atol=2e-15), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bin", type=Path, default=ROOT / "mir-desktop-app/native-workflows/target/debug/mir-native-workflows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def native(request, reject=False):
        result = subprocess.run([str(args.native_bin)], input=json.dumps(request, allow_nan=False), text=True, capture_output=True)
        if reject:
            assert result.returncode != 0, ("accepted invalid request", request, result.stdout)
            return result.stderr
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    requests, expected = [], []
    def case(request, expectation):
        requests.append(request); expected.append(expectation)

    seeds = [0, 1, -1, 42, -42, 2**32-1, 2**32, 2**63-1, -(2**63)]
    for seed in seeds:
        rng = random.Random(seed); order = list(range(100)); rng.shuffle(order)
        case({"operation": "rng-probe", "seed": seed},
             {"order": order, "random": [rng.random() for _ in range(1000)]})

    rng = np.random.default_rng(791124)
    event_cases = [([], [], []), ([0.0], [], []), ([], [0.5], [1.0]),
                   ([0.0, .5, 1., 1.5], [0.0, .5, 1.], [.5, 1., 1.5]),
                   ([.4, .6], [.5], [.55]), ([.0, .1], [.05, .15], [.1, .2])]
    for _ in range(250):
        ref = np.sort(rng.uniform(0, 2, rng.integers(0, 35))).tolist()
        est = np.sort(rng.uniform(0, 2, rng.integers(0, 35))).tolist()
        ready = (np.asarray(est) + rng.uniform(-.1, .8, len(est))).tolist()
        event_cases.append((ref, est, ready))
    for reference, estimate, ready in event_cases:
        case({"operation": "event-accuracy", "reference": reference, "estimate": estimate},
             event_accuracy_metrics(reference, estimate))
        case({"operation": "realtime-accuracy", "reference": reference, "predicted": estimate, "ready": ready},
             realtime_event_accuracy_metrics(reference, estimate, ready))
        case({"operation": "match", "reference": reference, "estimate": estimate},
             [[m.reference_index, m.estimate_index] for m in match_events(reference, estimate)])

    root = args.output.resolve() / "data"; root.mkdir()
    tracks = []
    for index in range(12):
        dataset = ("brid", "candombe", "salsaset_ft")[index % 3]
        track_id = f"{index:03d}"
        audio = root / f"{dataset}-{track_id}.wav"; audio.write_bytes(b"synthetic plan fixture; never played")
        annotation = root / f"{dataset}-{track_id}.beats"; annotation.write_text("0.0 1\n0.5 2\n")
        tracks.append(HeldoutTrack(f"{dataset}:{track_id}", dataset, track_id,
                                  f"Canción {index} · 演奏 🥁", audio, annotation,
                                  61.0 + index, tuple(np.arange(0, 61+index, .5)),
                                  tuple(np.arange(0, 61+index, 2.0)), "a"*64, "b"*64))
    tracks.sort(key=lambda t: t.uid)
    fold = FoldContext(root, 2, contract_hash="synthetic-contract", plan_membership_hash="synthetic-membership")
    catalog = {"schema": "mir.native-experiment-catalog/v1", "fold": fold.as_dict(),
               "split_provenance": {"contract_hash": fold.contract_hash, "plan_membership_hash": fold.plan_membership_hash},
               "test_uids": [t.uid for t in tracks], "excluded_uids": ["brid:999"],
               "source_manifests": [], "tracks": []}
    for t in tracks:
        row = asdict(t)
        for k in ("audio_path", "annotation_path"):
            row[k] = str(row[k])
        row["audio_file_sha256"] = hashlib.sha256(t.audio_path.read_bytes()).hexdigest()
        row["annotation_file_sha256"] = hashlib.sha256(t.annotation_path.read_bytes()).hexdigest()
        catalog["tracks"].append(row)
    catalog_path = args.output / "synthetic-catalog.json"; catalog_path.write_text(json.dumps(catalog))
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    plans = []
    for i in range(20):
        options = {"participant_id": f"P-élève-鼓-{i}", "participant_index": i-2,
                   "seed": seeds[i % len(seeds)], "excerpt_duration_seconds": None if i%4==0 else (60.0 if i%2==0 else 12.345),
                   "max_tracks": None if i%3==0 else i%8+1}
        plan = build_experiment_plan(tracks, fold=fold, **options)
        plans.append(plan)
        case({"operation": "plan", "catalog": str(catalog_path), "catalog_sha256": catalog_sha, "options": options}, plan.as_dict())

    captures = []
    plan = plans[0]
    for i, trial in enumerate(plan.trials):
        origin = 1_000_000_000_000 + i*100_000_000_000
        end = origin + int(trial.duration_seconds*1e9)
        times = [v for v in trial.beat_times_seconds if v>.15 and v<trial.duration_seconds-.8]
        inputs = [{"source": "synthetic-evdev", "control": "KEY_SPACE", "monotonic_ns": origin+int((v+rng.uniform(-.12,.12))*1e9), "value":1.0} for v in times[::2]]
        system = [{"event": "downbeat" if n%4==0 else "beat", "monotonic_ns": origin+int((v+.5)*1e9), "sequence":n, "predicted_track_seconds":v, "decision_ready_track_seconds":v+(.51 if i%3 else .025)} for n,v in enumerate(times)]
        # Track-relative readiness is explicitly independent of monotonic send;
        # preserve original validation and identity semantics.
        processes = [] if i%4==0 else [{"frame_index":n,"captured_monotonic_ns":origin+n*20_000_000,"ready_monotonic_ns":origin+n*20_000_000+(n+1)*4_000_000,"queue_seconds":n*.001,"processing_seconds":(n+1)*.004,"realtime_factor":(n+1)*.2,"classifier_process_seconds":.001,"dropped_audio_blocks":n//3} for n in range(8)]
        devices = [{"sequence":e["sequence"],"status":"event_applied" if n%3 else "bad_crc","config_revision":2,"received_monotonic_ns":e["monotonic_ns"]+5_000_000+n*1000,"source_host":"127.0.0.1","source_port":5005} for n,e in enumerate(system[::2])]
        raw = {"schema":"mir.rhythm-assist-trial-capture/v1","plan_id":plan.plan_id,"trial":trial.as_dict(),"input_device":"synthetic fixture, \"USB\"","playback_started_monotonic_ns":origin,"playback_ended_monotonic_ns":end,"input_events":inputs,"system_events":system,"process_observations":processes,"device_observations":devices,"pipeline_identity":"fixture","runtime_backend":"fixture","runtime_device":"host"}
        capture = TrialCapture.from_dict(raw); capture.validate(); captures.append(capture)
        case({"operation":"analyze","capture":capture.as_dict()}, analyze_trial(capture))
    analyses = [analyze_trial(c) for c in captures]
    case({"operation":"aggregate","analyses":analyses}, aggregate_trial_analyses(analyses))
    actual = native({"operation":"batch","requests":requests})
    for i, (a,e) in enumerate(zip(actual,expected)):
        compare(a,e,f"case-{i}/{requests[i]['operation']}")
        if requests[i]["operation"] in ("plan", "rng-probe"):
            assert a == e, f"case-{i} requires exact RNG, excerpt and identity equality"

    workspace = args.output / "native-workspace"
    native({"operation":"workspace-create","workspace":str(workspace),"plan":plan.as_dict()})
    source_ws = ExperimentWorkspace(plan, args.output / "source-workspace")
    for c in captures:
        p=args.output / "capture.json";p.write_text(json.dumps(c.as_dict()))
        req={"operation":"workspace-import","workspace":str(workspace),"capture":str(p)}
        compare(native(req), source_ws.save_capture(c))
        compare(native(req), source_ws.save_capture(c)) # idempotent re-import
    a=native({"operation":"workspace-report","workspace":str(workspace)})
    e=source_ws.aggregate();compare(a,e)
    # CSV numbers allow only the same f64 reduction bound; identity cells exact.
    def read_csv(p):
        with p.open(newline="") as f:return list(csv.DictReader(f))
    ac=read_csv(workspace/"trial-results.csv");ec=read_csv(source_ws.root/"trial-results.csv")
    for ar,er in zip(ac,ec):
        for k,v in er.items():
            try: compare(float(ar[k]),float(v),f"CSV/{k}")
            except ValueError: assert ar[k]==v,(k,ar[k],v)
    assert (workspace/"experiment-report.md").read_text() == (source_ws.root/"experiment-report.md").read_text()
    status=native({"operation":"workspace-status","workspace":str(workspace)})
    assert status["complete"] and status["next_incomplete_index"] is None

    rejected=[]
    def reject(r): rejected.append(native(r,reject=True).strip())
    reject({"operation":"event-accuracy","reference":[0,0],"estimate":[]})
    reject({"operation":"realtime-accuracy","reference":[],"predicted":[0],"ready":[]})
    reject({"operation":"event-accuracy","reference":[],"estimate":[],"tolerances_ms":[0]})
    reject({"operation":"plan","catalog":str(catalog_path),"catalog_sha256":"0"*64,"options":options})
    altered=copy.deepcopy(plan.as_dict());altered["participant_id"]="another participant"
    reject({"operation":"workspace-create","workspace":str(workspace),"plan":altered})
    p=args.output/"capture.json";bad=captures[0].as_dict();bad["notes"]="changed capture";p.write_text(json.dumps(bad))
    reject({"operation":"workspace-import","workspace":str(workspace),"capture":str(p)})
    with (workspace/".native-workflows.lock").open("r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reject({"operation":"workspace-status","workspace":str(workspace)})
    unsafe=copy.deepcopy(plan.as_dict());unsafe["trials"][0]["trial_id"]="../escape"
    rejected_root=args.output/"unsafe-workspace"
    reject({"operation":"workspace-create","workspace":str(rejected_root),"plan":unsafe})
    assert not rejected_root.exists(),"rejected plan changed the filesystem"
    bad=captures[0].as_dict();bad["trial"]["duration_seconds"]+=1;p.write_text(json.dumps(bad))
    reject({"operation":"workspace-import","workspace":str(workspace),"capture":str(p)})
    tracks[0].audio_path.write_bytes(b"changed")
    reject({"operation":"plan","catalog":str(catalog_path),"catalog_sha256":catalog_sha,"options":options})
    sources=[ROOT/"mir-desktop-app/src/mir_desktop_app"/f"{n}.py" for n in ("experiment","experiment_analysis","experiment_workspace")]
    report={"passed":True,"reference_cases":len(requests),"rng_seeds":len(seeds),"exact_plans":len(plans),"capture_count":len(captures),"workspace_idempotent_imports":2*len(captures),"rejections":rejected,"sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},"scope":"software workflow parity; synthetic participant events are not accuracy measurements"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
