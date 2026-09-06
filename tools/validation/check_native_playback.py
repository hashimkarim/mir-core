"""Exercise actual native playback/trial orchestration on isolated host devices.

The C fixture implements the public PortAudio and evdev ABI, with a real paced
callback thread. It never contacts speakers, participant devices or the phone.
Python playback, capture validation and analysis remain independent oracles.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mir-desktop-app/src"))
from mir_desktop_app import playback as source
from mir_desktop_app.experiment import TrialSpec, TrialCapture
from mir_desktop_app.pipeline import FoldContext
from mir_desktop_app.experiment_analysis import analyze_trial
from mir_desktop_app.experiment_panel import _aligned_runtime_system_event
from mir_desktop_app.haptics_protocol import ActuationPacket, DeviceAck, AckStatus
from check_native_playback_math import compare
from mir_desktop_app.tap_latency import analyze_tap_latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, default=ROOT / "mir-desktop-app/native-workflows/target/debug/mir-native-workflows")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).parent / "fixtures/native_playback_host.c"
    library = out / "libportaudio.so.2"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(fixture), "-o", str(library), "-ldl", "-lpthread"], check=True)
    evdev = out / "evdev-fixture"; evdev.touch()
    played = out / "played.f32"
    env = {**os.environ, "LD_LIBRARY_PATH": str(out) + ":" + os.environ.get("LD_LIBRARY_PATH", ""), "LD_PRELOAD": str(library), "MIR_TEST_EVDEV": str(evdev), "MIR_TEST_PLAYED_PCM": str(played)}
    def native(r, changes=None, reject=False):
        p = subprocess.run([str(args.native_bin)], input=json.dumps(r), text=True, capture_output=True, env={**env, **(changes or {})}, timeout=30)
        if reject:
            assert p.returncode != 0, p.stdout
            assert "panicked" not in p.stderr, p.stderr
            return p.stderr
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)
    rng = np.random.default_rng(83391)
    checks = 0
    # Source GUI labels, deduplication and source command settings.
    nodes = [{"type": "PipeWire:Interface:Node", "info": {"props": {"media.class": kind, "node.name": f"sink-{i}", "node.description": name, "node.nick": nick, "device.profile.description": profile}}} for i, (kind, name, nick, profile) in enumerate([
        ("Audio/Sink", "Audio Digital Stereo (HDMI 2)", "", ""),
        ("Audio/Sink/Internal", "Device 4", "", ""),
        ("Audio/Sink", "Playback device", "Device", "Playback"),
        ("Audio/Sink", "Music Analog Stereo", "", "Analog Stereo"),
        ("Audio/Source", "Input", "", ""),
        ("Audio/Sink", "External", "DAC", "Surround 5.1"),
    ])]
    nodes += [nodes[0], {}, {"type": "wrong"}]
    actual = native({"operation": "playback-nodes", "nodes": nodes, "default_sink": "sink-3"})
    assert actual == [asdict(o) for o in source._pipewire_outputs_from_dump(nodes, default_sink="sink-3")]; checks += 1
    output = source.PlaybackOutput("pipewire:test", "Test", "pipewire", "private-sink", True)
    assert native({"operation": "playback-command", "output": asdict(output), "sample_rate": 22050, "channels": 2}) == source._pipewire_playback_command("pw-play", output, sample_rate=22050, channels=2); checks += 1
    origin = 1_000_000_000_000
    for delay, num, den, invalid in [(13, 1, 22050, False), (-3, 1, 48000, False), (100, 1, 0, True), (1, 1, 44100, True)]:
        now = 7 if invalid else origin
        candidate = now + round(max(0, delay)*num/max(1, den)*1e9)
        observed = origin + 1_000_000; streaming = origin-7
        fallback = not origin-1e9 <= candidate <= observed+1e9
        expected = [streaming+20_000_000 if fallback else candidate, fallback]
        assert native({"operation": "playback-clock", "line": f"stream time: now:{now} rate:{num}/{den} foo delay:{delay}", "launched_ns": origin, "observed_ns": observed, "streaming_ns": streaming}) == expected; checks += 1
    for i in range(100):
        frame = int(rng.integers(5, 200)); offset = float(rng.uniform(-.1, .1)); queue = float(rng.uniform(0, .1)); work = float(rng.uniform(0, .1))
        sent = origin + round((frame*.02+.3)*1e9)
        runtime = SimpleNamespace(event_audio_seconds=frame*.02-.02, emitted_frame_index=frame, queue_seconds=queue, pipeline_seconds=work, event=SimpleNamespace(name="BEAT"), route_label="stock")
        record = SimpleNamespace(runtime=runtime, datagram=SimpleNamespace(sent_monotonic_ns=sent, packet=SimpleNamespace(sequence=i)))
        expected = _aligned_runtime_system_event(record, playback_started_ns=origin, hop_seconds=.02, stream_offset_seconds=offset).as_dict()
        request = {"operation": "align-runtime-event", "event": {"event": "beat", "frame_index": frame, "audio_seconds": runtime.event_audio_seconds, "queue_seconds": queue, "pipeline_seconds": work, "sent_ns": sent, "sequence": i, "route": "stock"}, "start_ns": origin, "offset_seconds": offset}
        compare(native(request), expected); checks += 1
    for key, value in (("frame_index", 2**64-1), ("audio_seconds", 1e300), ("pipeline_seconds", 1e300)):
        bad=copy.deepcopy(request);bad["event"][key]=value
        native(bad,reject=True)
    rate = 22050; audio = (rng.standard_normal((int(rate*1.6), 2))*.05).astype(np.float32)
    path = out / "audio.wav"; sf.write(path, audio, rate, subtype="FLOAT")
    trial = TrialSpec("trial-playback", "no_assist", "fixture:001", "fixture", "Fixture", path, hashlib.sha256(path.read_bytes()).hexdigest(), "a"*64, 0.0, 1.6, (.45,.95,1.45), (.45,), 0).as_dict()
    request = {"operation": "playback", "trial": trial, "output": {"identifier": "portaudio:default", "label": "Test output", "backend": "portaudio", "target": None, "is_default": True}, "resampler_library": str(ROOT / "mir-core/native-dsp/build/libmir_dsp.so")}
    playback_cases = []
    for resampling in (False, True):
        result = native(request, {"MIR_TEST_RESAMPLE": "1"} if resampling else None)
        expected = soxr.resample(audio, rate, 48000, quality="HQ") if resampling else audio
        actual = np.fromfile(played, np.float32).reshape(-1, 2)
        np.testing.assert_array_equal(actual[:len(expected)], expected)
        assert np.all(actual[len(expected):] == 0)
        duration = (result["playback_ended_monotonic_ns"]-result["playback_started_monotonic_ns"])*1e-9
        assert 1.55 <= duration <= 1.8, duration
        assert result["clock_basis"] == "portaudio-dac-callback"
        playback_cases.append({"resampling": resampling, "samples_exact": True, "duration_seconds": duration})
    failures = [native(request, {key: "1"}, reject=True) for key in ("MIR_TEST_UNDERFLOW", "MIR_TEST_BAD_CLOCK")]
    # Run the actual nonblocking PipeWire writer and clock parser using a
    # private backend executable; the PCM crossing stdin must remain unchanged.
    backend=out/"pw-play"
    backend.write_text("#!"+sys.executable+"\n"+'''import json,os,sys,time
from pathlib import Path
Path(os.environ["MIR_TEST_ARGV"]).write_text(json.dumps(sys.argv[1:]))
now=time.monotonic_ns()
print("paused -> streaming",file=sys.stderr,flush=True)
line=f"stream time: now:{now} rate:1/22050 delay:441"
if not os.environ.get("MIR_TEST_FINAL_CLOCK"):print(line,file=sys.stderr,flush=True)
raw=sys.stdin.buffer.read()
Path(os.environ["MIR_TEST_PLAYED_PCM"]).write_bytes(raw)
time.sleep(len(raw)/4/2/22050+0.02)
if os.environ.get("MIR_TEST_FINAL_CLOCK"):sys.stderr.write(line);sys.stderr.flush()
''')
    backend.chmod(0o700)
    pwrequest={**request,"output":asdict(output)}
    argv=out/"pw-argv.json"
    changes={"PATH":str(out)+":"+os.environ["PATH"],"MIR_TEST_ARGV":str(argv)}
    result=native(pwrequest,changes)
    assert result["clock_basis"]=="pipewire-stream-time"
    assert json.loads(argv.read_text())==source._pipewire_playback_command("pw-play",output,sample_rate=rate,channels=2)[1:]
    np.testing.assert_array_equal(np.fromfile(played,np.float32).reshape(-1,2),audio)
    playback_cases.append({"backend":"pipewire","samples_exact":True,"clock_basis":result["clock_basis"]})
    final_clock=native(pwrequest,{**changes,"MIR_TEST_FINAL_CLOCK":"1"})
    assert final_clock["clock_basis"]=="pipewire-stream-time"
    playback_cases.append({"backend":"pipewire","final_unterminated_clock":True,"clock_basis":final_clock["clock_basis"]})
    # Real trial orchestration, private paced output and private input events.
    plan = {"schema": "mir.rhythm-assist-experiment-plan/v1", "plan_id": "plan-fixture", "participant_id": "fixture-not-human", "seed": 42, "fold": FoldContext(out, 0).as_dict(), "trials": [trial], "rt_tolerances_ms": [30,50,70,100,150]}
    trial_cases = []
    for condition in ("no_assist", "ground_truth"):
        workspace = out / condition; p = copy.deepcopy(plan); p["trials"][0]["condition"] = condition
        native({"operation": "workspace-create", "workspace": str(workspace), "plan": p})
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); receiver.bind(("127.0.0.1", 0)); receiver.settimeout(.1)
        packets = []; stop = threading.Event(); errors = []
        def receive():
            while not stop.is_set():
                try:
                    raw, addr = receiver.recvfrom(256); packet = ActuationPacket.from_bytes(raw); packets.append(packet)
                    ack = DeviceAck(AckStatus.OK, packet.sequence, 7)
                    receiver.sendto(ack.to_bytes(), addr)
                except socket.timeout: pass
                except BaseException as exc: errors.append(str(exc)); return
        worker = threading.Thread(target=receive); worker.start()
        r = {**request, "operation": "trial-run", "workspace": str(workspace), "input": {"backend": "evdev", "identifier": str(evdev)}, "target": "%s:%s" % receiver.getsockname()}
        try: result = native(r)
        finally: stop.set(); worker.join(1); receiver.close()
        assert not errors, errors
        saved = json.loads((workspace/"captures/trial-playback.json").read_text())
        capture = TrialCapture.from_dict(saved); capture.validate()
        assert len(capture.input_events) == 3, capture.input_events
        compare(result["analysis"], analyze_trial(capture))
        assert len(packets) == (3 if condition == "ground_truth" else 0)
        assert len(capture.system_events) == len(packets)
        if packets:
            assert [p.sequence for p in packets] == [0,1,2]
            assert all(0 < p.delay_us <= 250000 for p in packets)
            assert len(capture.device_observations) == 3
        assert result["status"]["complete"]
        failures.append(native(r, reject=True))
        trial_cases.append({"condition": condition, "input_events": 3, "packets": len(packets), "source_capture_and_analysis_passed": True})
    for failure in ("cancel", "underflow"):
        workspace = out / f"failed-{failure}"; native({"operation": "workspace-create", "workspace": str(workspace), "plan": plan})
        r = {**request, "operation": "trial-run", "workspace": str(workspace), "input": {"backend": "evdev", "identifier": str(evdev)}}
        if failure == "cancel":
            process = subprocess.Popen([str(args.native_bin)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            process.stdin.write(json.dumps(r)); process.stdin.close(); process.stdin = None
            time.sleep(.3); start = time.monotonic(); process.send_signal(signal.SIGTERM); stdout, stderr = process.communicate(timeout=3)
            assert process.returncode != 0 and "canceled" in stderr, (stdout, stderr)
            assert time.monotonic()-start < 1.0
        else: failures.append(native(r, {"MIR_TEST_UNDERFLOW": "1"}, reject=True))
        assert not list((workspace/"captures").glob("*.json"))
    calibration=native({**request,"operation":"tap-calibrate","bpm":240,"count_in_beats":1,"measured_beats":3,"input":{"backend":"evdev","identifier":str(evdev)}})
    expected=analyze_tap_latency(calibration["tap_seconds"],calibration["measured_beats_seconds"])
    assert len(calibration["tap_seconds"])==2
    for key,value in asdict(expected).items():compare(calibration[key],value,f"calibration/{key}")
    report = {"passed": True, "source_cases": checks, "playback_cases": playback_cases, "trial_cases": trial_cases, "failure_cases": failures, "invalid_event_clocks_rejected":3, "tap_calibration":{"input_events":len(calibration["tap_seconds"]),"source_scores_exact":True}, "cancel_under_one_second": True, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "scope": "isolated host callback, input, trial and UDP integration; no physical devices accessed or acoustic latency measured"}
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
