#!/usr/bin/env python3
"""Check both PF RNG contracts against frozen draws, states and native replay.

Run in MIR. Baselines change only with --write; normal validation never updates
them. Legacy NumPy is exercised here, while deployment keeps portable-v1.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from mir_core.beats.schema import (
    EventActivations,
    to_exclusive_beat_downbeat_activation_data,
)
from mir_core.postprocessing.particle_filter import (
    PARTICLE_FILTER_RNG_LEGACY_GLOBAL_V1,
    PARTICLE_FILTER_RNG_PORTABLE_V1,
    ParticleFilterTracker,
    PortableParticleFilterRNG,
)
import inspect
import mir_core

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEFAULT_MANIFEST = FIXTURES / "particle_filter_rng_contract.json"
MAGIC = b"MIRPFC1\0"
MODES = (PARTICLE_FILTER_RNG_LEGACY_GLOBAL_V1, PARTICLE_FILTER_RNG_PORTABLE_V1)
PROFILES = (
    (
        "small",
        42,
        800,
        dict(min_bpm=80.0, max_bpm=180.0, particle_size=96, down_particle_size=24),
    ),
    (
        "locked4",
        12345,
        800,
        dict(
            min_bpm=80.0,
            max_bpm=180.0,
            particle_size=96,
            down_particle_size=24,
            min_beats_per_bar=4,
            max_beats_per_bar=4,
        ),
    ),
    ("packaged", 0, 800, {}),
    (
        "sustained",
        2**32 - 1,
        4096,
        dict(min_bpm=80.0, max_bpm=180.0, particle_size=96, down_particle_size=24),
    ),
)
BASELINE = {
    "repository": "Hashim-K/mir-core",
    "commit": "14658b3302b00d7b19b42511a5df39172d399ced",
    "path": "mir_core/postprocessing/particle_filter.py",
    "numpy": "1.26.4",
    "madmom_commit": "27f032e8947204902c675e5e341a3faf5dc86dae",
}


def canonical_trace() -> np.ndarray:
    """Return a deterministic activation trace exercising both injections."""

    frame_count = 800
    frame = np.arange(frame_count, dtype=np.int64)
    all_beats = (0.014 + (frame * 29 % 23) * 0.0011).astype(np.float32)
    downbeats = (0.002 + (frame * 7 % 9) * 0.0006).astype(np.float32)

    cursor = 14
    intervals = (22, 24, 21, 25, 23, 26, 20, 24)
    pulse = 0
    while cursor < frame_count - 3:
        # Strong centers exercise BeatNet's >0.8 beat-particle injection.
        all_beats[cursor] = np.float32(0.91 + 0.01 * (pulse % 7))
        if pulse % 4 == 0:
            # Canonical downbeat remains a subset of all-beat, while its
            # exclusive channel crosses the >0.7 down-particle threshold.
            downbeats[cursor] = np.float32(0.73 + 0.025 * (pulse % 3))
        else:
            downbeats[cursor] = np.float32(0.018 + 0.006 * (pulse % 3))
        # Moderate shoulders resample without injecting; low following frames
        # keep pulses separated and expose state-motion decisions.
        all_beats[cursor - 1] = np.float32(0.53 + 0.02 * (pulse % 2))
        downbeats[cursor - 1] = np.float32(0.035)
        all_beats[cursor + 1] = np.float32(0.21)
        downbeats[cursor + 1] = np.float32(0.012)
        cursor += intervals[pulse % len(intervals)]
        pulse += 1

    # A few non-periodic distractors exercise the information gate without
    # making the fixture depend on an external activation cache.
    for index, strength in ((77, 0.62), (188, 0.47), (333, 0.68), (611, 0.58)):
        all_beats[index] = np.float32(strength)
        downbeats[index] = np.float32(0.08)
    return np.column_stack((all_beats, downbeats)).astype(np.float32)


def _particle_fingerprint(values: np.ndarray) -> int:
    fingerprint = 0xCBF29CE484222325
    for value in np.asarray(values).reshape(-1):
        fingerprint ^= int(value)
        fingerprint = (fingerprint * 0x100000001B3) & ((1 << 64) - 1)
    return fingerprint


def _verify_packaged_profile():
    signature = inspect.signature(ParticleFilterTracker)
    names = set(signature.parameters) - {"beat_callback", "rng_contract", "random_seed"}

    def effective(payload):
        values = {
            name: payload.get(name, signature.parameters[name].default)
            for name in names
        }
        if values["beats_per_bar"] is None:
            values["beats_per_bar"] = []
        values["fps"] = 50.0
        return values

    expected = effective({})
    directory = Path(mir_core.__file__).resolve().parent / "checkpoints/trained/beatnet"
    paths = sorted(directory.glob("**/postprocessors/*particle-filter/params.json"))
    if not paths or any(
        effective(json.loads(path.read_text())) != expected for path in paths
    ):
        raise AssertionError(
            "Packaged PF parameters changed; extend the native contract profiles before validation."
        )


def assert_case_frozen(name, metadata, expected):
    if metadata != expected.get(name):
        raise AssertionError(
            f"Frozen RNG contract changed: {name}. Review the regression; do not regenerate in CI."
        )


class _Recorder:
    """Record distribution calls without replacing the generator's algorithms."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.reset()

    def reset(self):
        self.delegate.reset()
        self.tape = bytearray()
        self.rows = []
        self.row_ids = {}
        self.calls = Counter()

    def _integer(self, upper, value):
        self.tape.extend(struct.pack("<cQQ", b"R", int(upper), int(value)))
        self.calls["integer"] += 1

    def uniform_choices(self, upper, count):
        values = self.delegate.uniform_choices(upper, count)
        for value in values:
            self._integer(upper, value)
        return values

    def randbelow(self, upper):
        value = self.delegate.randbelow(upper)
        self._integer(upper, value)
        return value

    def categorical(self, choices, weights):
        value = self.delegate.categorical(choices, weights)
        choices = np.asarray(choices).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        positive = weights > 0.0
        row = tuple(map(float, weights[positive]))
        if row not in self.row_ids:
            self.row_ids[row] = len(self.rows)
            self.rows.append(row)
        selected = np.flatnonzero(choices[positive] == value)
        if len(selected) != 1:
            raise AssertionError(
                "Transition draw is not a unique positive-weight choice"
            )
        self.tape.extend(struct.pack("<cII", b"C", self.row_ids[row], int(selected[0])))
        self.calls["categorical"] += 1
        return value

    def choice(self, choices, size, *, p):
        if size != 1:
            raise AssertionError("Unexpected transition draw count")
        return np.asarray([self.categorical(choices, p)], dtype=np.int64)

    def uniform_offsets(self, count, high):
        values = self.delegate.uniform_offsets(count, high)
        for value in values:
            self.tape.extend(struct.pack("<cdd", b"O", float(high), float(value)))
            self.calls["offset"] += 1
        return values

    def sample_without_replacement(self, population, count):
        values = self.delegate.sample_without_replacement(population, count)
        self.tape.extend(struct.pack("<cII", b"S", population, count))
        self.tape.extend(np.asarray(values, dtype="<u4").tobytes())
        self.calls["sample"] += 1
        return values


class _PortableRecorder(_Recorder, PortableParticleFilterRNG):
    # The current Python resampler selects its versioned scalar arithmetic with
    # isinstance(PortableParticleFilterRNG). Preserve that dispatch when wrapping
    # the RNG. The uninstrumented run below independently checks this assumption.
    pass


def _states(tracker):
    return (
        _particle_fingerprint(tracker.particles),
        _particle_fingerprint(tracker.down_particles),
    )


def run_python(config, seed, values, *, record):
    ambient = np.random.get_state()
    try:
        np.random.seed(seed)
        tracker = ParticleFilterTracker(**config)
        recorder = None
        if record:
            kind = (
                _PortableRecorder if config["rng_contract"] == MODES[1] else _Recorder
            )
            recorder = kind(tracker._rng)
            tracker._rng = recorder
            # Legacy reset intentionally does not rewind ambient NumPy state.
            # Explicit reseeding recreates the independently checked initial run.
            np.random.seed(seed)
            tracker.reset()
        initial = _states(tracker)
        frames = bytearray()
        event_count = 0
        for row in values:
            exclusive = to_exclusive_beat_downbeat_activation_data(
                EventActivations(row.reshape(1, 2)), dtype=np.float64
            )
            decoded = tracker.process(exclusive)
            label = 0
            if len(decoded) != event_count:
                assert len(decoded) == event_count + 1
                label = int(decoded[-1, 1])
                event_count += 1
            frames.extend(
                struct.pack(
                    "<ffBQQ", float(row[0]), float(row[1]), label, *_states(tracker)
                )
            )
        return initial, bytes(frames), recorder, event_count
    finally:
        np.random.set_state(ambient)


def make_fixture(mode, seed, frame_count, parameters):
    config = dict(
        fps=50.0,
        num_tempi=300,
        min_bpm=55.0,
        max_bpm=215.0,
        min_beats_per_bar=2,
        max_beats_per_bar=4,
        particle_size=1500,
        down_particle_size=250,
        rng_contract=mode,
        **({"random_seed": seed} if mode == MODES[1] else {}),
    )
    config.update(parameters)
    values = np.tile(canonical_trace(), ((frame_count + 799) // 800, 1))[:frame_count]
    initial, frames, recorder, event_count = run_python(
        config, seed, values, record=True
    )
    bare_initial, bare_frames, _, bare_events = run_python(
        config, seed, values, record=False
    )
    if (initial, frames, event_count) != (bare_initial, bare_frames, bare_events):
        raise AssertionError("Recording changed the real Python tracker")
    data = bytearray(
        struct.pack(
            "<8sIIQIIddII",
            MAGIC,
            MODES.index(mode),
            frame_count,
            seed,
            config["particle_size"],
            config["down_particle_size"],
            config["min_bpm"],
            config["max_bpm"],
            config["min_beats_per_bar"],
            config["max_beats_per_bar"],
        )
    )
    data.extend(struct.pack("<QQ", *initial))
    data.extend(frames)
    data.extend(struct.pack("<I", len(recorder.rows)))
    for row in recorder.rows:
        data.extend(struct.pack("<I", len(row)))
        data.extend(np.asarray(row, dtype="<f8").tobytes())
    data.extend(struct.pack("<Q", len(recorder.tape)))
    data.extend(recorder.tape)
    metadata = dict(
        rng_contract=mode,
        seed=seed,
        frames=frame_count,
        events=event_count,
        calls=dict(recorder.calls),
        sha256=hashlib.sha256(data).hexdigest(),
        state_trace_sha256=hashlib.sha256(frames).hexdigest(),
        bytes=len(data),
    )
    return bytes(data), metadata


def assert_frozen(path, data):
    if not path.is_file() or path.read_bytes() != data:
        raise AssertionError(
            f"Frozen RNG contract changed: {path}. Review the regression; do not regenerate in CI."
        )


def check(
    fixtures=None,
    *,
    manifest_path=DEFAULT_MANIFEST,
    write=False,
    replay=None,
    production=None,
):
    if bool(replay) != bool(production) or (replay and fixtures is None):
        raise ValueError(
            "Native validation requires fixtures and both native executables"
        )
    _verify_packaged_profile()
    manifest_path = (
        fixtures / "manifest.json" if fixtures is not None else manifest_path
    )
    expected = None if write else json.loads(manifest_path.read_text())
    if fixtures is not None and not write:
        if expected != json.loads(DEFAULT_MANIFEST.read_text()):
            raise AssertionError("Native and shared RNG contract manifests differ")
    cases = {}
    for profile, seed, frames, parameters in PROFILES:
        for index, mode in enumerate(MODES):
            name = f"{profile}-{'legacy' if index == 0 else 'portable'}.bin"
            data, metadata = make_fixture(mode, seed, frames, parameters)
            path = fixtures / name if fixtures is not None else None
            if write and path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            elif not write:
                assert_case_frozen(name, metadata, expected["cases"])
                if path is not None:
                    assert_frozen(path, data)
            if replay:
                subprocess.run([str(replay), str(path)], check=True)
            if production and index == 1:
                subprocess.run([str(production), str(path)], check=True)
            cases[name] = metadata
            print(
                f"PASS {name}: {frames} frames, {metadata['events']} events", flush=True
            )
    manifest = dict(
        schema="mir.particle-filter-dual-rng/v1", baseline=BASELINE, cases=cases
    )
    path = manifest_path
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    elif json.loads(path.read_text()) != manifest:
        raise AssertionError("Frozen RNG manifest changed")
    return dict(
        **manifest,
        native_reference_checked=replay is not None,
        native_production_checked=production is not None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        help="Native binary fixtures; omit for the shared Python manifest",
    )
    parser.add_argument(
        "--write", action="store_true", help="Explicitly regenerate reviewed baselines"
    )
    parser.add_argument("--replay", type=Path, help="Native reference-mode executable")
    parser.add_argument(
        "--production", type=Path, help="Native production-mode executable"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.replay or args.production) and args.fixtures is None:
        parser.error("Native replay requires --fixtures")
    if bool(args.replay) != bool(args.production):
        parser.error("Native validation requires both --replay and --production")
    result = check(
        args.fixtures, write=args.write, replay=args.replay, production=args.production
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
