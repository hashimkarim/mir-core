"""Trace a PF divergence while holding implementation, RNG and seed constant.

Consumes the saved artifacts from check_promoted_streams.py. This diagnostic
does not change a decoder, input, tolerance or expected event.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from check_promoted_streams import digest, read_outputs, read_u32
from mir_desktop_app.runtime import CausalPostprocessor
import mir_core.postprocessing.particle_filter as pf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('saved_run', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--model-id', required=True)
    p.add_argument('--selector', default='stock-particle-filter')
    p.add_argument('--postprocessor-library', required=True, type=Path)
    args = p.parse_args()
    os.environ['MIR_EMBEDDED_PP_HOST_LIBRARY'] = str(args.postprocessor_library.resolve())
    root = Path(__file__).resolve().parents[3]
    catalog_path = root / 'mir-android-app/app/src/main/assets/models/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    item = next(m for m in catalog['beat_models'] if m['id'] == args.model_id)
    parameters = dict(item['postprocessors'][args.selector]['parameters'])
    assert parameters['method'] == 'particle_filter'
    parameters['rng_contract'] = 'portable-splitmix64-v1'
    folder = args.saved_run / args.model_id.replace('/', '_')
    reference_file = folder / 'python_reference.npz'
    replay_file = folder / 'cpp.replay'
    reference = np.load(reference_file)['activations']
    with replay_file.open('rb') as stream:
        count = read_u32(stream)
        actual = np.stack([read_outputs(stream)['activations'].reshape(2) for _ in range(count)])[:-1]
        assert not stream.read(1)
    assert actual.shape == reference.shape
    python = [CausalPostprocessor(fps=50, parameters=parameters, random_seed=42, backend='python') for _ in range(2)]
    native = [CausalPostprocessor(fps=50, parameters=parameters, random_seed=42, backend='native') for _ in range(3)]
    assert all(d.backend == 'cpp-ctypes' for d in native)
    result = dict(schema='mir.pf-input-sensitivity/v1', id=args.model_id, selector=args.selector,
        seed=42, rng_contract=parameters['rng_contract'], frames=len(reference),
        catalog_sha256=digest(catalog_path), reference_sha256=digest(reference_file),
        cpp_replay_sha256=digest(replay_file), library_sha256=digest(args.postprocessor_library),
        max_activation_difference=float(np.max(np.abs(actual-reference))),
        first_particle_difference=None, first_rng_state_difference=None,
        first_event_difference=None, same_input_native_repeat_exact=True,
        python_cpp_same_input_exact=[True, True])
    original = pf._universal_resample
    captured = [[], []]
    side = 0

    def capture(particles, weights, rng):
        before = int(rng.state)
        value = original(particles, weights, rng)
        captured[side].append((np.array(particles, copy=True), np.array(weights, copy=True),
                               before, np.array(value, copy=True)))
        return value

    pf._universal_resample = capture
    counts = [0, 0]
    try:
        for frame, pair in enumerate(zip(reference, actual)):
            before_equal = python[0]._tracker._rng.state == python[1]._tracker._rng.state
            captured = [[], []]
            rows = []
            for side, (decoder, value) in enumerate(zip(python, pair)):
                rows.append(decoder.process_canonical_values(*map(float, value)))
            for j, value in enumerate(pair):
                output = native[j].process_canonical_values(*map(float, value))
                result['python_cpp_same_input_exact'][j] &= np.array_equal(output, rows[j])
                counts[j] += len(output)
                if j == 0:
                    repeated = native[2].process_canonical_values(*map(float, value))
                    result['same_input_native_repeat_exact'] &= np.array_equal(output, repeated)
            if result['first_event_difference'] is None and not np.array_equal(*rows):
                result['first_event_difference'] = dict(frame=frame, seconds=frame/50,
                    reference=rows[0].tolist(), variant=rows[1].tolist())
            if result['first_rng_state_difference'] is None and python[0]._tracker._rng.state != python[1]._tracker._rng.state:
                result['first_rng_state_difference'] = dict(frame=frame, seconds=frame/50)
            if result['first_particle_difference'] is None:
                equal = all(np.array_equal(getattr(python[0]._tracker, name), getattr(python[1]._tracker, name))
                            for name in ('particles', 'down_particles'))
                if not equal:
                    first = dict(frame=frame, seconds=frame/50, rng_equal_before_frame=before_equal,
                        reference_activation=pair[0].tolist(), variant_activation=pair[1].tolist(),
                        resampling_calls=[len(calls) for calls in captured])
                    for a, b in zip(*captured):
                        if np.array_equal(a[3], b[3]):
                            continue
                        first.update(resampling_rng_states_equal=a[2] == b[2],
                            resampling_particles_equal=np.array_equal(a[0], b[0]),
                            max_weight_difference=float(np.max(np.abs(a[1]-b[1]))))
                        if a[2] == b[2] and np.array_equal(a[0], b[0]):
                            position = int(np.flatnonzero(a[3] != b[3])[0])
                            rng = pf.PortableParticleFilterRNG(42)
                            rng.state = a[2]
                            location = rng.uniform_offsets(len(a[0]), 1/len(a[0]))[position] + position*(1/len(a[0]))
                            sums = []
                            for call in (a, b):
                                total = 0.0
                                for w in call[1]: total += float(w)
                                cumulative = 0.0
                                cdf = []
                                for w in call[1]/total:
                                    cumulative += float(w)
                                    cdf.append(cumulative)
                                cdf[-1] = 1.0
                                sums.append(cdf)
                            choices = [int(np.searchsorted(cdf, location)) for cdf in sums]
                            boundary = min(choices)
                            first.update(particle_slot=position, identical_random_location=float(location),
                                selected_indices=choices, selected_states=[int(a[3][position]), int(b[3][position])],
                                cumulative_probability_at_boundary=[float(cdf[boundary]) for cdf in sums])
                        break
                    result['first_particle_difference'] = first
                    print(json.dumps(first), flush=True)
            if frame % 1000 == 0:
                print(f'processed {frame}/{len(reference)}', flush=True)
    finally:
        pf._universal_resample = original
        for decoder in native: decoder._tracker.close()
    result['event_counts'] = counts
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    assert result['same_input_native_repeat_exact']
    assert all(result['python_cpp_same_input_exact'])


if __name__ == '__main__':
    main()
