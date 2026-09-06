"""Cache causal frontends on shared, offline-resampled complete recordings.

This isolates frontend/inference fidelity and conventional musical accuracy.
It does not measure source-arrival latency through a live sample-rate converter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import soxr

from accuracy_inventory import digest, write_new


def one_track(task):
    track, output = task
    from mir_core.preprocessing import BeatNetPreProcessor
    from mir_native_runtime import LogSpectFrontend
    output = Path(output)
    stem = track['uid'].replace(':', '__')
    record_path = output/(stem+'.json')
    identity = {'audio_sha256':track['audio_hash'], 'annotation_sha256':track['annotation_hash'],
                'contract':'shared-offline-pcm-causal-features-soxr-HQ-22050-441-2290/v2'}
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['identity'] == identity
        for name, item in record['arrays'].items():
            assert digest(output/item['file']) == item['sha256'], name
        return {'uid':track['uid'], 'resumed':True, 'frames':record['frames']}
    assert digest(Path(track['audio_path'])) == identity['audio_sha256']
    assert digest(Path(track['annotation_path'])) == identity['annotation_sha256']
    started = time.monotonic()
    audio, rate = sf.read(track['audio_path'], dtype='float32', always_2d=True)
    audio = np.asarray(audio.mean(axis=1), np.float32)
    assert np.isfinite(audio).all()
    audio = np.ascontiguousarray(soxr.resample(audio, rate, 22050, quality='HQ'), np.float32)
    reference = BeatNetPreProcessor(mode='realtime')
    native = LogSpectFrontend.from_preset('beatnet')
    count = len(audio)//441
    original = np.empty((count,272), np.float32)
    deployed = np.empty_like(original)
    window = np.zeros(2290, np.float32)
    for frame, hop in enumerate(audio[:count*441].reshape(-1,441)):
        window[:-441] = window[441:]
        window[-441:] = hop
        original[frame] = np.asarray(reference.process_audio(window), np.float32)[-1]
        deployed[frame] = native.process_hop(hop)
    assert np.isfinite(original).all() and np.isfinite(deployed).all()
    delta = np.abs(deployed-original)
    arrays = {}
    for name, value in [('python',original), ('rust',deployed)]:
        path = output/(stem+'.'+name+'.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        arrays[name] = {'file':path.name, 'sha256':digest(path), 'shape':list(value.shape)}
    record = {'schema':'mir.heldout-causal-features/v1', 'uid':track['uid'], 'identity':identity,
              'source_sample_rate':rate, 'resampled_samples':len(audio),
              'source_conversion':'shared offline SoXR HQ; live resampling latency excluded',
              'trailing_samples_less_than_one_hop':len(audio)-count*441,
              'frames':count, 'arrays':arrays, 'reference':'original madmom causal window',
              'deployment':'shared Rust/C++ LogSpectFrontend beatnet preset',
              'max_abs_error':float(delta.max()),
              'existing_atol':1e-6, 'violating_elements':int(np.count_nonzero(delta>1e-6)),
              'wall_seconds':time.monotonic()-started}
    write_new(record_path, record)
    return {key:record[key] for key in ['uid','frames','max_abs_error','violating_elements','wall_seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    tracks = [row for path in sorted(args.catalogs.glob('fold-*.json'))
              for row in json.loads(path.read_text())['tracks']]
    assert len({row['uid'] for row in tracks}) == len(tracks)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(row,str(args.output)) for row in tracks[:args.limit]]
    if args.workers == 1:
        for task in tasks:
            print(json.dumps(one_track(task)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(one_track, tasks):
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
