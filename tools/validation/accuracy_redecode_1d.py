"""Re-score saved neural outputs with a corrected decoder, retaining the old run.

Every 1D selector is independently checked on both saved activation streams.
No inference, checkpoint, input, annotation, or parameter selection is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from accuracy_inventory import digest, write_new
from accuracy_replay import decode, scores, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    os.environ['MIR_EMBEDDED_PP_HOST_LIBRARY'] = str(args.library.resolve())
    library_hash = digest(args.library)
    inventory = json.loads((args.catalogs/'inventory.json').read_text())
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    assert digest(catalog_path) == inventory['catalog_sha256']
    catalog = {row['id']: row for row in json.loads(catalog_path.read_text())['beat_models']}
    tracks = {row['uid']: row for path in args.catalogs.glob('fold-*.json')
              for row in json.loads(path.read_text())['tracks']}
    seen = set()
    while True:
        for path in sorted(args.replay.glob('*/*.json')):
            if path in seen:
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # The producer writes the completion record last.
            if record.get('schema') != 'mir.heldout-port-score/v1':
                continue
            destination = args.output/path.relative_to(args.replay)
            identity = {'original_score_sha256': digest(path), 'runner_sha256': digest(Path(__file__)),
                        'decoder_library_sha256': library_hash, 'arrays_sha256': record['arrays']['sha256']}
            if destination.exists():
                saved = json.loads(destination.read_text())
                assert saved['identity'] == identity
                assert digest(destination.parent/saved['arrays']['file']) == saved['arrays']['sha256']
                seen.add(path)
                continue
            model = catalog[record['model_id']]
            assert stable_hash(model['postprocessors']) == record['identity']['postprocessors_sha256']
            array_path = path.parent/record['arrays']['file']
            assert digest(array_path) == record['arrays']['sha256']
            results, updated, reused = [], {}, {}
            with np.load(array_path, allow_pickle=False) as arrays:
                for prior in record['decoders']:
                    if prior['method'] != 'heydari_1d_state_space':
                        continue
                    selector = prior['selector']
                    parameters = model['postprocessors'][selector]['parameters']
                    key = stable_hash(parameters)
                    if key not in reused:
                        native_original, _ = decode(arrays['python'], parameters, 'native')
                        python_deployed, _ = decode(arrays['native'], parameters, 'python')
                        native_deployed, native_seconds = decode(arrays['native'], parameters, 'native')
                        reused[key] = native_original, python_deployed, native_deployed, native_seconds
                    native_original, python_deployed, native_deployed, native_seconds = reused[key]
                    reference = arrays[selector+'_python_events']
                    native_scores = scores(native_deployed, tracks[record['track_uid']])
                    results.append(dict(prior, native_metrics=native_scores,
                        metric_deltas={key: native_scores[key]-value for key, value in prior['python_metrics'].items()},
                        event_stream_exact=np.array_equal(reference, native_deployed),
                        native_events=len(native_deployed),
                        decoder_seconds={'python': prior['decoder_seconds']['python'], 'native': native_seconds},
                        same_original_input_decoder_events_exact=np.array_equal(reference, native_original),
                        same_deployed_input_decoder_events_exact=np.array_equal(python_deployed, native_deployed),
                        changed_by_fix=not np.array_equal(arrays[selector+'_native_events'], native_deployed)))
                    for name, values in [('native', native_deployed), ('same_input_python', python_deployed), ('original_input_native', native_original)]:
                        updated[selector+'_'+name+'_events'] = values
            destination.parent.mkdir(parents=True, exist_ok=True)
            new_arrays = destination.with_suffix('.npz')
            with new_arrays.open('xb') as stream:
                np.savez_compressed(stream, **updated)
            value = {'schema': 'mir.heldout-1d-decoder-correction/v1', 'identity': identity,
                'model_id': record['model_id'], 'track_uid': record['track_uid'],
                'arrays': {'file': new_arrays.name, 'sha256': digest(new_arrays)}, 'decoders': results}
            write_new(destination, value)
            seen.add(path)
            print(json.dumps({'completed': len(seen), 'model': record['model_id'], 'track': record['track_uid'],
                'selectors': len(results), 'changed_streams': sum(row['changed_by_fix'] for row in results),
                'same_input_failures': sum(not row['same_original_input_decoder_events_exact'] or
                                          not row['same_deployed_input_decoder_events_exact'] for row in results)}), flush=True)
        if len(seen) == len(inventory['beat_jobs']) or not args.watch:
            break
        time.sleep(5)
    write_new(args.output/'completion.json', {'complete': len(seen) == len(inventory['beat_jobs']),
              'pairs': len(seen), 'expected': len(inventory['beat_jobs']), 'decoder_library_sha256': library_hash})


if __name__ == '__main__':
    main()
