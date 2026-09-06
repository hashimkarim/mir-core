"""Check identical-input decoders for every differing saved event stream.

This leaves the baseline arrays and scores untouched. An explicitly bound native
library must be the library used by the baseline; corrected decoders can be
excluded when their full re-score has a separate evidence directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from accuracy_inventory import digest, write_new
from accuracy_replay import decode, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--exclude-method', action='append', default=[])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    os.environ['MIR_EMBEDDED_PP_HOST_LIBRARY'] = str(args.library.resolve())
    library_hash = digest(args.library)
    inventory = json.loads((args.catalogs/'inventory.json').read_text())
    expected = {(row['model_id'], row['track_uid']) for row in inventory['beat_jobs']}
    assert len(expected) == len(inventory['beat_jobs'])
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    assert digest(catalog_path) == inventory['catalog_sha256']
    catalog = {row['id']: row for row in json.loads(catalog_path.read_text())['beat_models']}
    seen, checked, failures = set(), 0, 0
    while True:
        for path in sorted(args.replay.glob('*/*.json')):
            if path in seen:
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if record.get('schema') != 'mir.heldout-port-score/v1':
                continue
            assert record['complete'] and (record['model_id'], record['track_uid']) in expected
            model = catalog[record['model_id']]
            assert record['identity']['postprocessors_sha256'] == stable_hash(model['postprocessors'])
            selectors = [row for row in record['decoders'] if not row['event_stream_exact']
                         and row['method'] not in args.exclude_method]
            identity = {'baseline_score_sha256': digest(path),
                        'baseline_arrays_sha256': record['arrays']['sha256'],
                        'decoder_library_sha256': library_hash,
                        'runner_sha256': digest(Path(__file__)),
                        'decode_helper_sha256': digest(Path(__file__).with_name('accuracy_replay.py')),
                        'excluded_methods': sorted(args.exclude_method)}
            destination = args.output/path.relative_to(args.replay)
            if destination.exists():
                value = json.loads(destination.read_text())
                assert value['identity'] == identity
                assert digest(destination.parent/value['arrays']['file']) == value['arrays']['sha256']
            else:
                array_path = path.parent/record['arrays']['file']
                assert digest(array_path) == record['arrays']['sha256']
                results, evidence, reused = [], {}, {}
                with np.load(array_path, allow_pickle=False) as arrays:
                    for row in selectors:
                        selector = row['selector']
                        parameters = model['postprocessors'][selector]['parameters']
                        key = stable_hash(parameters)
                        if key not in reused:
                            cpp_original, _ = decode(arrays['python'], parameters, 'native')
                            python_deployed, _ = decode(arrays['native'], parameters, 'python')
                            reused[key] = cpp_original, python_deployed
                        cpp_original, python_deployed = reused[key]
                        original_exact = np.array_equal(cpp_original, arrays[selector+'_python_events'])
                        deployed_exact = np.array_equal(python_deployed, arrays[selector+'_native_events'])
                        results.append({'selector': selector, 'method': row['method'],
                            'same_original_input_decoder_events_exact': original_exact,
                            'same_deployed_input_decoder_events_exact': deployed_exact,
                            'attribution': 'input_numerical_sensitivity' if original_exact and deployed_exact
                                else 'same_input_decoder_divergence_requires_investigation'})
                        evidence[selector+'_original_input_native_events'] = cpp_original
                        evidence[selector+'_deployed_input_python_events'] = python_deployed
                destination.parent.mkdir(parents=True, exist_ok=True)
                arrays_path = destination.with_suffix('.npz')
                with arrays_path.open('xb') as stream:
                    np.savez_compressed(stream, **evidence)
                value = {'schema': 'mir.heldout-decoder-attribution/v2', 'identity': identity,
                    'model_id': record['model_id'], 'track_uid': record['track_uid'], 'results': results,
                    'arrays': {'file': arrays_path.name, 'sha256': digest(arrays_path)}}
                write_new(destination, value)
            seen.add(path)
            checked += len(value['results'])
            failures += sum(row['attribution'] != 'input_numerical_sensitivity' for row in value['results'])
            print(json.dumps({'pairs_scanned': len(seen), 'differing_streams_checked': checked,
                              'same_input_failures': failures}), flush=True)
        if len(seen) == len(expected) or not args.watch:
            break
        time.sleep(5)
    write_new(args.output/'completion.json', {'complete': len(seen) == len(expected),
        'pairs_scanned': len(seen), 'expected': len(expected), 'differing_streams_checked': checked,
        'same_input_failures': failures, 'decoder_library_sha256': library_hash})


if __name__ == '__main__':
    main()
