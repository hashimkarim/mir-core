"""Attribute differing beat events without changing the frozen accuracy run.

Run both independent decoders on each saved activation stream. This separates
inference sensitivity from a decoder translation error on identical inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from accuracy_inventory import digest, write_new
from accuracy_replay import decode, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    catalog = {row['id']: row for row in json.loads(catalog_path.read_text())['beat_models']}
    for path in sorted(args.replay.glob('*/*.json')):
        record = json.loads(path.read_text())
        if record.get('schema') != 'mir.heldout-port-score/v1':
            continue
        selectors = [row for row in record['decoders'] if not row['event_stream_exact']]
        if not selectors:
            continue
        model = catalog[record['model_id']]
        assert record['identity']['postprocessors_sha256'] == stable_hash(model['postprocessors'])
        destination = args.output/path.relative_to(args.replay)
        identity = {'score_sha256': digest(path), 'runner_sha256': digest(Path(__file__)),
                    'arrays_sha256': record['arrays']['sha256']}
        if destination.exists():
            assert json.loads(destination.read_text())['identity'] == identity
            continue
        array_path = path.parent/record['arrays']['file']
        assert digest(array_path) == record['arrays']['sha256']
        results = []
        reused = {}
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
                results.append({'selector': selector,
                    'same_original_input_decoder_events_exact': original_exact,
                    'same_deployed_input_decoder_events_exact': deployed_exact,
                    'attribution': 'input_numerical_sensitivity' if original_exact and deployed_exact
                                   else 'same_input_decoder_divergence_requires_investigation',
                    'original_input_cpp_events': len(cpp_original),
                    'deployed_input_python_events': len(python_deployed)})
        value = {'schema': 'mir.heldout-decoder-attribution/v1', 'identity': identity,
                 'model_id': record['model_id'], 'track_uid': record['track_uid'], 'results': results}
        write_new(destination, value)
        print(json.dumps({key: value[key] for key in ('model_id', 'track_uid', 'results')}), flush=True)


if __name__ == '__main__':
    main()
