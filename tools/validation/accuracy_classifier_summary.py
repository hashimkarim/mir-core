"""Summarize frozen causal classifier decisions without selecting a model.

Recording-weighted metrics give each complete recording equal weight. Pooled
window metrics are also retained, with their different duration weighting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from accuracy_inventory import digest, write_new

LABELS = ['brid', 'candombe', 'salsa', 'other']
ROUTES = ['brid', 'candombe', 'salsa', 'latin_general', 'stock']
DIFFERENCES = ['routing_differences', 'top_label_differences', 'timestamp_differences',
               'probability_violations', 'same_input_router_state_differences']


def metrics(matrix):
    values = np.asarray(matrix, np.float64)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    denom = support+predicted
    f1 = np.divide(2*np.diag(values), denom, out=np.zeros(len(values)), where=denom > 0)
    return {'accuracy': float(np.trace(values)/values.sum()), 'macro_f1': float(f1.mean()),
            'class_f1': f1.tolist(), 'confusion_matrix': values.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text())
    expected = {(family, track['fold'], track['uid']) for family in ('efficientat', 'yamnet')
                for track in sample['tracks']}
    groups, seen, records, evidence = defaultdict(list), set(), [], {}
    for path in sorted(args.replay.glob('*/*.json')):
        if path.name.endswith(('.input.json', '.python.json')) or path.name == 'policy.json':
            continue
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if row.get('schema') != 'mir.heldout-causal-classifier-score/v1':
            continue
        key = row['family'], row['fold'], row['track_uid']
        assert row['complete'] and key in expected and key not in seen
        assert row['identity']['sample_sha256'] == digest(args.sample)
        seen.add(key)
        evidence[str(path.relative_to(args.replay))] = digest(path)
        for item in row['evidence'].values():
            assert digest(path.parent/item['file']) == item['sha256']
        actual = [json.loads(line) for line in (path.parent/row['evidence']['native']['file']).read_text().splitlines()]
        done = actual.pop()
        assert done['complete'] and done['decisions'] == row['comparisons']['windows']
        original = json.loads((path.parent/row['evidence']['python']['file']).read_text())
        decisions = {'python': original, 'native': [value['decision'] for value in actual]}
        item = {key: row[key] for key in ('family', 'fold', 'track_uid', 'dataset_id', 'label', 'true_route')}
        item['comparisons'] = row['comparisons']
        item['audio_seconds'] = row['samples']/row['source_rate']
        item['first_decision_available_seconds'] = decisions['native'][0]['availability_seconds']
        item['last_decision_available_seconds'] = decisions['native'][-1]['availability_seconds']
        item['matrices'] = {}
        for backend, values in decisions.items():
            assert len(values) == row['comparisons']['windows'] > 0
            matrices = {}
            for kind, labels, true, predicted in [('label', LABELS, row['label'], 'top_label'),
                                                  ('route', ROUTES, row['true_route'], 'routed_label')]:
                matrix = np.zeros((len(labels), len(labels)), np.int64)
                for decision in values:
                    matrix[labels.index(true), labels.index(decision['routing'][predicted])] += 1
                matrices[kind] = matrix.tolist()
                count_key = backend+('_route_correct' if kind == 'route' else '_correct')
                assert int(np.trace(matrix)) == row['comparisons'][count_key]
            item['matrices'][backend] = matrices
        records.append(item)
        groups[row['family']+'/all'].append(item)
        groups[row['family']+'/fold_'+str(row['fold'])].append(item)
    missing = sorted(expected-seen)
    if missing and not args.allow_partial:
        raise RuntimeError(f'{len(missing)} classifier recording pairs remain')
    summaries = []
    for name, values in sorted(groups.items()):
        counts = {key: sum(row['comparisons'][key] for row in values) for key in ['windows']+DIFFERENCES}
        result = {'id': name, 'recordings': len(values), 'audio_hours': sum(row['audio_seconds'] for row in values)/3600,
            'comparisons': counts, 'max_probability_error': max(row['comparisons']['max_probability_error'] for row in values),
            'port_gate_pass': all(counts[key] == 0 for key in DIFFERENCES), 'metrics': {}}
        result['first_decision_available_seconds_range'] = [
            min(row['first_decision_available_seconds'] for row in values),
            max(row['first_decision_available_seconds'] for row in values)]
        for backend in ('python', 'native'):
            result['metrics'][backend] = {}
            for kind in ('label', 'route'):
                matrices = [np.asarray(row['matrices'][backend][kind], np.float64) for row in values]
                result['metrics'][backend][kind] = {
                    'pooled_windows': metrics(sum(matrices)),
                    'equal_recordings': metrics(sum(matrix/matrix.sum() for matrix in matrices))}
        summaries.append(result)
    report = {'schema': 'mir.causal-classifier-accuracy-summary/v1', 'complete': not missing,
        'sample_sha256': digest(args.sample), 'runner_sha256': digest(Path(__file__)),
        'expected_pairs': len(expected), 'completed_pairs': len(seen), 'missing_pairs': missing,
        'label_order': LABELS, 'route_order': ROUTES, 'rows_are_truth_columns_are_prediction': True,
        'scope': sample['scope'], 'accuracy_threshold': None,
        'interpretation': 'Descriptive accuracy on the prespecified stratified sample; no deployment-mixture estimate or post-hoc accuracy pass threshold.',
        'decision_coverage': 'All complete source recordings are fed in chunks, without EOF padding. Scores cover emitted decisions after the initial feature window; '
                             'availability metadata expresses algorithmic sample requirements, not measured device execution latency.',
        'summaries': summaries, 'recordings': records, 'evidence_sha256': evidence}
    write_new(args.output, report)
    print(json.dumps({'complete': report['complete'], 'pairs': len(seen), 'summaries': [
        {'id': row['id'], 'recordings': row['recordings'], 'port_gate_pass': row['port_gate_pass'],
         'native_label_macro_f1_equal_recordings': row['metrics']['native']['label']['equal_recordings']['macro_f1'],
         'native_route_accuracy_equal_recordings': row['metrics']['native']['route']['equal_recordings']['accuracy']}
        for row in summaries]}, indent=2))


if __name__ == '__main__':
    main()
