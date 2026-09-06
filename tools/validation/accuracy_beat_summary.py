"""Summarize all frozen beat scores, with an explicit corrected 1D overlay.

No model or decoder is chosen using these scores. Conventional F1 is descriptive;
numeric parity and identical-input decoder comparisons remain separate gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from accuracy_inventory import digest, write_new


def read_records(root, schema):
    for path in sorted(root.glob('*/*.json')):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if row.get('schema') == schema:
            yield path, row


def paired(values, name):
    original = np.asarray([row['python_metrics'][name] for row in values if name in row['python_metrics']])
    native = np.asarray([row['native_metrics'][name] for row in values if name in row['python_metrics']])
    if not len(original):
        return None
    delta = native-original
    assert np.isfinite(original).all() and np.isfinite(native).all()
    seed = int.from_bytes(hashlib.sha256((name+'\0'+','.join(sorted(row['track_uid'] for row in values))).encode()).digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    means = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
    return {'recordings': len(delta), 'python_mean': float(original.mean()), 'native_mean': float(native.mean()),
        'mean_delta': float(delta.mean()), 'mean_absolute_delta': float(np.abs(delta).mean()),
        'max_absolute_delta': float(np.abs(delta).max()), 'minimum_delta': float(delta.min()),
        'maximum_delta': float(delta.max()), 'improved': int(np.count_nonzero(delta > 0)),
        'worsened': int(np.count_nonzero(delta < 0)), 'equal': int(np.count_nonzero(delta == 0)),
        'paired_recording_bootstrap_95pct': [float(x) for x in np.percentile(means, [2.5, 97.5])]}


def summarize(name, values):
    assert len({row['track_uid'] for row in values}) == len(values)
    return {'id': name, 'recordings': len(values), 'selection_scopes': sorted({row['selection_scope'] for row in values}),
        'exact_event_streams': sum(row['event_stream_exact'] for row in values),
        'same_input_failures': sum(row['same_input_pass'] is False for row in values),
        'different_streams_awaiting_attribution': sum(not row['event_stream_exact'] and row['same_input_pass'] is None for row in values),
        'beat_f1': paired(values, 'fmeasure'), 'downbeat_f1': paired(values, 'db_fmeasure')}


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    root = args.evidence_root
    inventory = json.loads((root/'catalogs/inventory.json').read_text())
    expected = {(row['model_id'], row['track_uid']) for row in inventory['beat_jobs']}
    expected_counts = Counter(row['model_id'] for row in inventory['beat_jobs'])
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    assert digest(catalog_path) == inventory['catalog_sha256']
    catalog = {row['id']: row for row in json.loads(catalog_path.read_text())['beat_models']}
    selection_path = root/'catalogs/decoder-selection-audit-v2.json'
    selection = {(row['model_id'], row['selector']): row['scope'] for row in
                 json.loads(selection_path.read_text())['deployment_contracts']}
    corrected = { (row['model_id'], row['track_uid']): (path, row) for path, row in
                 read_records(root/'beat-1d-correction-v2', 'mir.heldout-1d-decoder-correction/v1')}
    attributed = { (row['model_id'], row['track_uid']): (path, row) for path, row in
                  read_records(root/'decoder-attribution-v2', 'mir.heldout-decoder-attribution/v2')}
    records, rows, bindings, seen = [], [], {}, set()
    overlay_pending, attribution_pending = [], []
    for path, record in read_records(root/'beat-replay-v1', 'mir.heldout-port-score/v1'):
        key = record['model_id'], record['track_uid']
        assert record['complete'] and key in expected and key not in seen
        seen.add(key)
        assert digest(path.parent/record['arrays']['file']) == record['arrays']['sha256']
        bindings[str(path.relative_to(root))] = digest(path)
        record_info = {name: record[name] for name in ('model_id', 'track_uid', 'fold', 'frames', 'duration_seconds',
            'max_activation_error', 'activation_violations', 'android_graph_host_activations_bit_exact', 'scope')}
        records.append(record_info)
        overlays, diagnostics = {}, {}
        if key in corrected:
            correction_path, correction = corrected[key]
            assert correction['identity']['original_score_sha256'] == digest(path)
            assert digest(correction_path.parent/correction['arrays']['file']) == correction['arrays']['sha256']
            bindings[str(correction_path.relative_to(root))] = digest(correction_path)
            overlays = {row['selector']: row for row in correction['decoders']}
        else:
            overlay_pending.append(key)
        if key in attributed:
            diagnostic_path, diagnostic = attributed[key]
            assert diagnostic['identity']['baseline_score_sha256'] == digest(path)
            assert digest(diagnostic_path.parent/diagnostic['arrays']['file']) == diagnostic['arrays']['sha256']
            bindings[str(diagnostic_path.relative_to(root))] = digest(diagnostic_path)
            diagnostics = {row['selector']: row for row in diagnostic['results']}
        else:
            attribution_pending.append(key)
        for original in record['decoders']:
            if original['method'] == 'heydari_1d_state_space':
                if original['selector'] not in overlays:
                    continue
                decoder = overlays[original['selector']]
                same_input = decoder['same_original_input_decoder_events_exact'] and decoder['same_deployed_input_decoder_events_exact']
            else:
                decoder = original
                diagnostic = diagnostics.get(original['selector'])
                same_input = (diagnostic['same_original_input_decoder_events_exact'] and diagnostic['same_deployed_input_decoder_events_exact']) if diagnostic else None
            rows.append(dict(decoder, **record_info, selection_scope=selection[(record['model_id'], decoder['selector'])],
                             same_input_pass=same_input))
    missing = sorted(expected-seen)
    complete = not (missing or overlay_pending or attribution_pending)
    if not complete and not args.allow_partial:
        raise RuntimeError(f'Pending: {len(missing)} inference pairs, {len(overlay_pending)} 1D overlays, '
                           f'{len(attribution_pending)} decoder attribution records')
    per_model, per_family, per_dataset = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        model = row['model_id']
        family = model.rsplit('/fold_', 1)[0]
        selector = row['selector']
        per_model[(model, selector)].append(row)
        per_family[(family, selector)].append(row)
        per_dataset[(family, selector, row['track_uid'].split(':')[0])].append(row)
    model_selectors = [summarize('/'.join(key), values) for key, values in sorted(per_model.items())]
    family_selectors = [summarize('/'.join(key), values) for key, values in sorted(per_family.items())]
    dataset_selectors = [summarize('/'.join(key), values) for key, values in sorted(per_dataset.items())]
    models = []
    for model_id, model in sorted(catalog.items()):
        completed = [row for row in records if row['model_id'] == model_id]
        default = model['default_postprocessor']
        default_scores = per_model[(model_id, default)]
        models.append({'model_id': model_id, 'completed': len(completed), 'expected': expected_counts[model_id],
            'frames': sum(row['frames'] for row in completed), 'default_selector': default,
            'original_f32_violating_values': sum(row['activation_violations'] for row in completed),
            'original_f32_violating_recordings': sum(row['activation_violations'] > 0 for row in completed),
            'max_activation_error': max((row['max_activation_error'] for row in completed), default=None),
            'android_host_bit_exact_recordings': sum(row['android_graph_host_activations_bit_exact'] for row in completed),
            'default_scores': summarize(model_id+'/'+default, default_scores) if default_scores else None})
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'summary.json', {'schema': 'mir.full-recording-beat-accuracy-summary/v1',
        'complete': complete, 'expected_pairs': len(expected), 'completed_pairs': len(seen),
        'missing_pairs': missing, 'pending_1d_corrections': overlay_pending, 'pending_attribution': attribution_pending,
        'frames': sum(row['frames'] for row in records), 'decoder_recording_comparisons': len(rows),
        'model_decoder_combinations': len(model_selectors),
        'activation_tolerance': {'rtol': 2e-5, 'atol': 2e-6},
        'original_f32_violating_values': sum(row['activation_violations'] for row in records),
        'original_f32_violating_recordings': sum(row['activation_violations'] > 0 for row in records),
        'android_host_nonexact_recordings': sum(not row['android_graph_host_activations_bit_exact'] for row in records),
        'same_input_decoder_failures': sum(row['same_input_pass'] is False for row in rows),
        'reference_rng': 'portable-splitmix64-v1; seed 42 independently reset for every recording and decoder',
        'metric_contract': 'Conventional whole-recording beat/downbeat F1 at 70 ms; recordings have equal weight.',
        'accuracy_threshold': None,
        'bootstrap_contract': '2000 paired whole-recording resamples using identity-seeded PCG64; descriptive intervals, no model selection.',
        'feature_scope_erratum': 'Frozen feature v1 uses shared whole-recording SoXR HQ converted PCM. '
                                 'The frontend and inference are causal on that PCM; raw-source streaming resampler latency is not measured.',
        'android_scope': 'Shipped Android ONNX assets executed by host ORT, not an Android device timing run.',
        'selector_scope_erratum': 'Manifest-bound selection scopes override baseline v1 display-name classification; fixed PF is not a sweep.',
        'runner_sha256': digest(Path(__file__)), 'selection_audit_sha256': digest(selection_path),
        'models': models, 'model_selectors': model_selectors, 'family_selectors': family_selectors,
        'dataset_selectors': dataset_selectors, 'records': records, 'evidence_sha256': bindings})
    flat_rows = []
    for row in rows:
        flat = {key: row[key] for key in ('model_id', 'track_uid', 'fold', 'selector', 'method', 'selection_scope',
                                         'frames', 'event_stream_exact', 'same_input_pass')}
        flat['one_d_stream_changed_by_fix'] = row.get('changed_by_fix')
        for name in ('fmeasure', 'db_fmeasure'):
            for backend in ('python', 'native'):
                flat[backend+'_'+name] = row[backend+'_metrics'].get(name)
            flat['delta_'+name] = row['metric_deltas'].get(name)
        flat_rows.append(flat)
    if flat_rows:
        write_csv(args.output/'recording-decoder-scores.csv', flat_rows)
    flat_groups = []
    for row in model_selectors:
        flat = {key: row[key] for key in ('id', 'recordings', 'exact_event_streams', 'same_input_failures')}
        flat['selection_scope'] = ','.join(row['selection_scopes'])
        for name in ('beat_f1', 'downbeat_f1'):
            for metric in ('python_mean', 'native_mean', 'mean_delta', 'max_absolute_delta'):
                flat[name+'_'+metric] = row[name][metric] if row[name] else None
        flat_groups.append(flat)
    if flat_groups:
        write_csv(args.output/'model-decoder-scores.csv', flat_groups)
    print(json.dumps({'complete': complete, 'pairs': len(seen), 'frames': sum(row['frames'] for row in records),
        'same_input_failures': sum(row['same_input_pass'] is False for row in rows),
        'original_f32_violating_values': sum(row['activation_violations'] for row in records),
        'model_decoder_combinations': len(model_selectors)}))


if __name__ == '__main__':
    main()
