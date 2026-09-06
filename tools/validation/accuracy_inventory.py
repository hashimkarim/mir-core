"""Freeze available checkpoints and their original evaluation memberships.

This inventory does not select a model or decoder from test scores. Shared
cross-fold tuning and unknown upstream training membership remain explicit.
Audio contents are verified by the evaluation worker before use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from beatlab.split_contract import build_split_plan_contract
from mir_desktop_app.experiment import load_heldout_catalog
from mir_desktop_app.pipeline import FoldContext
from splitplan.adapter import (
    CANONICAL_DATASET_IDS, CANONICAL_TARGET_PROFILES,
    build_canonical_split_artifact, load_canonical_corpus,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def decoder_selection_contract(workspace: Path, model: dict, selector: str) -> dict:
    """Use provenance, rather than the display name 'tuned', to classify selection."""
    decoder = model['postprocessors'][selector]
    if decoder['kind'] == 'stock':
        return {'scope': 'untuned-stock'}
    relative = Path('mir-core/mir_core/checkpoints/trained')/model['bundle_id']/\
        'postprocessors'/selector/'source-manifest.json'
    source = json.loads((workspace/relative).read_text())
    policy = source['stage_config']['selection_policy']
    assert source['stage_config']['test_used_for_selection'] is False
    scope = {'fixed': 'fixed-parameters-no-score-selection',
             'sweep_selection': 'shared-validation-sweep-not-nested-heldout'}.get(policy, 'selection-audit-required')
    return {'scope': scope, 'selection_policy': policy, 'source_manifest': str(relative),
            'source_manifest_sha256': digest(workspace/relative)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.processed_root.resolve(), args.output.resolve()
    if (out/'inventory.json').exists():
        parser.error('Use a fresh inventory output directory')
    artifact = build_canonical_split_artifact(root, seed=42, n_folds=5, validation_fraction=.1)
    contract = build_split_plan_contract(artifact.plan, dataset_hashes=artifact.dataset_hashes,
                                        target_profiles=CANONICAL_TARGET_PROFILES)
    corpus = load_canonical_corpus(root)
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    bundles = {}
    package = args.workspace/'mir-core/mir_core/checkpoints'
    for path in sorted((package/'trained').glob('**/manifest.json')):
        bundle = json.loads(path.read_text())
        assert bundle['split_contract']['plan_membership_hash'] == contract['plan_membership_hash'], path
        assert bundle['source']['test_used_for_checkpoint_selection'] is False, path
        for checkpoint in bundle['checkpoints']:
            assert checkpoint['selection_split'] == 'validation'
            assert digest(path.parent/checkpoint['path']) == checkpoint['sha256']
        bundles[bundle['bundle_id']] = (path, bundle)
    write_new(out/'split-artifact.json', artifact.to_dict())
    write_new(out/'split-contract.json', contract)
    catalogs = []
    seen = set()
    for fold_index in range(5):
        tracks = load_heldout_catalog(FoldContext(root, fold_index))
        uids = {t.uid for t in tracks}
        assert not (seen & uids)
        excluded = {uid for ds in CANONICAL_TARGET_PROFILES.values() for dataset in ds
                    for role in ('train', 'validation')
                    for uid in artifact.plan.uids(dataset, fold_index, role)}
        assert not (uids & excluded)
        rows = []
        for track in tracks:
            row = asdict(track)
            for key in ('audio_path', 'annotation_path'):
                row[key] = str(row[key])
            assert digest(Path(row['annotation_path'])) == track.annotation_hash
            rows.append(row)
        value = {'schema': 'mir.heldout-port-catalog/v1', 'fold': fold_index,
                 'split_membership_hash': contract['plan_membership_hash'],
                 'test_uids': sorted(uids), 'excluded_uids': sorted(excluded), 'tracks': rows}
        write_new(out/f'fold-{fold_index}.json', value)
        catalogs.append(value)
        seen.update(uids)
    jobs = []
    for model in catalog['beat_models']:
        fold = model.get('fold')
        if model['id'].startswith('beatnet/stock/'):
            checkpoint = package/'beatnet'/dict(baseline='model_1_weights.pt',
                baseline_alt0='baseline_alt0.pt', baseline_alt1='baseline_alt1.pt')[model['condition']]
            dataset_ids = {'brid', 'candombe', 'salsaset_ft'}
            eligible = catalogs
            scope = 'benchmark_generalization; upstream_training_membership_not_certified'
        else:
            path, bundle = bundles[model['bundle_id']]
            checkpoint = path.parent/next(x['path'] for x in bundle['checkpoints'] if x['fold_index'] == fold)
            dataset_ids = ({'brid', 'candombe', 'salsaset_ft'} if model['target'] == 'latin_general'
                           else set(CANONICAL_TARGET_PROFILES[model['target']]))
            eligible = [catalogs[fold]]
            scope = 'original_checkpoint_test_fold; stock_decoders_are_untuned'
        assert digest(checkpoint) == model['source_sha256']
        graph = catalog_path.parent.parent/model['asset']
        assert digest(graph) == model['asset_sha256']
        for fold_catalog in eligible:
            for row in fold_catalog['tracks']:
                if row['dataset_id'] not in dataset_ids:
                    continue
                jobs.append({'model_id': model['id'], 'fold': fold_catalog['fold'],
                    'track_uid': row['uid'], 'checkpoint': str(checkpoint),
                    'checkpoint_sha256': model['source_sha256'], 'android_graph': str(graph),
                    'android_graph_sha256': model['asset_sha256'],
                    'postprocessors': sorted(model['postprocessors']), 'scope': scope})
    # Classifier membership comes from all eight canonical source manifests.
    # Availability is counted here; contents are hashed before any actual score.
    classifier_rows = []
    for manifest in corpus.manifests:
        value = json.loads(manifest.path.read_text())
        ext = value['audio']['ext']
        for row in value['tracks']:
            uid = f"{manifest.dataset_id}:{row['track_id']}"
            folds = [fold for fold in range(5) if uid in artifact.plan.uids(manifest.dataset_id, fold, 'test')]
            assert len(folds) == 1, uid
            path = root/manifest.dataset_id/'audio'/(row['track_id']+ext)
            classifier_rows.append({'uid': uid, 'dataset_id': manifest.dataset_id,
                'fold': folds[0], 'audio_path': str(path), 'available': path.is_file(),
                'audio_hash': row['audio_hash'], 'annotation_hash': row['annotation_hash'],
                'duration_seconds': row['audio']['duration_seconds']})
    write_new(out/'classifier-corpus.json', classifier_rows)
    inventory = {'schema': 'mir.heldout-port-inventory/v1',
        'catalog_sha256': digest(catalog_path), 'split_contract': contract,
        'beat_tracks': len(seen), 'beat_tracks_per_fold': [len(c['tracks']) for c in catalogs],
        'beat_models': len(catalog['beat_models']), 'beat_jobs': jobs,
        'classifier_models': len(catalog['classifiers']),
        'classifier_audio': {ds: {'tracks': sum(r['dataset_id']==ds for r in classifier_rows),
            'available': sum(r['dataset_id']==ds and r['available'] for r in classifier_rows)}
            for ds in CANONICAL_DATASET_IDS},
        'selection_rule': 'No selection or tuning using any evaluation score; all shipped decoders reported.',
        'tuned_decoder_scope': 'Shared parameter files cover all five folds. Strict fold-local tuning isolation requires auditing original sweep selection membership.',
        'physical_scope': 'ESP32-only per user; phone excluded; sensor onset requires external evidence.',
        'other_architectures': 'Separate pretrained-baseline inventory is required; initialized architecture probes are not musical-accuracy results.'}
    write_new(out/'inventory.json', inventory)
    print(json.dumps({k:v for k,v in inventory.items() if k in ['beat_tracks','beat_tracks_per_fold','beat_models','classifier_models','classifier_audio']}), flush=True)
    print(json.dumps({'beat_jobs':len(jobs), 'per_model_family':dict(Counter(j['model_id'].rsplit('/fold_',1)[0] for j in jobs))}), flush=True)


if __name__ == '__main__':
    main()
