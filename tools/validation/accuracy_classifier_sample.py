"""Freeze a stratified causal-music screen before observing classifier scores.

Each source-dataset/label/test-fold cell contributes at most five complete
recordings, ranked by a fixed SHA-256 rule. This is a stratified screen, not an
estimate weighted to the original corpus or the deployment duration mixture.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from accuracy_inventory import digest, write_new
from splitplan.adapter import load_canonical_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus = load_canonical_corpus(args.processed_root)
    frozen = json.loads((args.catalogs/'split-artifact.json').read_text())
    for manifest in corpus.manifests:
        assert digest(manifest.path) == frozen['source_manifests'][manifest.dataset_id]['manifest_sha256']
    candidates = json.loads((args.catalogs/'classifier-corpus.json').read_text())
    by_uid = corpus.track_by_uid
    groups = defaultdict(list)
    excluded = []
    for row in candidates:
        source = by_uid[row['uid']]
        label = dict(source.record.strata)['router_label']
        if label == 'exclude':
            excluded.append(row['uid'])
            continue
        assert label in {'brid', 'candombe', 'salsa', 'other'}
        row = dict(row, label=label)
        groups[(row['fold'], row['dataset_id'], label)].append(row)
    rows, strata = [], []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda item: hashlib.sha256(('mir-causal-accuracy-screen-v1\0'+item['uid']).encode()).hexdigest())
        selected = items[:5]
        assert all(row['available'] for row in selected)
        rows.extend(selected)
        strata.append({'fold': key[0], 'dataset': key[1], 'label': key[2],
                       'eligible': len(items), 'selected': len(selected)})
    plan = {'schema': 'mir.classifier-causal-accuracy-sample/v1',
            'selection_rule': 'first five SHA256(mir-causal-accuracy-screen-v1 NUL uid) per fold/dataset/label',
            'selection_uses_scores': False, 'whole_recordings': True,
            'scope': 'stratified held-out screen; not full-corpus or deployment-mixture accuracy',
            'catalog_sha256': digest(args.catalogs/'classifier-corpus.json'),
            'split_artifact_sha256': digest(args.catalogs/'split-artifact.json'),
            'excluded_by_original_label': sorted(excluded), 'strata': strata, 'tracks': rows}
    write_new(args.output, plan)
    print(json.dumps({'tracks': len(rows), 'hours': sum(row['duration_seconds'] for row in rows)/3600,
                      'labels': dict(Counter(row['label'] for row in rows)), 'sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
