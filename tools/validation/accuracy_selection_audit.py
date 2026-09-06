"""Audit the recorded decoder selection code and original fold memberships.

Potential overlap is reported without asserting that every available identity
was actually retained in the historical sweep. This never reselects parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from accuracy_inventory import decoder_selection_contract, digest, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--catalogs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace
    inventory = json.loads((args.catalogs/'inventory.json').read_text())
    plan_path = args.catalogs/'split-artifact.json'
    plan = json.loads(plan_path.read_text())['split_plan']
    catalog_path = root/'mir-android-app/app/src/main/assets/models/catalog.json'
    assert digest(catalog_path) == inventory['catalog_sha256']
    catalog = json.loads(catalog_path.read_text())
    sources, selections, contracts = {}, {}, []
    for model in catalog['beat_models']:
        for selector in model['postprocessors']:
            contract = decoder_selection_contract(root, model, selector)
            contracts.append({'model_id': model['id'], 'selector': selector, **contract})
            if 'source_manifest' not in contract or contract['source_manifest'] in selections:
                continue
            source = json.loads((root/contract['source_manifest']).read_text())
            policy = source['stage_config']['selection_policy']
            value = {'manifest': contract['source_manifest'], 'sha256': contract['source_manifest_sha256'],
                'recorded_source_commit': source['git_commit'], 'postprocessor': source['postprocessor_id'],
                'selection_hash': source['selection_hash'], 'selection_policy': policy,
                'scope': contract['scope'], 'dataset_ids': source['dataset_ids']}
            if policy == 'sweep_selection':
                commit = source['git_commit']
                if commit not in sources:
                    file = 'beatlab/cross_fold_sweep.py'
                    code = subprocess.check_output(['git', 'show', commit+':'+file], cwd=root/'mir-train-hpc')
                    lines = code.decode().splitlines()
                    assert 'Select one decoder parameter set from rotating validation folds.' in lines[0]
                    sources[commit] = {'file': file, 'sha256': hashlib.sha256(code).hexdigest(),
                        'description': lines[:7],
                        'pooled_evidence': [{'line': i+1, 'text': line.strip()} for i, line in enumerate(lines)
                            if 'pooled cross-fold histogram' in line or 'source_split' in line and
                               'rotating_out_of_training_validation' in line]}
                    assert sources[commit]['pooled_evidence']
                datasets = source['dataset_ids']
                validation = {uid for ds in datasets for fold in plan['datasets'][ds]['folds']
                              for uid in fold['roles']['validation']}
                value['cross_fold_validation_union_count'] = len(validation)
                value['potential_overlap_by_test_fold'] = []
                for fold in range(5):
                    test = {uid for ds in datasets for uid in plan['datasets'][ds]['folds'][fold]['roles']['test']}
                    overlap = sorted(test & validation)
                    value['potential_overlap_by_test_fold'].append({'fold': fold, 'test_count': len(test),
                        'also_in_other_folds_validation_count': len(overlap),
                        'also_in_other_folds_validation': overlap})
            else:
                assert policy == 'fixed' and source['stage_config']['source_parameter_file'] is None
            selections[contract['source_manifest']] = value
    report = {'schema': 'mir.decoder-selection-isolation-audit/v2',
        'catalog_sha256': digest(catalog_path), 'split_artifact_sha256': digest(plan_path),
        'runner_sha256': digest(Path(__file__)), 'historical_sources': sources,
        'selection_counts': dict(Counter(row['selection_policy'] for row in selections.values())),
        'selections': list(selections.values()), 'deployment_contracts': contracts,
        'conclusion': 'Swept DBN and 1D parameters pool rotating validation folds and are not nested held-out estimates. '
                      'The particle-filter presets named tuned use fixed parameters, so the sweep caveat does not apply to them.',
        'scope': 'Checkpoint fold isolation is preserved. Potential selection overlap uses canonical identities; '
                 'exact retained historical candidate-track usage is not fully available locally.',
        'baseline_erratum': 'The frozen accuracy replay v1 selection_scope field classifies every tuned display name '
                            'as shared-fold-tuned. Use these manifest-bound contracts instead; no numeric results changed.'}
    write_new(args.output, report)
    print(json.dumps({'sources': len(sources), 'selection_counts': report['selection_counts'],
                      'deployment_contracts': len(contracts)}))


if __name__ == '__main__':
    main()
