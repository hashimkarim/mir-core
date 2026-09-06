"""Score complete held-out recordings with original and deployed causal routers.

Only frozen validation-selected heads/policies are used. Source audio is fed
incrementally at its original rate; no whole-recording normalization is added.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import soundfile as sf
import torch

from accuracy_inventory import digest, write_new
from classifierlab.causal_streaming import build_causal_feature_frontend
from classifierlab.live_runtime import LiveCausalFeatureFrontend, LiveClassifierRouter
from classifierlab.native_export import export_native_classifier_checkpoint
from classifierlab.router_evaluation import load_classifier_checkpoint
from classifierlab.window_routing import RuntimeRouterState, runtime_policy_from_dict
from mir_core.checkpoints import load_trained_model_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['efficientat', 'yamnet'], required=True)
    parser.add_argument('--limit-tracks', type=int)
    parser.add_argument('--fold', type=int, choices=range(5), action='append')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.workspace
    plan = json.loads(args.sample.read_text())
    assert plan['selection_uses_scores'] is False and plan['whole_recordings']
    binary = root/'mir-desktop-app/native-runtime/target/debug/examples/classifier_accuracy_replay'
    trained = load_trained_model_bundle('classifier', 'latin_router', args.family)
    for fold in (args.fold or range(5)):
        case = args.output/f'{args.family}-fold-{fold}'
        case.mkdir(parents=True, exist_ok=True)
        checkpoint = trained.checkpoint_path(fold)
        bundle = load_classifier_checkpoint(checkpoint, device='cpu')
        exported = export_native_classifier_checkpoint(checkpoint, cache_root=args.output/'artifacts',
            include_audio_frontend=True, efficientat_source_root=root/'EfficientAT')
        policy_value = bundle.router_config['runtime_policy']
        assert policy_value['selection']['split_role'] == 'validation'
        policy = runtime_policy_from_dict(policy_value)
        policy_path = case/'policy.json'
        if not policy_path.exists():
            write_new(policy_path, policy_value)
        assert json.loads(policy_path.read_text()) == policy_value
        cfg = {'kind': args.family, 'artifact': exported['audio_frontend']['manifest_path']}
        if args.family == 'efficientat':
            cfg['feature_config'] = bundle.feature_config
        else:
            cfg['backend_library'] = str(root/'mir-embedded-ai/native-litert/build/libmir_litert.so')
        classifier = {'artifact': exported['manifest_path'], 'policy': str(policy_path),
            'policy_sha256': digest(policy_path), 'frontend': cfg,
            'resampler_library': str(root/'mir-core/native-dsp/build/libmir_dsp.so')}
        reference_config = dict(bundle.feature_config)
        if args.family == 'efficientat':
            reference_config['repo_path'] = str(root/'EfficientAT')
        source_frontend = build_causal_feature_frontend(reference_config, device=torch.device('cpu'),
                                                       fast_forward_batch_size=1)
        jobs = [row for row in plan['tracks'] if row['fold'] == fold]
        if args.limit_tracks is not None:
            jobs = jobs[:args.limit_tracks]
        for track in jobs:
            stem = track['uid'].replace(':', '__').replace('/', '__')
            result_path = case/(stem+'.json')
            identity = {'sample_sha256': digest(args.sample), 'checkpoint_sha256': digest(checkpoint),
                'policy_sha256': digest(policy_path), 'audio_sha256': track['audio_hash'],
                'runner_sha256': digest(Path(__file__)), 'native_binary_sha256': digest(binary),
                'classifier_manifest_sha256': digest(Path(exported['manifest_path'])),
                'frontend_manifest_sha256': digest(Path(exported['audio_frontend']['manifest_path']))}
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                assert previous['identity'] == identity and previous['complete']
                for item in previous['evidence'].values():
                    assert digest(case/item['file']) == item['sha256']
                continue
            started = time.monotonic()
            audio_path = Path(track['audio_path'])
            assert digest(audio_path) == track['audio_hash']
            audio, source_rate = sf.read(audio_path, dtype='float32', always_2d=True)
            audio = audio.mean(axis=1, dtype=np.float32)
            assert np.isfinite(audio).all() and len(audio)
            pcm_path = case/(stem+'.f32')
            with pcm_path.open('xb') as stream:
                stream.write(audio.astype('<f4', copy=False).tobytes())
            replay_path = case/(stem+'.input.json')
            write_new(replay_path, {'source_rate': source_rate, 'classifier': classifier,
                                    'pcm_f32le': str(pcm_path), 'chunk_samples': 4096})
            actual_path = case/(stem+'.native.jsonl')
            with actual_path.open('x') as stream, (case/(stem+'.stderr.log')).open('x') as error:
                subprocess.run([str(binary), str(replay_path)], stdout=stream, stderr=error, check=True)
            actual_rows = [json.loads(line) for line in actual_path.read_text().splitlines()]
            native_summary = actual_rows.pop()
            assert native_summary['complete'] and native_summary['samples'] == len(audio)
            actual = [row['decision'] for row in actual_rows]
            assert native_summary['decisions'] == len(actual)
            live = LiveCausalFeatureFrontend(source_frontend, source_sample_rate=source_rate)
            reference = LiveClassifierRouter(model=bundle.model, labels=bundle.labels, policy=policy,
                                              frontend=live, device=torch.device('cpu'))
            original_infer = reference._infer_logits
            original_logits = []
            def record_logits(values):
                logits = original_infer(values)
                original_logits.append(logits.tolist())
                return logits
            reference._infer_logits = record_logits
            expected = []
            for start in range(0, len(audio), 4096):
                expected.extend(dataclasses.asdict(row) for row in reference.process_audio(audio[start:start+4096]))
            for row, logits in zip(expected, original_logits, strict=True):
                row['logits'] = logits
                for key in ('probabilities', 'ema_probabilities'):
                    row['routing'][key] = np.asarray(row['routing'][key]).tolist()
            expected_path = case/(stem+'.python.json')
            write_new(expected_path, expected)
            assert len(expected) == len(actual) and len(expected) > 0
            comparisons = {'windows': len(expected), 'routing_differences': 0, 'top_label_differences': 0,
                'timestamp_differences': 0, 'probability_violations': 0, 'max_probability_error': 0.0,
                'same_input_router_state_differences': 0,
                'python_correct': 0, 'native_correct': 0, 'python_route_correct': 0, 'native_route_correct': 0}
            true_route = policy_value['parameters']['execution_routes']['classifier_to_route'][track['label']]
            same_input_router = RuntimeRouterState(bundle.labels, policy)
            for got, want in zip(actual, expected, strict=True):
                comparisons['timestamp_differences'] += int(any(got[key] != want[key] for key in (
                    'feature_start_frame', 'feature_end_frame', 'source_start_seconds', 'source_end_seconds', 'availability_seconds')))
                g, w = got['routing'], want['routing']
                same = dataclasses.asdict(same_input_router.update_logits(np.asarray(got['logits'], np.float64)))
                discrete = set(g)-{'confidence', 'probabilities', 'ema_probabilities'}
                comparisons['same_input_router_state_differences'] += int(any(g[key] != same[key] for key in discrete))
                for key in ('confidence', 'probabilities', 'ema_probabilities'):
                    np.testing.assert_allclose(g[key], same[key], rtol=1e-12, atol=1e-15)
                comparisons['routing_differences'] += int(g['routed_label'] != w['routed_label'])
                comparisons['top_label_differences'] += int(g['top_label'] != w['top_label'])
                for key in ('probabilities', 'ema_probabilities'):
                    delta = np.abs(np.asarray(g[key])-np.asarray(w[key]))
                    comparisons['max_probability_error'] = max(comparisons['max_probability_error'], float(delta.max()))
                    comparisons['probability_violations'] += int(np.count_nonzero(delta > 2e-3+5e-4*np.abs(w[key])))
                for name, routing in (('native', g), ('python', w)):
                    comparisons[name+'_correct'] += int(routing['top_label'] == track['label'])
                    comparisons[name+'_route_correct'] += int(routing['routed_label'] == true_route)
            evidence = {name: {'file': path.name, 'sha256': digest(path)} for name, path in
                        [('python', expected_path), ('native', actual_path), ('input', replay_path), ('pcm', pcm_path)]}
            result = {'schema': 'mir.heldout-causal-classifier-score/v1', 'complete': True,
                'family': args.family, 'fold': fold, 'track_uid': track['uid'], 'dataset_id': track['dataset_id'],
                'label': track['label'], 'true_route': true_route, 'identity': identity, 'evidence': evidence,
                'source_rate': source_rate, 'samples': len(audio), 'scope': plan['scope'],
                'comparisons': comparisons, 'wall_seconds': time.monotonic()-started}
            write_new(result_path, result)
            print(json.dumps({key: result[key] for key in ('family', 'fold', 'track_uid', 'comparisons', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
