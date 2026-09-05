"""Validate the versioned deployment computation and preserve the old baseline.

The independent bitwise reference implements the specified operation sequence
in NumPy. Original PyTorch float64/float32 computations remain separate numeric
accuracy checks; the original float32 baseline and its tolerance are unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from check_promoted_streams import (ROOT, NAMES, RTOL, ATOL, np, ort, torch,
    BeatNetCRNN, list_trained_model_bundles, extract_checkpoint_state_dict,
    normalize_state_dict_keys, ensure_streaming_beatnet_onnx,
    digest, features, inputs, read_u32, read_outputs, events)
from mir_core.native.beatnet import EXPORTER_NAME
from precise_reference import NumpyPreciseStep


def exact_comparison(actual, expected):
    return {name:dict(exact=np.array_equal(actual[name].view(np.uint32), expected[name].view(np.uint32)),
                     within_original_tolerance=np.allclose(actual[name], expected[name], rtol=RTOL, atol=ATOL),
                     max_abs=float(np.max(np.abs(actual[name]-expected[name]))),
                     different_elements=int(np.count_nonzero(actual[name].view(np.uint32) != expected[name].view(np.uint32)))) for name in NAMES}


def frame_hashes(values):
    result = bytearray()
    for frame in range(len(values['activations'])):
        h = hashlib.sha256()
        for name in NAMES: h.update(values[name][frame].astype('<f4').tobytes())
        result.extend(h.digest())
    return result


def parallel_run(args):
    """Independent checkpoints run in isolated processes; each graph is single-threaded."""
    out=args.output.resolve()
    if (out/'report.json').exists():
        raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((ROOT/'mir-android-app/app/src/main/assets/models/catalog.json').read_text())
    models=catalog['beat_models'][:args.limit]
    if args.model_id: models=[m for m in models if m['id'] in args.model_id]
    if not models: raise ValueError('No selected model')
    count=min(args.workers,len(models))
    groups=[models[i::count] for i in range(count)]
    def worker(index):
        folder=out/'workers'/str(index);folder.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--workers','1',
            '--output',str(folder),'--cpp-replay',str(args.cpp_replay.resolve()),
            '--postprocessor-library',str(args.postprocessor_library.resolve()),
            '--legacy-run',str(args.legacy_run.resolve()),'--frames',str(args.frames)]
        for model in groups[index]: command.extend(('--model-id',model['id']))
        with (folder/'worker.log').open('w') as stream:
            status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        print(f'worker {index}: exit {status}',flush=True)
        return folder,status
    with ThreadPoolExecutor(max_workers=count) as pool:
        results=list(pool.map(worker,range(count)))
    reports=[json.loads((folder/'report.json').read_text()) for folder,_ in results]
    indexes=[json.loads((folder/'phone/precisePorts/index.json').read_text()) for folder,_ in results]
    report=dict(reports[0]);phone_index=dict(indexes[0])
    for r,index in zip(reports,indexes):
        assert r['features']['sha256']==report['features']['sha256']
        assert r['catalog_sha256']==report['catalog_sha256']
        assert index['catalog_sha256']==phone_index['catalog_sha256']
        assert r['numerical_contract']==report['numerical_contract']
    rows={m['id']:m for r in reports for m in r['models']}
    phone_rows={m['id']:m for index in indexes for m in index['models']}
    assert len(rows)==sum(len(r['models']) for r in reports)==len(models)
    report['models']=[rows[m['id']] for m in models]
    phone_index['models']=[phone_rows[m['id']] for m in models]
    report['all_catalog_models_selected']=len(models)==len(catalog['beat_models'])
    report['passed']=all(status==0 for _,status in results) and all(m['passed'] for m in report['models'])
    report['workers']=count
    phone=out/'phone/precisePorts';phone.mkdir(parents=True)
    for folder,_ in results:
        for source in (folder/'phone/precisePorts').iterdir():
            if source.name!='index.json': shutil.copyfile(source,phone/source.name)
    shutil.copyfile(results[0][0]/'features.f32',out/'features.f32')
    report['features']={**report['features'],'path':str(out/'features.f32')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (phone/'index.json').write_text(json.dumps(phone_index,indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpp-replay', type=Path, required=True)
    p.add_argument('--postprocessor-library', type=Path, required=True)
    p.add_argument('--legacy-run', type=Path, required=True)
    p.add_argument('--frames', type=int, default=4096)
    p.add_argument('--limit', type=int)
    p.add_argument('--model-id', action='append')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    if args.workers < 1: p.error('workers must be positive')
    if args.workers > 1: return parallel_run(args)
    out = args.output.resolve()
    if (out/'report.json').exists():
        p.error('Use a fresh output directory; existing evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    phone = out/'phone/precisePorts'; phone.mkdir(parents=True, exist_ok=True)
    os.environ['MIR_EMBEDDED_PP_HOST_LIBRARY'] = str(args.postprocessor_library.resolve())
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    values, feature_record = features(out, args.frames)
    legacy_report = json.loads((args.legacy_run/'report.json').read_text())
    assert legacy_report['features']['sha256'] == feature_record['sha256']
    legacy_rows = {r['id']:r for r in legacy_report['models']}
    shutil.copyfile(out/'features.f32', phone/'features.f32')
    replay_input = out/'inputs.replay'; inputs(replay_input, values)
    assets = ROOT/'mir-android-app/app/src/main/assets'
    catalog_path = assets/'models/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    bundles = {b.bundle_id:b for b in list_trained_model_bundles(verify_files=True)}
    report = dict(schema='mir.precise-stream-parity/v1', numerical_contract=EXPORTER_NAME,
        reference='independent-numpy-fixed-operation-sequence',
        mathematical_reference='original-pytorch-model-float64-with-float32-state-boundaries',
        legacy_reference='unchanged-original-pytorch-float32',
        features=feature_record, catalog_sha256=digest(catalog_path),
        cpp_replay_sha256=digest(args.cpp_replay),
        gate='bit-exact deployed outputs/states/events against independent NumPy; original tolerance against original PyTorch computations',
        models=[])
    phone_index = dict(schema='mir.precise-phone-golden/v1', numerical_contract=EXPORTER_NAME,
        golden_source='cpp-rust-onnx-deployment; bitwise validated against independent NumPy and numerically against original PyTorch',
        frames=len(values), features_sha256=digest(phone/'features.f32'),
        catalog_sha256=digest(catalog_path), models=[])
    selected=catalog['beat_models'][:args.limit]
    if args.model_id:
        if set(args.model_id)-{m['id'] for m in selected}: p.error('Unknown model id')
        selected=[m for m in selected if m['id'] in args.model_id]
    if not selected: p.error('No selected models')
    report['all_catalog_models_selected']=len(selected)==len(catalog['beat_models'])
    for item in selected:
        row = dict(id=item['id']); started=time.monotonic()
        try:
            old = legacy_rows[item['id']]
            if item['id'].startswith('beatnet/stock/'):
                filename = {'baseline':'model_1_weights.pt', 'baseline_alt0':'baseline_alt0.pt', 'baseline_alt1':'baseline_alt1.pt'}[item['condition']]
                checkpoint = ROOT/'mir-core/mir_core/checkpoints/beatnet'/filename
            else:
                bundle = bundles[item.get('bundle_id') or item['id'].rsplit('/fold_',1)[0]]
                checkpoint = bundle.checkpoint_path(item['fold'])
                assert item['split_contract_hash'] == bundle.split_contract['contract_hash']
            assert digest(checkpoint) == item['source_sha256'] == old['source_sha256']
            assert digest(assets/item['asset']) == item['asset_sha256']
            assert item['numerical_contract'] == EXPORTER_NAME
            config = {k:item[k] for k in ('input_dim','hidden_dim','num_layers')}
            model = BeatNetCRNN(**config).eval()
            state = normalize_state_dict_keys(extract_checkpoint_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)), 'beatnet')
            missing, extra = model.load_state_dict(state, strict=False)
            assert set(missing) <= {'hidden','cell'} and not extra
            artifact = ensure_streaming_beatnet_onnx(model, model_name='beatnet', model_config=config,
                checkpoint_sha256=digest(checkpoint), cache_root=out/'exports')
            assert artifact.manifest['exporter']['name'] == EXPORTER_NAME
            assert all(torch.equal(parameter,state[name]) for name,parameter in model.named_parameters())
            folder = out/item['id'].replace('/','_'); folder.mkdir(exist_ok=True)
            replay_output = folder/'cpp.replay'
            subprocess.run([str(args.cpp_replay.resolve()), str(artifact.manifest_path), str(replay_input), str(replay_output)], check=True, capture_output=True, text=True)
            with replay_output.open('rb') as stream:
                assert read_u32(stream) == len(values)+1
                outputs = [read_outputs(stream) for _ in range(len(values)+1)]
                assert not stream.read(1)
            reset_exact = all(np.array_equal(outputs[0][n],outputs[-1][n]) for n in NAMES)
            native = {n:np.stack([f[n].reshape(2) if n=='activations' else f[n] for f in outputs[:-1]]) for n in NAMES}
            del outputs
            precise = deepcopy(model).double().eval()
            precise.reset_hidden(); precise.hidden=precise.hidden.double(); precise.cell=precise.cell.double()
            oracle = NumpyPreciseStep(model)
            expected = {n:[] for n in NAMES}; shipped={n:[] for n in NAMES}; mathematical={n:[] for n in NAMES}
            options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
            android=ort.InferenceSession(str(assets/item['asset']), options, providers=['CPUExecutionProvider'])
            hidden=np.zeros((item['num_layers'],1,item['hidden_dim']),np.float32); cell=hidden.copy()
            with torch.inference_mode():
                for frame in values:
                    logits=precise(torch.from_numpy(frame[None,None,:]).double())
                    probabilities=torch.softmax(logits,dim=1).reshape(3).numpy()
                    rounded_hidden=precise.hidden.float(); rounded_cell=precise.cell.float()
                    math_reference=(np.array([probabilities[0]+probabilities[1],probabilities[1]],np.float32),
                               rounded_hidden.numpy().copy(),rounded_cell.numpy().copy())
                    precise.hidden=rounded_hidden.double(); precise.cell=rounded_cell.double()
                    reference=oracle(frame)
                    actual=android.run(None,{'feature':frame[None,None,:],'hidden':hidden,'cell':cell})
                    hidden,cell=actual[1:]; actual[0]=actual[0].reshape(2)
                    for n,a,b in zip(NAMES,actual,reference): expected[n].append(b); shipped[n].append(a)
                    for n,a in zip(NAMES,math_reference): mathematical[n].append(a)
            expected={n:np.stack(v) for n,v in expected.items()}; shipped={n:np.stack(v) for n,v in shipped.items()}
            mathematical={n:np.stack(v) for n,v in mathematical.items()}
            legacy_path=args.legacy_run/item['id'].replace('/','_')/'python_reference.npz'
            legacy=np.load(legacy_path)['activations']
            difference=np.abs(expected['activations']-legacy)
            legacy_comparison=dict(rtol=RTOL,atol=ATOL,max_abs=float(difference.max()),
                violating_elements=int(np.count_nonzero(difference>ATOL+RTOL*np.abs(legacy))))
            comparisons={'cpp_rust':exact_comparison(native,expected), 'android_graph_host':exact_comparison(shipped,expected),
                         'android_vs_cpp':exact_comparison(shipped,native)}
            math_comparison=exact_comparison(expected,mathematical)
            decoder_rows=[]
            for selector,pp in item['postprocessors'].items():
                reference_events=events(expected['activations'],pp['parameters'])
                legacy_events=events(legacy,pp['parameters'])
                math_events=events(mathematical['activations'],pp['parameters'])
                variants={name:events(result['activations'],pp['parameters']) for name,result in [('cpp_rust',native),('android_graph_host',shipped)]}
                decoder_rows.append(dict(selector=selector,method=pp['method'],reference_events=reference_events.tolist(),
                    variants={name:dict(count=len(v),exact=np.array_equal(v,reference_events)) for name,v in variants.items()},
                    legacy_float32=dict(count=len(legacy_events),exact=np.array_equal(legacy_events,reference_events))))
                decoder_rows[-1]['pytorch_float64']=dict(count=len(math_events),exact=np.array_equal(math_events,reference_events))
            zero=np.zeros_like(hidden)
            reset=android.run(None,{'feature':values[0][None,None,:],'hidden':zero,'cell':zero})
            reset_exact &= all(np.array_equal(a.reshape(-1),shipped[n][0].reshape(-1)) for n,a in zip(NAMES,reset))
            np.savez_compressed(folder/'precise_reference.npz',**expected)
            prefix=item['id'].replace('/','_')
            hash_path=phone/(prefix+'.sha256'); hash_path.write_bytes(frame_hashes(native))
            activation_path=phone/(prefix+'.activations.f32'); native['activations'].astype('<f4').tofile(activation_path)
            phone_index['models'].append(dict(id=item['id'],asset_sha256=item['asset_sha256'],
                state_hashes=hash_path.name,state_hashes_sha256=digest(hash_path),
                activations=activation_path.name,activations_sha256=digest(activation_path),
                decoders=decoder_rows))
            row.update(source_sha256=digest(checkpoint),native_manifest=str(artifact.manifest_path),
                android_graph_sha256=item['asset_sha256'],frames=len(values),comparisons=comparisons,
                legacy_activation_comparison=legacy_comparison,reset_exact=reset_exact,decoders=decoder_rows,
                mathematical_reference_comparison=math_comparison,
                passed=reset_exact and legacy_comparison['violating_elements']==0 and
                    all(v['exact'] for c in comparisons.values() for v in c.values()) and
                    all(v['within_original_tolerance'] for v in math_comparison.values()) and
                    all(v['exact'] for d in decoder_rows for v in d['variants'].values()))
        except Exception as error:
            row.update(passed=False,error=str(error),traceback=traceback.format_exc())
        row['seconds']=time.monotonic()-started; report['models'].append(row)
        report['passed']=len(report['models'])==len(selected) and all(m['passed'] for m in report['models'])
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (phone/'index.json').write_text(json.dumps(phone_index,indent=2)+'\n')
        print(row['id'],row['passed'],row.get('error',''),
              {k:{n:v['different_elements'] for n,v in c.items()} for k,c in row.get('comparisons',{}).items()},flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
