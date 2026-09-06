"""Sustained trained-checkpoint parity, complete states, reset and decoder events.

The observable gate stays rtol=2e-5/atol=2e-6. State diagnostics retain that
legacy criterion AND report a separate, explicit continuous-stream budget.
Neither numerical check is a held-out musical-accuracy benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import traceback

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr
import torch
from beatlab.models import extract_checkpoint_state_dict, normalize_state_dict_keys
from mir_core.checkpoints import list_trained_model_bundles
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.native import ensure_streaming_beatnet_onnx
from mir_core.preprocessing import BeatNetPreProcessor
from mir_desktop_app.runtime import CausalPostprocessor

ROOT=Path(__file__).resolve().parents[3]
NAMES=('activations','next_hidden','next_cell')
RTOL=2e-5
ATOL=2e-6

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def features(output,frames):
    previews=sorted((ROOT/'mir-data/datasets/local/thesis_stem_previews').glob('*/source-preview.wav'))
    if len(previews)!=3:raise ValueError('Expected three saved salsa/candombe/samba source previews')
    pre=BeatNetPreProcessor(mode='realtime')
    window=np.zeros(2290,np.float32)
    values=[];sources=[]
    for p in previews:
        info=sf.info(p)
        audio,rate=sf.read(p,frames=int(info.samplerate*20),dtype='float32',always_2d=True)
        audio=soxr.resample(audio.mean(axis=1),rate,22050,quality='HQ')
        sources.append(dict(path=str(p),sha256=digest(p),source_sample_rate=rate,samples_used=len(audio)))
        for hop in audio[:len(audio)//441*441].reshape(-1,441):
            window[:-441]=window[441:];window[-441:]=hop
            values.append(np.asarray(pre.process_audio(window),np.float32)[-1].copy())
    real=np.asarray(values,np.float32)
    rng=np.random.default_rng(20260905)
    # Keep at least 60 seconds of unbroken, actual music features; then silence,
    # an explicit finite stress segment and replayed audio, with no model reset.
    values=np.concatenate([real,np.zeros((128,272),np.float32),rng.normal(0,.2,(64,272)).astype(np.float32),real[:1024]])
    if len(values)<frames:raise ValueError('Requested stream exceeds available reference material')
    values=np.ascontiguousarray(values[:frames],dtype='<f4')
    path=output/'features.f32';values.tofile(path)
    return values,dict(path=str(path),sha256=digest(path),frames=len(values),feature_dim=272,
        source_audio=sources,unbroken_music_frames=min(len(real),frames),silence_frames=128,seed=20260905)

def inputs(path,values):
    with path.open('wb') as f:
        f.write(struct.pack('<I',len(values)+1))
        for i,x in enumerate([*values,values[0]]):
            f.write(struct.pack('<IIQQQ',int(i in (0,len(values))),3,1,1,272))
            f.write(x.astype('<f4').tobytes())

def read_u32(stream):return struct.unpack('<I',stream.read(4))[0]
def read_outputs(stream):
    result={}
    for _ in range(read_u32(stream)):
        name=stream.read(read_u32(stream)).decode()
        rank=read_u32(stream);shape=struct.unpack('<'+'Q'*rank,stream.read(8*rank))
        result[name]=np.frombuffer(stream.read(int(np.prod(shape))*4),dtype='<f4').reshape(shape)
    return result

def compare(actual,expected,state_atol):
    result={}
    for name in NAMES:
        a=actual[name];b=expected[name]
        assert a.shape==b.shape and np.isfinite(a).all()
        delta=np.abs(a-b)
        legacy=delta>ATOL+RTOL*np.abs(b)
        tolerance=ATOL if name=='activations' else state_atol
        violations=delta>tolerance+RTOL*np.abs(b)
        index=np.unravel_index(np.argmax(delta),delta.shape)
        result[name]=dict(max_abs=float(delta[index]),peak_index=[int(i) for i in index],
            expected_at_peak=float(b[index]),actual_at_peak=float(a[index]),
            legacy_violating_elements=int(legacy.sum()),stream_violating_elements=int(violations.sum()),
            rtol=RTOL,atol=tolerance,pass_stream=not bool(violations.any()))
    return result

def events(values,parameters):
    parameters=dict(parameters)
    if parameters.get('method')=='particle_filter':
        # The portable native contract fixes both RNG and reduction order;
        # legacy NumPy-global randomness is a different, unsupported contract.
        parameters['rng_contract']='portable-splitmix64-v1'
    decoder=CausalPostprocessor(fps=50,parameters=parameters,random_seed=42,backend='native')
    assert decoder.backend=='cpp-ctypes'
    rows=[]
    try:
        for x in values:
            out=decoder.process_canonical_values(float(x[0]),float(x[1]))
            rows.extend(out.tolist())
    finally:
        decoder._tracker.close()
    return np.asarray(rows,dtype=np.float64).reshape(-1,2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cpp-replay',required=True,type=Path)
    parser.add_argument('--postprocessor-library',required=True,type=Path)
    parser.add_argument('--frames',type=int,default=4096)
    parser.add_argument('--state-atol',type=float,required=True,help='Explicit continuous-state budget; legacy stricter results are always retained')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    assert args.frames>=288 and args.state_atol>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['MIR_EMBEDDED_PP_HOST_LIBRARY']=str(args.postprocessor_library.resolve())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assets=ROOT/'mir-android-app/app/src/main/assets'
    catalog_path=assets/'models/catalog.json';catalog=json.loads(catalog_path.read_text())
    bundles={b.bundle_id:b for b in list_trained_model_bundles(verify_files=True)}
    x,source=features(out,args.frames)
    replay=out/'inputs.replay';inputs(replay,x)
    result=dict(schema='mir.promoted-stream-validation/v1',catalog_sha256=digest(catalog_path),
        cpp_replay_sha256=digest(args.cpp_replay),postprocessor_library_sha256=digest(args.postprocessor_library),
        features=source,observable_tolerance=dict(rtol=RTOL,atol=ATOL),
        state_tolerance=dict(rtol=RTOL,atol=args.state_atol),legacy_state_atol=ATOL,
        note='State budget is explicit. Legacy violations remain reported. Decoded events must agree exactly; this does not measure musical accuracy.',models=[])
    for item in catalog['beat_models'][:args.limit]:
        row=dict(id=item['id'])
        try:
            if item['id'].startswith('beatnet/stock/'):
                filename={'baseline':'model_1_weights.pt','baseline_alt0':'baseline_alt0.pt','baseline_alt1':'baseline_alt1.pt'}[item['condition']]
                checkpoint=ROOT/'mir-core/mir_core/checkpoints/beatnet'/filename
            else:
                bundle=bundles[item.get('bundle_id') or item['id'].rsplit('/fold_',1)[0]]
                assert item['split_contract_hash']==bundle.split_contract['contract_hash']
                checkpoint=bundle.checkpoint_path(item['fold'])
            assert digest(checkpoint)==item['source_sha256']
            assert digest(assets/item['asset'])==item['asset_sha256']
            config={k:item[k] for k in ('input_dim','hidden_dim','num_layers')}
            model=BeatNetCRNN(**config).eval()
            state=normalize_state_dict_keys(extract_checkpoint_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)),'beatnet')
            assert set(dict(model.named_parameters()))<=set(state)
            missing,extra=model.load_state_dict(state,strict=False)
            assert set(missing)<={'hidden','cell'} and not extra
            assert all(torch.equal(p,state[k]) for k,p in model.named_parameters())
            artifact=ensure_streaming_beatnet_onnx(model,model_name='beatnet',model_config=config,checkpoint_sha256=digest(checkpoint),cache_root=out/'exports')
            folder=out/item['id'].replace('/','_');folder.mkdir(exist_ok=True)
            cpp_path=folder/'cpp.replay'
            subprocess.run([str(args.cpp_replay.resolve()),str(artifact.manifest_path),str(replay),str(cpp_path)],check=True,capture_output=True,text=True)
            with cpp_path.open('rb') as stream:
                assert read_u32(stream)==len(x)+1
                cpp_frames=[read_outputs(stream) for _ in range(len(x)+1)]
                assert not stream.read(1)
            assert all(np.array_equal(cpp_frames[0][n],cpp_frames[-1][n]) for n in NAMES)
            cpp={n:np.stack([f[n].reshape(2) if n=='activations' else f[n] for f in cpp_frames[:-1]]) for n in NAMES}
            del cpp_frames
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            android=ort.InferenceSession(str(assets/item['asset']),sess_options=options,providers=['CPUExecutionProvider'])
            hidden=np.zeros((item['num_layers'],1,item['hidden_dim']),np.float32);cell=hidden.copy()
            expected={n:[] for n in NAMES};shipped={n:[] for n in NAMES}
            model.reset_hidden()
            with torch.inference_mode():
                for frame in x:
                    prob=torch.softmax(model(torch.from_numpy(frame[None,None,:])),dim=1).reshape(3).numpy()
                    ref=(np.array([prob[0]+prob[1],prob[1]]),model.hidden.numpy().copy(),model.cell.numpy().copy())
                    actual=android.run(None,{'feature':frame[None,None,:],'hidden':hidden,'cell':cell})
                    hidden,cell=actual[1:]
                    actual[0]=actual[0].reshape(2)
                    for n,a,b in zip(NAMES,actual,ref):expected[n].append(b);shipped[n].append(a)
            expected={n:np.stack(v) for n,v in expected.items()};shipped={n:np.stack(v) for n,v in shipped.items()}
            np.savez_compressed(folder/'python_reference.npz',**expected)
            comparisons={'cpp_rust':compare(cpp,expected,args.state_atol),'android_graph_host':compare(shipped,expected,args.state_atol)}
            row['comparisons']=comparisons
            zero=np.zeros_like(hidden)
            reset=android.run(None,{'feature':x[0][None,None,:],'hidden':zero,'cell':zero})
            assert all(np.array_equal(a.reshape(-1),shipped[n][0].reshape(-1)) for n,a in zip(NAMES,reset))
            decoders=[]
            for selector,pp in item['postprocessors'].items():
                reference=events(expected['activations'],pp['parameters'])
                variants={name:events(values['activations'],pp['parameters']) for name,values in [('cpp_rust',cpp),('android_graph_host',shipped)]}
                decoders.append(dict(selector=selector,method=pp['method'],reference_events=len(reference),
                    variants={name:dict(count=len(values),pass_exact=np.array_equal(values,reference)) for name,values in variants.items()}))
            row.update(source_sha256=digest(checkpoint),android_graph_sha256=digest(assets/item['asset']),
                native_manifest=str(artifact.manifest_path),native_graph_sha256=artifact.manifest['onnx']['sha256'],
                frames=len(x),reset_exact=True,comparisons=comparisons,decoders=decoders,
                passed=all(v['pass_stream'] for p in comparisons.values() for v in p.values()) and all(v['pass_exact'] for d in decoders for v in d['variants'].values()))
        except Exception as e:row.update(passed=False,error=str(e),traceback=traceback.format_exc())
        result['models'].append(row)
        result['passed']=len(result['models'])==len(catalog['beat_models']) and all(r['passed'] for r in result['models'])
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(row['id'],row['passed'],row.get('error',''),flush=True)
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
