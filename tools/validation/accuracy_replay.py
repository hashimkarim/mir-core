"""Score complete recordings with frozen checkpoints and decoder parameters.

Original float32 Python inference/causal preprocessing is compared with the
released Rust inference/frontend and C++ decoders. PF uses the same production
SplitMix64 contract on both sides, isolating port effects from RNG substitution.
Android graphs execute on the host here; this is not Android device timing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import onnxruntime as ort
import torch

from accuracy_inventory import digest, write_new
from beatlab.models import extract_checkpoint_state_dict, normalize_state_dict_keys
from mir_core.evaluation.metrics import compute_beat_metrics, compute_downbeat_metrics
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.native import ensure_streaming_beatnet_onnx
from mir_desktop_app.runtime import CausalPostprocessor
from mir_native_runtime import StreamingSession

THREADS_INITIALIZED = False

def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def decode(values, parameters, backend):
    parameters = dict(parameters)
    if parameters.get('method') == 'particle_filter':
        parameters['rng_contract'] = 'portable-splitmix64-v1'
    decoder = CausalPostprocessor(fps=50, parameters=parameters, random_seed=42, backend=backend)
    assert decoder.backend == ('cpp-ctypes' if backend == 'native' else 'python')
    events = []
    started = time.monotonic()
    try:
        for frame, activation in enumerate(values):
            result = decoder.process_canonical_values(float(activation[0]),float(activation[1]))
            for event in result:
                events.append([float(event[0]),int(event[1]),frame])
    finally:
        if hasattr(decoder._tracker,'close'):
            decoder._tracker.close()
    return np.asarray(events,np.float64).reshape(-1,3),time.monotonic()-started


def scores(events, track):
    beats = np.asarray(track['beats_seconds'],np.float64)
    downbeats = np.asarray(track['downbeats_seconds'],np.float64)
    result = compute_beat_metrics(events[:,0],beats)
    if len(downbeats):
        result.update(compute_downbeat_metrics(events[events[:,1]==1,0],downbeats))
    return result


def run_model(task):
    global THREADS_INITIALIZED
    model_item, jobs, options = task
    if not THREADS_INITIALIZED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        THREADS_INITIALIZED = True
    catalogs = {row['uid']:row for path in Path(options['catalogs']).glob('fold-*.json')
                for row in json.loads(path.read_text())['tracks']}
    output = Path(options['output'])/model_item['id'].replace('/','__')
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = Path(jobs[0]['checkpoint'])
    assert digest(checkpoint) == model_item['source_sha256']
    config = {key:model_item[key] for key in ('input_dim','hidden_dim','num_layers')}
    model = BeatNetCRNN(**config).eval()
    state = normalize_state_dict_keys(extract_checkpoint_state_dict(
        torch.load(checkpoint,map_location='cpu',weights_only=False)), 'beatnet')
    missing,extra = model.load_state_dict(state,strict=False)
    assert set(missing) <= {'hidden','cell'} and not extra
    assert all(torch.equal(parameter,state[name]) for name,parameter in model.named_parameters())
    artifact = ensure_streaming_beatnet_onnx(model,model_name='beatnet',model_config=config,
        checkpoint_sha256=digest(checkpoint),cache_root=Path(options['output'])/'artifacts')
    native = StreamingSession(str(artifact.model_path),threads=1)
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads=ort_options.inter_op_num_threads=1
    android_path = Path(jobs[0]['android_graph'])
    assert digest(android_path) == model_item['asset_sha256']
    android = ort.InferenceSession(str(android_path),ort_options,providers=['CPUExecutionProvider'])
    completed = 0
    for job in jobs:
        uid = job['track_uid'];track = catalogs[uid];stem = uid.replace(':','__')
        features = Path(options['features'])
        feature_record = json.loads((features/(stem+'.json')).read_text())
        assert feature_record['identity']['audio_sha256'] == track['audio_hash']
        identity = {'checkpoint_sha256':model_item['source_sha256'],
            'native_graph_sha256':artifact.manifest['onnx']['sha256'],
            'android_graph_sha256':model_item['asset_sha256'],
            'feature_record_sha256':digest(features/(stem+'.json')),
            'annotation_sha256':track['annotation_hash'],
            'postprocessors_sha256':stable_hash(model_item['postprocessors']),
            'runner_sha256':options['runner_sha256'], 'reference_rng':'portable-splitmix64-v1',
            'decoder_reference_backend':options['reference_backend'],
            'metric_contract':'mir-core-declared-conventional-full-recording-70ms/v1'}
        record_path = output/(stem+'.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['identity'] == identity and record['complete']
            assert digest(output/record['arrays']['file']) == record['arrays']['sha256']
            completed += 1
            continue
        started = time.monotonic()
        inputs = {}
        for name in ('python','rust'):
            item = feature_record['arrays'][name];path = features/item['file']
            assert digest(path) == item['sha256']
            inputs[name] = np.load(path,mmap_mode='r',allow_pickle=False)
        count = len(inputs['python'])
        assert count == len(inputs['rust'])
        model.reset_hidden();native.reset()
        hidden = np.zeros((config['num_layers'],1,config['hidden_dim']),np.float32)
        cell = hidden.copy()
        activations = {name:np.empty((count,2),np.float32) for name in ('python','native','android_host')}
        with torch.inference_mode():
            for frame in range(count):
                tensor = torch.from_numpy(np.array(inputs['python'][frame][None,None,:],copy=True))
                probabilities = torch.softmax(model(tensor),dim=1).reshape(3).numpy()
                activations['python'][frame] = [probabilities[0]+probabilities[1],probabilities[1]]
                deployed_input = np.ascontiguousarray(inputs['rust'][frame])
                activations['native'][frame] = native.infer(deployed_input).activations
                actual = android.run(None,{'feature':deployed_input[None,None,:],'hidden':hidden,'cell':cell})
                activations['android_host'][frame] = actual[0].reshape(2)
                hidden,cell = actual[1:]
        inference_seconds = time.monotonic()-started
        assert all(np.isfinite(value).all() for value in activations.values())
        android_exact = np.array_equal(activations['native'].view(np.uint32),activations['android_host'].view(np.uint32))
        comparisons = []
        arrays = dict(activations)
        for selector, decoder in sorted(model_item['postprocessors'].items()):
            reference, reference_time = decode(activations['python'],decoder['parameters'],options['reference_backend'])
            actual, native_time = decode(activations['native'],decoder['parameters'],'native')
            arrays[selector+'_python_events'] = reference
            arrays[selector+'_native_events'] = actual
            reference_scores,actual_scores = scores(reference,track),scores(actual,track)
            comparisons.append({'selector':selector,'method':decoder['method'],
                'selection_scope':'shared-fold-tuned' if decoder['kind']=='tuned' else 'untuned-stock',
                'python_metrics':reference_scores,'native_metrics':actual_scores,
                'metric_deltas':{key:actual_scores[key]-value for key,value in reference_scores.items()},
                'event_stream_exact':np.array_equal(reference,actual),
                'python_events':len(reference),'native_events':len(actual),
                'decoder_seconds':{'python':reference_time,'native':native_time}})
        array_path = output/(stem+'.npz')
        with array_path.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        difference = np.abs(activations['native']-activations['python'])
        record = {'schema':'mir.heldout-port-score/v1','complete':True,
            'model_id':model_item['id'],'track_uid':uid,'fold':job['fold'],'scope':job['scope'],
            'identity':identity,'frames':count,'duration_seconds':track['duration_seconds'],
            'arrays':{'file':array_path.name,'sha256':digest(array_path)},
            'max_activation_error':float(difference.max()),
            'activation_tolerance':{'rtol':2e-5,'atol':2e-6},
            'activation_violations':int(np.count_nonzero(difference>2e-6+2e-5*np.abs(activations['python']))),
            'android_graph_host_activations_bit_exact':android_exact,
            'android_graph_host_max_abs_error':float(np.max(np.abs(activations['native']-activations['android_host']))),
            'decoders':comparisons,'inference_seconds':inference_seconds,
            'wall_seconds':time.monotonic()-started}
        write_new(record_path,record)
        completed += 1
        print(json.dumps({'model':model_item['id'],'uid':uid,'frames':count,
            'android_host_exact':android_exact,'seconds':record['wall_seconds'],
            'completed_for_model':completed,'total_for_model':len(jobs)}),flush=True)
    return {'model':model_item['id'],'completed_tracks':completed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--catalogs',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--model-id',action='append')
    parser.add_argument('--limit-tracks',type=int)
    parser.add_argument('--reference-backend',choices=['python','native'],default='python')
    args = parser.parse_args()
    inventory = json.loads((args.catalogs/'inventory.json').read_text())
    catalog_path = args.workspace/'mir-android-app/app/src/main/assets/models/catalog.json'
    assert digest(catalog_path) == inventory['catalog_sha256']
    catalog = json.loads(catalog_path.read_text())
    options = {key:str(getattr(args,key)) for key in ('catalogs','features','output')}
    options.update(reference_backend=args.reference_backend,runner_sha256=digest(Path(__file__)))
    tasks=[]
    for model in catalog['beat_models']:
        if args.model_id and model['id'] not in args.model_id:
            continue
        jobs=[row for row in inventory['beat_jobs'] if row['model_id']==model['id']]
        tasks.append((model,jobs[:args.limit_tracks],options))
    assert tasks
    args.output.mkdir(parents=True,exist_ok=True)
    if args.workers == 1:
        for task in tasks:
            print(json.dumps(run_model(task)),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(run_model,task) for task in tasks]):
                print(json.dumps(future.result()),flush=True)


if __name__ == '__main__':
    main()
