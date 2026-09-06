"""Export all ten promoted Android classifier sources using the desktop ABI.

Checks real calibrated manifest loading and compares rebuilt Rust against PyTorch.
Writes only to a disposable artifact cache and this evidence directory.
"""
import argparse
import copy
import struct
import subprocess
import sys
import tempfile
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch

from classifierlab.native_export import export_native_classifier_checkpoint
from classifierlab.router_evaluation import load_classifier_checkpoint
from mir_core.checkpoints import list_trained_model_bundles
from mir_native_runtime import FeatureModelSession

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--cpp-replay',type=Path,required=True)
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'mir-core/tools'))
from port_validation_plugin import read_outputs
CACHE = Path('/tmp/mir-port-audit-classifiers')
CATALOG = json.loads((ROOT/'mir-android-app/app/src/main/assets/models/catalog.json').read_text())
BUNDLES = {b.bundle_id:b for b in list_trained_model_bundles(verify_files=True)}
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
results = []
(OUT/'classifier-manifests').mkdir(exist_ok=True)


def identity(manifest):
    return {key:value for key,value in {
        'abi_version':manifest['abi_version'], 'architecture':manifest['architecture'],
        'checkpoint_sha256':manifest['checkpoint_sha256'], 'exporter':manifest['exporter']['name'],
        'frontend_contract':manifest['feature_frontend'], 'model_config_sha256':manifest['model_config_sha256'],
        'opset':manifest['onnx']['opset'], 'output_contract':manifest['output_contract'],
        'shape_contract':manifest['shape_contract'], 'torch_version':manifest['exporter']['torch_version'],
        'wrapper':manifest['wrapper'],
    }.items()}


def key_for(payload):
    canonical = json.dumps(payload,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


for item in CATALOG['classifiers']:
    record = {'id':item['id']}
    try:
        bundle_id = item.get('bundle_id') or item['id'].rsplit('/fold_',1)[0]
        source = BUNDLES[bundle_id].checkpoint_path(item['fold'])
        exported = export_native_classifier_checkpoint(source,cache_root=CACHE)
        manifest_path = Path(exported['manifest_path'])
        manifest = json.loads(manifest_path.read_text())
        saved = OUT/'classifier-manifests'/(item['id'].replace('/','_')+'.json')
        saved.write_text(manifest_path.read_text())
        payload = identity(manifest)
        assert key_for(payload) == manifest['artifact_key']
        record.update(source=str(source), source_sha256=exported['checkpoint']['sha256'],
                      artifact_key=manifest['artifact_key'],manifest=str(saved),
                      temperature=manifest['calibration_temperature'])
        neighbor_keys = []
        for direction in (-math.inf,math.inf):
            changed = copy.deepcopy(payload)
            temp = math.nextafter(changed['output_contract'][1]['temperature'],direction)
            changed['output_contract'][1]['temperature'] = temp
            neighbor_keys.append({'temperature':temp,'artifact_key':key_for(changed)})
        record['neighbor_float_identity_keys'] = neighbor_keys
        native = FeatureModelSession(exported['model_path'],threads=1)
        record['load_passed'] = True
        reference = load_classifier_checkpoint(source,device='cpu')
        shape = tuple(manifest['export_example_shape'])
        rng = np.random.default_rng(20260905)
        probes = [np.zeros(shape,np.float32),np.ones(shape,np.float32),
                  np.linspace(-1,1,np.prod(shape),dtype=np.float32).reshape(shape)]
        probes.extend(rng.normal(0,scale,shape).astype(np.float32) for scale in (0.01,0.1,1,3))
        maxima = {'logits':0.0,'probabilities':0.0}
        with torch.inference_mode():
            for probe in probes:
                logits = reference.model(torch.from_numpy(probe))
                expected = {'logits':logits.numpy(), 'probabilities':torch.softmax(logits/manifest['calibration_temperature'],dim=-1).numpy()}
                actual = native.infer(probe)
                for name in expected:
                    maxima[name] = max(maxima[name],float(np.max(np.abs(actual[name]-expected[name]))))
                    np.testing.assert_allclose(actual[name],expected[name],rtol=2e-5,atol=2e-6)
        with tempfile.TemporaryDirectory(prefix='mir-promoted-head-replay-') as raw:
            incoming=Path(raw)/'inputs.replay'; outgoing=Path(raw)/'outputs.replay'
            with incoming.open('wb') as f:
                f.write(struct.pack('<I',len(probes)))
                for probe in probes:
                    f.write(struct.pack('<II',1,probe.ndim))
                    f.write(struct.pack('<'+'Q'*probe.ndim,*probe.shape))
                    f.write(probe.astype('<f4').tobytes())
            subprocess.run([str(args.cpp_replay.resolve()),str(manifest_path),str(incoming),str(outgoing)],check=True,capture_output=True,text=True)
            cpp_max={'logits':0.,'probabilities':0.}
            with outgoing.open('rb') as f, torch.inference_mode():
                assert struct.unpack('<I',f.read(4))[0]==len(probes)
                for probe in probes:
                    actual=read_outputs(f)
                    logits=reference.model(torch.from_numpy(probe))
                    expected={'logits':logits.numpy(),'probabilities':torch.softmax(logits/manifest['calibration_temperature'],dim=-1).numpy()}
                    assert set(actual)==set(expected)
                    for name in expected:
                        np.testing.assert_allclose(actual[name],expected[name],rtol=2e-5,atol=2e-6)
                        cpp_max[name]=max(cpp_max[name],float(np.max(np.abs(actual[name]-expected[name]))))
                assert not f.read(1)
        record.update(passed=True,probes=len(probes),max_abs_by_output=maxima,cpp_max_abs_by_output=cpp_max)
    except Exception as exc:
        record.update(passed=False,error=str(exc),traceback=traceback.format_exc())
    results.append(record)
    (OUT/'promoted-classifiers.json').write_text(json.dumps(results,indent=2)+'\n')
    print(record['id'],record['passed'],record.get('error',''),flush=True)

raise SystemExit(0 if all(r['passed'] for r in results) else 1)
