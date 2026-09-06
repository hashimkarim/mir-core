"""Measure backend and precision sensitivity without changing a parity threshold."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from beatlab.models import extract_checkpoint_state_dict, normalize_state_dict_keys
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.checkpoints import list_trained_model_bundles

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("features",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument('--model-id')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    catalog=json.loads((root/"mir-android-app/app/src/main/assets/models/catalog.json").read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    features=np.fromfile(args.features,dtype='<f4').reshape(-1,272)
    records=[]
    for item in (r for r in catalog['beat_models'] if r['id']==args.model_id or (args.model_id is None and r['id'].startswith('beatnet/stock/'))):
        if item['id'].startswith('beatnet/stock/'):
            filename={'baseline':'model_1_weights.pt','baseline_alt0':'baseline_alt0.pt','baseline_alt1':'baseline_alt1.pt'}[item['condition']]
            source=root/'mir-core/mir_core/checkpoints/beatnet'/filename
        else:
            bundles={b.bundle_id:b for b in list_trained_model_bundles(verify_files=True)}
            source=bundles[item.get('bundle_id') or item['id'].rsplit('/fold_',1)[0]].checkpoint_path(item['fold'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['source_sha256']
        model=BeatNetCRNN().eval()
        state=normalize_state_dict_keys(extract_checkpoint_state_dict(torch.load(source,map_location='cpu',weights_only=False)),'beatnet')
        missing,extra=model.load_state_dict(state,strict=False)
        assert set(missing)<={'hidden','cell'} and not extra
        model.reset_hidden()
        manual=copy.deepcopy(model)
        fp64=copy.deepcopy(model).double()
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        runtime=ort.InferenceSession(str(root/'mir-android-app/app/src/main/assets'/item['asset']),sess_options=options,providers=['CPUExecutionProvider'])
        h=np.zeros((2,1,150),np.float32);cell=h.copy()
        comparison={key:np.zeros(3) for key in ['onnx_vs_torch32','torch_no_mkldnn_vs_torch32','torch32_vs_fp64','onnx_vs_fp64']}
        violations={key:np.zeros(3,dtype=np.int64) for key in comparison}
        for x in features:
            native=runtime.run(None,{'feature':x[None,None,:],'hidden':h,'cell':cell})
            h,cell=native[1:]
            refs=[]
            with torch.inference_mode():
                for m,flag in [(model,True),(manual,False),(fp64,False)]:
                    with torch.backends.mkldnn.flags(enabled=flag):
                        logits=m(torch.from_numpy(x[None,None,:]).to(next(m.parameters()).dtype))
                        prob=torch.softmax(logits,dim=1).reshape(3).numpy()
                    refs.append((np.array([prob[0]+prob[1],prob[1]]),m.hidden.numpy(),m.cell.numpy()))
            for key,a,b in [('onnx_vs_torch32',native,refs[0]),('torch_no_mkldnn_vs_torch32',refs[1],refs[0]),('torch32_vs_fp64',refs[0],refs[2]),('onnx_vs_fp64',native,refs[2])]:
                comparison[key]=np.maximum(comparison[key],[np.max(np.abs(x-y)) for x,y in zip(a,b)])
                violations[key]+=np.asarray([np.count_nonzero(np.abs(x-y)>2e-6+2e-5*np.abs(y)) for x,y in zip(a,b)])
        records.append(dict(id=item['id'],frames=len(features),max_abs_events_hidden_cell={k:v.tolist() for k,v in comparison.items()},legacy_violating_elements_events_hidden_cell={k:v.tolist() for k,v in violations.items()}))
        args.output.write_text(json.dumps(records,indent=2)+'\n')
        print(records[-1],flush=True)

if __name__=='__main__':main()
