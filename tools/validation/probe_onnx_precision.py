"""Compare graph optimization policies against a saved PyTorch stream.

This is diagnostic only: it never modifies a shipped graph or runtime policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from check_promoted_streams import NAMES, compare, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('saved_run', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--model-id', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    catalog = json.loads((root/'mir-android-app/app/src/main/assets/models/catalog.json').read_text())
    item = next(x for x in catalog['beat_models'] if x['id'] == args.model_id)
    report = json.loads((args.saved_run/'report.json').read_text())
    row = next(x for x in report['models'] if x['id'] == args.model_id)
    manifest = Path(row['native_manifest'])
    native = json.loads(manifest.read_text())
    native_graph = manifest.parent/native['onnx']['filename']
    android_graph = root/'mir-android-app/app/src/main/assets'/item['asset']
    features = np.fromfile(args.saved_run/'features.f32', dtype='<f4').reshape(-1, 272)
    expected = dict(np.load(args.saved_run/args.model_id.replace('/', '_')/'python_reference.npz'))
    result = dict(schema='mir.onnx-optimization-precision/v1', id=args.model_id,
        feature_sha256=digest(args.saved_run/'features.f32'), policies=[])
    for graph in (native_graph, android_graph):
        for name, level in [('disabled', ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                            ('basic', ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                            ('extended', ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
                            ('all', ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.graph_optimization_level = level
            session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
            feature_name = session.get_inputs()[0].name
            hidden = np.zeros((item['num_layers'], 1, item['hidden_dim']), np.float32)
            cell = hidden.copy()
            values = {n: [] for n in NAMES}
            for frame in features:
                outputs = session.run(None, {feature_name:frame[None,None,:], 'hidden':hidden, 'cell':cell})
                hidden, cell = outputs[1:]
                outputs[0] = outputs[0].reshape(2)
                for n, value in zip(NAMES, outputs): values[n].append(value)
            values = {n:np.stack(v) for n,v in values.items()}
            entry = dict(graph=str(graph), graph_sha256=digest(graph), optimization=name,
                         comparisons=compare(values, expected, 5e-5))
            result['policies'].append(entry)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(graph.name, name, {n:v['stream_violating_elements'] for n,v in entry['comparisons'].items()}, flush=True)


if __name__ == '__main__': main()
