"""Build and replay a native desktop route bank without capture or device access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch
from beatlab.models import extract_checkpoint_state_dict, normalize_state_dict_keys
from mir_core.checkpoints.beatnet import base_checkpoint_path
from mir_core.models.beatnet.crnn import BeatNetCRNN
from mir_core.native import ensure_streaming_beatnet_onnx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classifier_replay",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    checkpoint=base_checkpoint_path();model=BeatNetCRNN().eval()
    state=normalize_state_dict_keys(extract_checkpoint_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)),"beatnet")
    missing,extra=model.load_state_dict(state,strict=False)
    assert set(missing)<={"hidden","cell"} and not extra
    artifact=ensure_streaming_beatnet_onnx(model,model_name="beatnet",model_config={"input_dim":272,"hidden_dim":128,"num_layers":2},
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),cache_root=out/"artifacts")
    source=json.loads(args.classifier_replay.read_text());assert source["source_rate"]==22050
    audio=np.concatenate([np.asarray(chunk,np.float32) for chunk in source["chunks"]])
    audio=np.pad(audio,(0,(-len(audio))%441));sf.write(out/"audio.wav",audio,22050,subtype="FLOAT")
    classifier=source["classifier"]
    labels=json.loads(Path(classifier["policy"]).read_text())["parameters"]["execution_routes"]["labels"]
    stages={label:{"artifact":str(artifact.manifest_path),"frontend":"beatnet",
        "decoder":{"kind":["beat-dbn","joint-dbn","particle-filter","state-space-1d"][index%4]}} for index,label in enumerate(labels)}
    config={"schema":"mir.native-desktop-session/v1","artifact":str(artifact.manifest_path),"frontend":"beatnet",
        "postprocessor_library":str(root/"mir-embedded-pp/build/ports-remediation/libmir_embedded_pp_host.so"),
        "source":{"kind":"wav","path":str(out/"audio.wav"),"realtime":False},
        "routing":{"classifier":classifier,"routes":stages,"execution":"synchronous-replay"}}
    path=out/"session.json";path.write_text(json.dumps(config,indent=2))
    (out/"routing.json").write_text(json.dumps(config["routing"],indent=2))
    env=dict(os.environ,MIR_NATIVE_ROUTE_TEST_CONFIG=str(path),MIR_EMBEDDED_PP_HOST_LIBRARY=config["postprocessor_library"])
    subprocess.run(["cargo","test","--locked","--manifest-path",str(root/"mir-desktop-app/native-engine/Cargo.toml"),
        "pipeline::tests::every_hot_route_matches_an_independent_uninterrupted_stage","--","--ignored"],check=True,env=env)
    binary=root/"mir-desktop-app/native-engine/target/debug/mir-native-engine"
    reports=[];decisions=[]
    for execution in ["synchronous-replay","asynchronous"]:
        recording=out/f"{execution}.ndjson"
        if recording.exists():raise FileExistsError(recording)
        config["routing"]["execution"]=execution;config["recording"]=str(recording)
        session=out/f"{execution}.json";session.write_text(json.dumps(config,indent=2))
        result=subprocess.run([str(binary),"--config",str(session),"--no-stdin"],text=True,capture_output=True)
        (out/f"{execution}.display.ndjson").write_text(result.stdout)
        if result.returncode:raise RuntimeError(result.stderr)
        rows=[json.loads(line) for line in recording.read_text().splitlines()]
        assert rows[0]["type"]=="ready" and rows[-1]["reason"]=="eof"
        frames=[r for r in rows if r["type"]=="frame"]
        assert len(frames)==len(audio)//441 and [r["frame_index"] for r in frames]==list(range(len(frames)))
        assert all(r["route"] in stages for r in frames)
        routed=[r["decision"] for r in rows if r["type"]=="routing"]
        assert len(routed)>=4
        decisions.append(routed)
        reports.append({"execution":execution,"frames":len(frames),"decisions":len(routed),"routes":len(stages),
            "status":"pass","source_sha256":hashlib.sha256((out/"audio.wav").read_bytes()).hexdigest(),
            "host_timing":rows[-1]})
    # Worker scheduling may apply a route on a different hop. Every classifier
    # window, logit and discrete decision must survive the EOF drain unchanged.
    for a,b in zip(*decisions,strict=True):
        for key in a:
            if key=="logits":np.testing.assert_allclose(a[key],b[key],rtol=2e-5,atol=2e-6)
            elif key=="routing":
                for field in a[key]:
                    if field in {"confidence","probabilities","ema_probabilities"}:np.testing.assert_allclose(a[key][field],b[key][field],rtol=2e-5,atol=2e-6)
                    else:assert a[key][field]==b[key][field]
            else:assert a[key]==b[key]
    (out/"report.json").write_text(json.dumps({"schema":"mir.native-desktop-routing-parity/v1","cases":reports,
        "all_route_states":"exact against independent uninterrupted stages","async_eof":"all decisions retained",
        "timing_scope":"host execution of these fixtures; no device latency measurement"},indent=2)+"\n")
    print(json.dumps({"status":"pass","cases":len(reports),"frames_per_case":len(frames),"routes":len(stages)}))


if __name__=="__main__":main()
