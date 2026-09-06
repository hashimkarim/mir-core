#!/usr/bin/env python3
"""Replay every promoted classifier from waveform through its frozen router.

Feature and classifier references execute the original PyTorch/TensorFlow code.
Discrete routes and causal timestamps must match; numerical tolerances are the
existing fixed frontend/classifier gates. This is not a music-accuracy score.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Both sides of this gate use their CPU contracts. YAMNet's reference class
# otherwise selects a visible GPU even when its wrapper receives device=cpu.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

from classifierlab.causal_streaming import build_causal_feature_frontend
from classifierlab.live_runtime import LiveCausalFeatureFrontend, LiveClassifierRouter
from classifierlab.native_export import export_native_classifier_checkpoint
from classifierlab.router_evaluation import load_classifier_checkpoint
from classifierlab.window_routing import runtime_policy_from_dict
from mir_core.checkpoints import load_trained_model_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--fold",type=int,action="append")
    parser.add_argument("--family",choices=["efficientat","yamnet"],action="append")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    binary=root/"mir-desktop-app/native-runtime/target/debug/examples/causal_classifier_replay"
    reports=[]
    for family in (args.family or ["efficientat","yamnet"]):
        trained=load_trained_model_bundle("classifier","latin_router",family)
        for fold in (args.fold or range(5)):
            case=output/f"{family}-fold-{fold}";case.mkdir(exist_ok=True)
            checkpoint=trained.checkpoint_path(fold)
            bundle=load_classifier_checkpoint(checkpoint,device="cpu")
            exported=export_native_classifier_checkpoint(checkpoint,cache_root=output/"artifacts",
                include_audio_frontend=True,efficientat_source_root=root/"EfficientAT")
            frontend=exported["audio_frontend"]
            policy_value=bundle.router_config["runtime_policy"]
            policy=runtime_policy_from_dict(policy_value)
            policy_path=case/"policy.json";policy_path.write_text(json.dumps(policy_value))
            cfg={"kind":family,"artifact":frontend["manifest_path"]}
            if family=="efficientat":cfg["feature_config"]=bundle.feature_config
            else:cfg["backend_library"]=str(root/"mir-embedded-ai/native-litert/build/libmir_litert.so")
            classifier={"artifact":exported["manifest_path"],"policy":str(policy_path),
                "policy_sha256":hashlib.sha256(policy_path.read_bytes()).hexdigest(),"frontend":cfg,
                "resampler_library":str(root/"mir-core/native-dsp/build/libmir_dsp.so")}
            reference_config=dict(bundle.feature_config)
            if family=="efficientat":reference_config["repo_path"]=str(root/"EfficientAT")
            source_frontend=build_causal_feature_frontend(reference_config,device=torch.device("cpu"),fast_forward_batch_size=1)
            live=LiveCausalFeatureFrontend(source_frontend,source_sample_rate=22050)
            reference=LiveClassifierRouter(model=bundle.model,labels=bundle.labels,policy=policy,
                frontend=live,device=torch.device("cpu"))
            duration=(policy.window_contract["segment_frames"]+4*policy.window_contract["hop_frames"]+4)/live.feature_fps + 1.1
            count=int(duration*22050)
            t=np.arange(count,dtype=np.float64)/22050
            audio=(0.08*np.sin(2*np.pi*220*t)+0.03*np.sin(2*np.pi*(660*t+35*t*t))
                +np.random.default_rng(2107).normal(0,0.005,count)).astype(np.float32)
            audio[:3000]=0;audio[count//2]=-0.85
            chunks=[[]];cursor=0;sizes=[441,883,97,4096,1701,12]
            while cursor<count:
                size=sizes[len(chunks)%len(sizes)]
                chunks.append(audio[cursor:cursor+size].tolist());cursor+=size
            chunks.append([])
            replay={"source_rate":22050,"frontend":cfg,"classifier":classifier,
                "resampler_library":classifier["resampler_library"],"chunks":chunks}
            path=case/"input.json";path.write_text(json.dumps(replay))
            frames=[]
            for chunk in chunks:frames.extend(live.process_audio(np.asarray(chunk,np.float32)))
            reference.reset()
            decisions=[]
            for chunk in chunks:decisions.extend(reference.process_audio(np.asarray(chunk,np.float32)))
            command=subprocess.run([str(binary),str(path)],text=True,capture_output=True)
            (case/"stderr.log").write_text(command.stderr)
            if command.returncode:raise RuntimeError(command.stderr)
            (case/"actual.json").write_text(command.stdout)
            result=json.loads(command.stdout);actual=result["result"]
            expected_features=np.stack([f.value for f in frames]);observed=np.asarray([f["values"] for f in actual["frames"]],np.float32)
            gate=frontend["parity_validation"]
            np.testing.assert_allclose(observed,expected_features,rtol=gate["rtol"],atol=gate["atol"])
            for got,want in zip(actual["frames"],frames,strict=True):
                for key in ["frame_index","source_start_seconds","source_end_seconds","availability_seconds"]:
                    assert got[key]==getattr(want,key),(family,fold,key)
            assert len(decisions)>=4 and result["reset_exact"]
            maximum_probability_error=0.0
            for got,want in zip(actual["decisions"],decisions,strict=True):
                for key in ["feature_start_frame","feature_end_frame","source_start_seconds","source_end_seconds","availability_seconds"]:
                    assert got[key]==getattr(want,key),(family,fold,key)
                expected=dataclasses.asdict(want.routing)
                for key,value in got["routing"].items():
                    if key in ["confidence","probabilities","ema_probabilities"]:
                        np.testing.assert_allclose(value,expected[key],rtol=5e-4,atol=2e-3)
                        maximum_probability_error=max(maximum_probability_error,float(np.max(np.abs(np.asarray(value)-expected[key]))))
                    else:assert value==expected[key],(family,fold,key,value,expected[key])
            report={"case":f"{family}-fold-{fold}","checkpoint_sha256":bundle.checkpoint_sha256,
                "policy_sha256":classifier["policy_sha256"],"frames":len(frames),"decisions":len(decisions),
                "max_feature_error":float(np.max(np.abs(observed-expected_features))),
                "max_probability_error":maximum_probability_error,"routes":"exact","timestamps":"exact","reset":"exact","status":"pass"}
            reports.append(report);print(json.dumps(report),flush=True)
            (output/"report.json").write_text(json.dumps({"schema":"mir.promoted-causal-router-parity/v1","complete":False,"cases":reports},indent=2)+"\n")
    sources=[root/"mir-train-hpc/classifierlab"/name for name in ["live_runtime.py","causal_streaming.py","window_routing.py"]]
    (output/"report.json").write_text(json.dumps({"schema":"mir.promoted-causal-router-parity/v1","complete":len(reports)==10,"cases":reports,
        "sources":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+"\n")


if __name__=="__main__":main()
