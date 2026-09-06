"""Bind a promoted classifier and all four source decoder families to a native
participant runtime, and reject identity, parameter and fold substitutions."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"mir-desktop-app/src"))
from mir_core.checkpoints import load_trained_model_bundle
from mir_core.checkpoints.beatnet import base_checkpoint_path
from classifierlab.router_evaluation import load_classifier_checkpoint
from mir_desktop_app.pipeline import InferencePipeline,FoldContext,BeatStage,ClassifierStage
from mir_desktop_app.runtime import CausalPostprocessor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--classifier-replay",type=Path,required=True)
    parser.add_argument("--native-session",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location("runtime_exporter",ROOT/"mir-desktop-app/tools/export_experiment_runtime.py");exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    native=json.loads(args.native_session.read_text());native.pop("recording",None);native["source"]={"kind":"pipewire","target":"operator-selected-monitor"};native["udp"]={"enabled":False}
    classifier=json.loads(args.classifier_replay.read_text())["classifier"]
    trained=load_trained_model_bundle("classifier","latin_router","yamnet")
    checkpoint=trained.checkpoint_path(0);original=load_classifier_checkpoint(checkpoint,device="cpu")
    fold=FoldContext(out,0,contract_hash=original.split_provenance.contract_hash)
    model_config=out/"beat-model.json";model_config.write_text(json.dumps({"input_dim":272,"hidden_dim":128,"num_layers":2}))
    classifier_config=out/"classifier.json";classifier_config.write_text(json.dumps(original.model_config))
    labels=json.loads(Path(classifier["policy"]).read_text())["parameters"]["execution_routes"]["labels"]
    stages=[];routes={};families=[]
    os.environ["MIR_EMBEDDED_PP_HOST_LIBRARY"]=native["postprocessor_library"]
    for i,label in enumerate(labels):
        method=["dbn","dbn_downbeat","particle_filter","heydari_1d_state_space"][i%4]
        selection={"method":method}
        if method in {"dbn","dbn_downbeat"}:selection["online"]=True
        if method=="particle_filter":selection["rng_contract"]="portable-splitmix64-v1"
        reference=CausalPostprocessor(fps=50,parameters=selection,random_seed=42,backend="native")
        tracker=reference._tracker
        if method in {"particle_filter","heydari_1d_state_space"}:decoder={"kind":"particle-filter" if method=="particle_filter" else "state-space-1d","parameters":tracker.native_config}
        elif method=="dbn":decoder={"kind":"beat-dbn","min_bpm":55,"max_bpm":215,"transition_lambda":100,"observation_lambda":16,"num_tempi":None}
        else:decoder={"kind":"joint-dbn","min_bpm":55,"max_bpm":215,"transition_lambda":100,"observation_lambda":16,"num_tempi":60,"beats_per_bar":[3,4],"meter_change_probability":1e-7}
        tracker.close()
        path=out/f"{label}-selection.json";path.write_text(json.dumps(selection))
        stages.append(BeatStage(label,model_config,base_checkpoint_path(),path))
        routes[label]={"artifact":native["artifact"],"frontend":"beatnet","decoder":decoder};families.append(method)
    native["routing"]={"classifier":classifier,"routes":routes,"execution":"asynchronous"}
    session=out/"session.json";session.write_text(json.dumps(native))
    pipeline=InferencePipeline("source-native-binding-gate",fold,tuple(stages),ClassifierStage(classifier_config,checkpoint))
    preset=out/"pipeline.json";preset.write_text(json.dumps(pipeline.as_dict()))
    engine=ROOT/"mir-desktop-app/native-engine/target/debug/mir-native-engine"
    bundle=exporter.export(preset,session,engine)
    result=out/"runtime-bundle.json";result.write_text(json.dumps(bundle,indent=2)+"\n")
    rejected=[]
    def reject(name):
        try:exporter.export(preset,session,engine)
        except (ValueError,subprocess.CalledProcessError) as exc:rejected.append({"case":name,"reason":str(exc)})
        else:raise AssertionError(f"accepted {name}")
    changed=pipeline.as_dict();changed["fold"]["fold_index"]=1;preset.write_text(json.dumps(changed));reject("different training fold");preset.write_text(json.dumps(pipeline.as_dict()))
    model_config.write_text(json.dumps({"hidden_dim":64}));reject("different model configuration");model_config.write_text(json.dumps({"input_dim":272,"hidden_dim":128,"num_layers":2}))
    altered=copy.deepcopy(native);altered["routing"]["routes"][labels[0]]["decoder"]["min_bpm"]=60;session.write_text(json.dumps(altered));reject("different decoder parameters");session.write_text(json.dumps(native))
    report={"passed":True,"source_decoder_families":families,"classifier_checkpoint":str(checkpoint),"source_pipeline_identity":bundle["source_pipeline_identity"],"runtime_bundle_sha256":exporter.digest(result),"rejections":rejected,"scope":"original artifact/fold/decoder identity binding; no participant or physical device measurements"}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
