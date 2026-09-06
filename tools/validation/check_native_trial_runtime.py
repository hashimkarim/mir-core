"""Live native inference trials using private callback and capture backends.

The capture process reads the actual PCM written by the output callback. This
exercises the native route bank, content alignment, UDP, capture and analysis.
These are host integration tests, not participant or trained-accuracy results.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"mir-desktop-app/src"))
from mir_desktop_app.experiment import TrialSpec,TrialCapture
from mir_desktop_app.pipeline import FoldContext
from mir_desktop_app.experiment_analysis import analyze_trial
from mir_desktop_app.haptics_protocol import ActuationPacket,DeviceAck,AckStatus
from check_native_playback_math import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runtime-session",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    workflow=ROOT/"mir-desktop-app/native-workflows/target/debug/mir-native-workflows"
    engine=ROOT/"mir-desktop-app/native-engine/target/debug/mir-native-engine"
    library=out/"libportaudio.so.2"
    subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-shared","-fPIC",str(Path(__file__).parent/"fixtures/native_playback_host.c"),"-o",str(library),"-ldl","-lpthread"],check=True)
    evdev=out/"evdev-fixture";evdev.touch()
    backend=out/"pw-cat"
    backend.write_text("#!"+sys.executable+"\n"+'''import os,struct,sys,time
from pathlib import Path
path=Path(os.environ["MIR_TEST_PLAYED_PCM"])
while not path.exists():time.sleep(.001)
with path.open("rb",buffering=0) as f:
    pending=b""
    while True:
        raw=f.read(441*2*4-len(pending))
        if not raw:time.sleep(.001);continue
        pending+=raw
        if len(pending)<441*2*4:continue
        values=struct.unpack("<882f",pending);pending=b""
        # float32 mean, identical to a two-channel monitor mix.
        mono=[struct.unpack("<f",struct.pack("<f",a+b))[0]*0.5 for a,b in zip(values[::2],values[1::2])]
        sys.stdout.buffer.write(struct.pack("<441f",*mono));sys.stdout.buffer.flush()
''');backend.chmod(0o700)
    original=json.loads(args.runtime_session.read_text())
    original.pop("recording",None);original["source"]={"kind":"pipewire","target":"isolated-monitor"}
    original["udp"]={"enabled":False};original["routing"]["execution"]="asynchronous"
    session=out/"runtime.json";session.write_text(json.dumps(original))
    result=subprocess.run([str(engine),"--config",str(session),"--validate-config"],text=True,capture_output=True,check=True)
    provenance=next(json.loads(l)["pipeline_provenance"] for l in result.stdout.splitlines() if json.loads(l)["type"]=="ready")
    fold=FoldContext(out,0).as_dict()
    bundle={"schema":"mir.native-experiment-runtime/v1","fold":fold,"session":str(session),"session_sha256":hashlib.sha256(session.read_bytes()).hexdigest(),"source_files":[],"provenance":provenance,"scope":"private integration fixture; source identity tested by export binding gate"}
    bundle_path=out/"runtime-bundle.json";bundle_path.write_text(json.dumps(bundle))
    base={**os.environ,"LD_LIBRARY_PATH":str(out)+":"+os.environ.get("LD_LIBRARY_PATH",""),"LD_PRELOAD":str(library),"MIR_TEST_EVDEV":str(evdev),"PATH":str(out)+":"+os.environ["PATH"]}
    def native(r,env=base,reject=False):
        p=subprocess.run([str(workflow)],input=json.dumps(r),text=True,capture_output=True,env=env,timeout=60)
        if reject:assert p.returncode!=0,p.stdout;return p.stderr
        assert p.returncode==0,p.stderr;return json.loads(p.stdout)
    # A nonrepeating stereo waveform establishes one unambiguous alignment.
    audio=np.random.default_rng(9918).normal(0,.025,(22050*8,2)).astype(np.float32)
    t=np.arange(len(audio))/22050;audio[:,0]+=np.asarray(.08*np.sin(2*np.pi*221*t),np.float32)
    path=out/"audio.wav";sf.write(path,audio,22050,subtype="FLOAT")
    reports=[]
    for condition in ("stock_beatnet","routed_system"):
        trial=TrialSpec("trial-inference",condition,"fixture:001","fixture","Fixture",path,hashlib.sha256(path.read_bytes()).hexdigest(),"a"*64,0.0,8.0,tuple(np.arange(.45,8,.5)),tuple(np.arange(.45,8,.5)[::4]),0).as_dict()
        plan={"schema":"mir.rhythm-assist-experiment-plan/v1","plan_id":"plan-fixture","participant_id":"fixture-not-human","seed":42,"fold":fold,"trials":[trial],"rt_tolerances_ms":[30,50,70,100,150]}
        workspace=out/condition;native({"operation":"workspace-create","workspace":str(workspace),"plan":plan})
        receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.bind(("127.0.0.1",0));receiver.settimeout(.1)
        stop=threading.Event();packets=[];errors=[]
        def listen():
            while not stop.is_set():
                try:
                    raw,addr=receiver.recvfrom(256);packet=ActuationPacket.from_bytes(raw);packets.append(packet)
                    receiver.sendto(DeviceAck(AckStatus.OK,packet.sequence,7).to_bytes(),addr)
                except socket.timeout:pass
                except BaseException as exc:errors.append(str(exc));return
        listener=threading.Thread(target=listen);listener.start()
        request={"operation":"trial-run","workspace":str(workspace),"input":{"backend":"evdev","identifier":str(evdev)},"output":{"identifier":"portaudio:default","label":"Fixture output","backend":"portaudio","target":None,"is_default":True},"target":"%s:%s"%receiver.getsockname(),"resampler_library":str(ROOT/"mir-core/native-dsp/build/libmir_dsp.so"),"runtime_bundle":str(bundle_path),"runtime_bundle_sha256":hashlib.sha256(bundle_path.read_bytes()).hexdigest()}
        try:result=native(request,{**base,"MIR_TEST_PLAYED_PCM":str(out/f"{condition}.f32")})
        finally:stop.set();listener.join(1);receiver.close()
        assert not errors,errors
        saved=json.loads((workspace/"captures/trial-inference.json").read_text());capture=TrialCapture.from_dict(saved);capture.validate()
        compare(result["analysis"],analyze_trial(capture))
        execution=json.loads((workspace/"execution/trial-inference.json").read_text())
        assert execution["alignment"]["stream_offset_seconds"]==0.0
        assert execution["alignment"]["confidence"]>.999999
        assert len(capture.process_observations)>300
        assert capture.system_events and packets
        assert len(capture.system_events)<=len(packets)
        assert all(p.delay_us==0 for p in packets)
        if condition=="stock_beatnet":assert {e.route_label for e in capture.system_events}=={"stock"}
        else:assert sum(f.classifier_process_seconds for f in capture.process_observations)>0
        reports.append({"condition":condition,"frames":len(capture.process_observations),"events":len(capture.system_events),"acknowledgements":len(capture.device_observations),"alignment":execution["alignment"],"source_capture_analysis_passed":True})
    report={"passed":True,"cases":reports,"scope":"private native callback-to-model-to-UDP host integration; no physical devices or accuracy claims"}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
