"""Compare native calibration stimuli, tap scoring and audio alignment to source."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"mir-desktop-app/src"))
from mir_desktop_app import audio_alignment as source
from mir_desktop_app.tap_latency import build_metronome_audio,analyze_tap_latency
from mir_desktop_app.experiment import TrialSpec
from mir_desktop_app.playback import AudioPlaybackSession
import soundfile as sf
import soxr


def compare(a,e,path="root"):
    if isinstance(e,dict):
        assert a.keys()==e.keys(),path
        for k in e:compare(a[k],e[k],path+"/"+k)
    elif isinstance(e,(list,tuple)):
        assert len(a)==len(e),path
        for i,(a,e) in enumerate(zip(a,e)):compare(a,e,f"{path}/{i}")
    elif isinstance(e,float):
        assert np.isclose(a,e,rtol=2e-10,atol=2e-10),(path,a,e)
    else:assert a==e,(path,a,e)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--native-bin",type=Path,default=ROOT/"mir-desktop-app/native-workflows/target/debug/mir-native-workflows")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def native(r):
        p=subprocess.run([str(args.native_bin)],input=json.dumps(r,allow_nan=False),capture_output=True,text=True,check=True)
        return json.loads(p.stdout)
    rng=np.random.default_rng(89195);requests=[];expected=[]
    for _ in range(200):
        taps=rng.uniform(-.3,8,size=rng.integers(0,40)).tolist();beats=np.arange(0,8,.5).tolist()
        result=analyze_tap_latency(taps,beats)
        requests.append({"operation":"tap-latency","taps":taps,"beats":beats})
        expected.append({**asdict(result),"matched_count":result.matched_count,"median_offset_ms":result.median_offset_ms,"jitter_ms":result.jitter_ms,"p95_absolute_ms":result.p95_absolute_ms})
    for n,m in [(0,0),(5,10),(10,0),(10,1),(40,5),(1001,73),(2010,331),(5000,1700)]:
        for kind in ("noise","constant"):
            captured=rng.standard_normal(n) if kind=="noise" else np.ones(n)
            template=rng.standard_normal(m) if kind=="noise" else np.ones(m)
            requests.append({"operation":"alignment-correlation","captured":captured.tolist(),"template":template.tolist()})
            expected.append(source._normalized_valid_correlation(captured,template).tolist())
    result=native({"operation":"batch","requests":requests})
    for i,(a,e) in enumerate(zip(result,expected)):compare(a,e,f"case-{i}")
    stimuli=[]
    for rate,bpm in [(48000,120),(22050,99.91),(8000,300),(2000,4000)]:
        expected_audio,beats=build_metronome_audio(sample_rate=rate,bpm=bpm,count_in_beats=2,measured_beats=3)
        result=native({"operation":"metronome","sample_rate":rate,"bpm":bpm,"count_in_beats":2,"measured_beats":3})
        a=np.asarray(result["audio"],np.float32).reshape(-1,2)
        np.testing.assert_allclose(a,expected_audio,rtol=0,atol=1.2e-7)
        assert result["measured_beats"]==list(beats)
        stimuli.append({"rate":rate,"bpm":bpm,"frames":len(a),"maximum_sample_error":float(np.max(np.abs(a-expected_audio)))})
    alignments=[]
    for rate,channels,offset,gain,wrong in [(1000,1,0,1,False),(1000,2,337,.07,False),(22050,2,7019,1,False),(1000,2,700,1,True),(1000,1,1500,1,False)]:
        ref=rng.standard_normal((rate*5,channels)).astype(np.float32)*.05
        # Fade the energetic window so correlation has one meaningful optimum.
        ref[:rate//2]*=.03
        mono=ref[:,0] if channels==2 else ref.mean(axis=1)
        cap=np.concatenate([np.zeros(offset,np.float32),mono*np.float32(gain),rng.standard_normal(rate*5).astype(np.float32)*.01])
        if wrong:cap=rng.standard_normal(len(cap)).astype(np.float32)
        settings=dict(template_seconds=.3,reference_search_seconds=1.0,initial_prefix_seconds=.5,retry_seconds=.2,max_capture_seconds=3.0,minimum_confidence=.7)
        original=source.ReferenceAudioAligner(ref,rate,rate,**settings)
        chunks=[];states=[];step=max(1,int(.037*rate));cursor=0
        while cursor<len(cap):
            chunk=cap[cursor:cursor+step];cursor+=len(chunk);chunks.append(chunk.tolist());original.feed(chunk)
            deadline=time.monotonic()+5
            while original._attempt_running:
                assert time.monotonic()<deadline,"reference alignment worker timed out"
                time.sleep(.001)
            states.append({"alignment":None if original.alignment is None else asdict(original.alignment),"best_confidence":original.best_confidence,"exhausted":original.exhausted})
        result=native({"operation":"alignment-replay","reference":ref.reshape(-1).tolist(),"channels":channels,"sample_rate":rate,"settings":settings,"chunks":chunks})
        compare(result,states,"alignment")
        if not wrong:assert result[-1]["alignment"]["stream_offset_seconds"]==offset/rate
        else:assert result[-1]["alignment"] is None and result[-1]["exhausted"]
        alignments.append({"sample_rate":rate,"channels":channels,"offset_samples":offset,"wrong_audio":wrong,"state":result[-1]})
    excerpts=[]
    for rate,channels,subtype,extension in [(100,1,"FLOAT","wav"),(22050,2,"PCM_16","wav"),(48000,2,"PCM_24","flac"),(44100,1,"FLOAT","wav")]:
        samples=rng.standard_normal((rate,channels)).astype(np.float32)*.1
        path=args.output.resolve()/f"audio-{rate}.{extension}";sf.write(path,samples,rate,subtype=subtype)
        for start in (.125,.765):
            trial=TrialSpec("trial-1","no_assist","fixture:001","fixture","fixture",path,hashlib.sha256(path.read_bytes()).hexdigest(),"b"*64,start,.5,(),(),0)
            expected_audio,actual_rate=AudioPlaybackSession(trial)._load_excerpt()
            request={"operation":"audio-excerpt","trial":trial.as_dict(),"include_audio":True}
            result=native(request);actual=np.asarray(result["audio"],np.float32).reshape(-1,channels)
            np.testing.assert_array_equal(actual,expected_audio);assert result["sample_rate"]==actual_rate
            record={"rate":rate,"channels":channels,"format":extension,"start":start,"frames":len(actual),"decoded_exact":True}
            if rate>100:
                target=16000;converted=soxr.resample(expected_audio,rate,target,quality="HQ")
                request.update(target_rate=target,resampler_library=str(ROOT/"mir-core/native-dsp/build/libmir_dsp.so"))
                result=native(request);actual=np.asarray(result["audio"],np.float32).reshape(-1,channels)
                np.testing.assert_allclose(actual,converted,rtol=0,atol=2e-7)
                record["resampling_maximum_error"]=float(np.max(np.abs(actual-converted)))
            excerpts.append(record)
        path.write_bytes(path.read_bytes()+b"changed")
        result=subprocess.run([str(args.native_bin)],input=json.dumps(request),capture_output=True,text=True)
        assert result.returncode!=0 and "hash" in result.stderr
    report={"passed":True,"tap_and_correlation_cases":len(requests),"metronomes":stimuli,"alignment_cases":alignments,"audio_excerpts":excerpts,"source_sha256":hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),"scope":"source math, decoded samples and causal alignment search parity; physical playback timing not measured"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
