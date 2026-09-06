"""Freeze the original dance dispatch lifecycle over independently saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    source=root/"mir-embedded-pp/experiments/dancebeat_methods/common/tournament.py"
    sys.path.insert(0,str(source.parent))
    from tournament import Event,Cue,dispatch_cues
    fixture=root/"mir-embedded-pp/test/fixtures/dance_port_reference.json"
    suite=json.loads(fixture.read_text())
    cases=[]
    for case in suite["cases"]:
        if not case["parameters"]["explicit_positions"]:continue
        tracks=suite["groups"][case["group"]]
        expected=[]
        for track,predictions in zip(tracks,case["expected"],strict=True):
            events=[Event(e[0]/1e6,e[1]/1e6,e[2]/1e6,e[3],e[4]) for e in track["events"]]
            cues=[Cue(p[0]/1e6,p[1]/1e6,p[2]/1e6,p[3]/65535) for p in predictions]
            expected.append([[round(c.trigger_source_time_s*1e6),round(c.scheduled_time_s*1e6),
                              round(c.predicted_time_s*1e6),round(c.confidence*65535)] for c in dispatch_cues(events,cues)])
        cases.append({"id":case["id"],"expected":expected})
    result={"schema":"mir.desktop-dance-dispatch-reference/v1",
        "prediction_fixture_sha256":hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "dispatch_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"cases":cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,separators=(",",":"))+"\n")
    print(f"Saved {len(cases)} original dispatch cases")


if __name__=="__main__":main()
