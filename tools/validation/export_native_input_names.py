"""Freeze Linux input control labels from the original Python evdev binding."""
import argparse
import json
from pathlib import Path
from evdev import ecodes

def payload():
    return {"key": {str(k):str(v) for k,v in ecodes.bytype[ecodes.EV_KEY].items()},
            "absolute": {str(k):str(v) for k,v in ecodes.bytype[ecodes.EV_ABS].items()}}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.check:
        assert json.loads(args.output.read_text())==payload(), "input control labels have drifted"
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(payload(),sort_keys=True,indent=2)+"\n")
