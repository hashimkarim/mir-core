"""Freeze independent Python decisions for each promoted router policy."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from classifierlab.window_routing import RuntimeRouterState, runtime_policy_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    records = []
    for item in catalog["classifiers"]:
        payload = item["runtime_policy"]
        policy = runtime_policy_from_dict(payload)
        state = RuntimeRouterState(item["labels"], policy)
        classes = len(item["labels"])
        rng = np.random.default_rng(20260905)
        probes = [np.zeros(classes)] * 12
        for label in range(classes):
            strong = np.full(classes, -5.0)
            strong[label] = 8.0
            probes.extend([strong] * 20)
            probes.extend([np.zeros(classes)] * 12)
        probes.extend(rng.normal(0, 5, (64, classes)))
        decisions = []
        for values in probes:
            decision = asdict(state.update_logits(values))
            for key in ("probabilities", "ema_probabilities"):
                decision[key] = decision[key].tolist()
            decisions.append({"logits": values.tolist(), "expected": decision})
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        records.append({"id": item["id"], "labels": item["labels"], "policy_json": raw,
                        "policy_sha256": hashlib.sha256(raw.encode()).hexdigest(), "decisions": decisions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mir.python-router-golden/v1", "source_catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(), "policies": records}, separators=(",", ":")) + "\n")
    print(f"Saved {len(records)} frozen policies and {sum(len(r['decisions']) for r in records)} full decisions")


if __name__ == "__main__":
    main()
