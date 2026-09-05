# mir-core
 
Shared Python package for the MSc thesis MIR project.
Successor to `mir-beat-env` from the prototype repo.
 
## Structure
 
```
mir_core/
  models/        — model definitions (beat detection, classifier)
  preprocessing/ — audio preprocessing pipeline
  evaluation/    — evaluation metrics and reporting
  datasets/      — dataset loader adapters and metadata interfaces
  splitting/     — deterministic, group-aware cross-dataset split plans
  export/        — checkpoint loading and export helpers
mir_env/
  verify_installation.py — environment sanity check
```
 
## Install
 
```bash
pip install -e .
```
 
## Usage
 
```python
import mir_core
```

### Shared split plans

`mir_core.splitting` is the task- and audio-profile-independent split engine:

```python
from mir_core.splitting import SplitRecord, build_split_plan

plan = build_split_plan(
    [
        SplitRecord(
            uid="dataset:001",
            dataset_id="dataset",
            group_id="artist:example",
            strata=(("label", "example"),),
        ),
        # Supply the complete canonical experiment universe.
    ],
    seed=42,
    n_folds=5,
    validation_fraction=0.1,
)
```

It assigns whole leakage groups to train, validation, and test for every fold,
validates global group isolation, and exposes canonical JSON serialization plus
self-verifying record, membership, and plan hashes. Dataset-specific manifest
adapters and post-split policies belong in the consuming repository.
 
## Rules
 
All other code repos (`mir-train-hpc`, `mir-desktop-app`, `mir-webapp`,
`mir-embedded-ai`) import from this package only.
No sibling repo imports another sibling repo directly.

## Third-party attribution

Madmom-derived behavior and the separate non-commercial/share-alike terms for
the packaged legacy BockTCN model pickles are documented in
[`third_party/madmom-NOTICE.md`](third_party/madmom-NOTICE.md).
