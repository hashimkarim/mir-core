# BeatNet baseline checkpoints

This folder packages the three published BeatNet weight files. `baseline` is
the published model that the thesis compares against and the starting point of
the runs that fine-tune from published weights.

## Contents

| Selector | Packaged file | Upstream file | Upstream training data | SHA-256 |
| --- | --- | --- | --- | --- |
| `baseline` | `model_1_weights.pt` | `model_1_weights.pt` | GTZAN | `619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84` |
| `baseline_alt0` | `baseline_alt0.pt` | `model_2_weights.pt` | Ballroom | `5878a18c079fa0b0139879b14ed2b5b7595faef8c3d16210aed141fd00fa2d58` |
| `baseline_alt1` | `baseline_alt1.pt` | `model_3_weights.pt` | Rock Corpus | `0c52a074ea38e8cb4a760ecfa3747c9cf91a1e3cd19f238eed80b0de763989ca` |

`__init__.py` holds the selector table and the pinned digests.

Source of `model_1_weights.pt`:

https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/models/model_1_weights.pt

The training data column follows the upstream BeatNet README. A copy of the
upstream repository is in
`thesis-docs/literature/codebases/beat-detection/beatnet/mjhydri/`.

## How to use it

```python
from mir_core.checkpoints import (
    beatnet_baseline_checkpoint_names,
    beatnet_baseline_checkpoint_path,
)

names = beatnet_baseline_checkpoint_names()
path = beatnet_baseline_checkpoint_path("baseline")
```

## Status

Current on 2026-10-09. The three files match the digests above and the
upstream files in the copy named above (checked with `sha256sum`).
