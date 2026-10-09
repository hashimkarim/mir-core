# BeatNet+ baseline checkpoints

This folder packages the three published BeatNet+ weight files. They mirror
`src/BeatNetPlus/models/` of the BeatNet+ reference code, of which a copy is in
`thesis-docs/literature/codebases/beat-detection/beatnet-plus/`.

## Contents

| Selector | Packaged file | Upstream file and role | SHA-256 |
| --- | --- | --- | --- |
| `baseline` | `baseline.pt` | `generic_weights.pt`, general-purpose BeatNet+ generic weights | `ed52f90e27ff9b5ef3c63f59c6d4b37366f60a21a48ea1d46d7c3e18d6f1977e` |
| `baseline_alt0` | `baseline_alt0.pt` | `generic_main_weights.pt`, generic main-branch weights | `5bbae630b4112f3c1193654e3dbc946b850caa06a2a82e10e8de9e48bb673519` |
| `baseline_alt1` | `baseline_alt1.pt` | `af_non_percussive_weights.pt`, auxiliary-freezing non-percussive adaptation | `15fe9c1ec8f2fca75dd3cecac72a3fbf57c9004e8726a786876ad5e269f4895f` |

`__init__.py` holds the selector table and the pinned digests.

## How to use it

```python
from mir_core.checkpoints import (
    beatnet_plus_baseline_checkpoint_names,
    beatnet_plus_baseline_checkpoint_path,
)

names = beatnet_plus_baseline_checkpoint_names()
path = beatnet_plus_baseline_checkpoint_path("baseline")
```

## Status

Current on 2026-10-09. The three files match the digests above and the
upstream files in the copy named above (checked with `sha256sum`).
