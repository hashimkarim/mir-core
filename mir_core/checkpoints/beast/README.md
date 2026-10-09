# BEAST baseline checkpoint

This folder packages the published BEAST weights, used as the baseline of the
BEAST model family.

## Contents

| Selector | Packaged file | Upstream file | SHA-256 |
| --- | --- | --- | --- |
| `baseline` | `baseline.pt` | `BEAST_param.pt` | `e77f272f1c1f68521958d3947b1d7b72616fda2281755417b96dd6d3f7ddddc4` |

`__init__.py` holds the selector table and the pinned digest.

`baseline.pt` is the published BEAST `BEAST_param.pt` checkpoint downloaded
from the upstream BEAST repository link:

https://drive.google.com/file/d/17yiv4cIsI1rBL8vUAtAVJN1pUPXQOAhl/view?usp=sharing

The upstream BEAST evaluation code loads `torch.load(path)["state_dict"]`.

## How to use it

```python
from mir_core.checkpoints import beast_baseline_checkpoint_path

path = beast_baseline_checkpoint_path("baseline")
```

## Status

Current on 2026-10-09. The file matches the digest above (checked with
`sha256sum`). The file was not compared with the one behind the download link
(not verified, 2026-10-09).
