# Packaged trained models

This directory contains the completed five-fold model candidates used by
BeatLab, ClassifierLab, SystemLab, and deployment applications. Checkpoints are
stored as:

```text
<model-family>/<target>/<condition>/checkpoints/seed_42_fold_<fold>.pt
```

Every beat-tracking bundle attaches six online postprocessors: a stock and a
tuned choice for each of the three decoders.

| Decoder | Stock | Tuned |
| --- | --- | --- |
| Joint DBN | `stock-dbn` | `tuned-dbn` |
| 1D state space | `stock-1d` | `tuned-1d` |
| Particle filter | `stock-pf` | `tuned-pf` |

Stock choices are the original parameters and are the same for every fold
(`postprocessors/<id>/params.json`). Tuned choices hold one parameter file per
fold (`postprocessors/<id>/fold_<fold>.json`), each selected on that fold's
validation songs only, so a test song never influences the setting it is
evaluated with. `selection.json` in the same directory records the selected
parameters and validation scores of every fold, and `source-manifest.json` the
procedure. Classifier bundles have no beat postprocessor because their causal
routing policy is embedded in each checkpoint.

All bundles are deliberately marked `candidate`. `default_postprocessor` is
`tuned-dbn` for every beat-tracking bundle.

Use the stable Python API instead of constructing paths, and pass the fold:

```python
from mir_core.checkpoints import (
    trained_checkpoint_path,
    trained_postprocessor_path,
)

checkpoint = trained_checkpoint_path(
    "beatnet", "candombe", "scratch", fold_index=2
)
tuned_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "tuned-dbn", fold_index=2
)
stock_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "stock-dbn"
)
classifier = trained_checkpoint_path(
    "classifier", "latin_router", "efficientat", fold_index=2
)
```

A tuned postprocessor requires `fold_index` and raises without it. A stock one
accepts and ignores it, so fold-aware callers can always pass it.
`TrainedModelBundle.postprocessor_is_per_fold()` tells the two apart;
`postprocessors`, `stock_postprocessors` and `tuned_postprocessors` list them.
The legacy combined stock catalog remains available through
`beatnet_stock_postprocessor_selection_path()`.

## How the tuned settings were selected

On 6 October 2026, every setting that the August searches had tried for a
decoder, and that the ported decoder can run, was scored on live scores: each
fold's model on that fold's validation songs, through the uncached streaming
frontend of the ported engine. The setting with the highest joint RT-F1 at
70 ms was selected per fold.

This replaced the shared tuned settings of August, for two reasons. Those
searches scored settings on stored features, whose frames are centred about
24 ms later than those of the live frontend; the setting chosen for the salsa
specialist then announces beats too late in the running system. And a shared
setting is chosen from the validation songs of all folds, which are test
songs of other folds.

The frozen particle-filter contract of the port
(`mir_core.testing.particle_filter_contract`) covers the default parameter
profile, which is `stock-pf`. The `tuned-pf` settings differ from that profile
(for example in `lambda_b`, `lambda_d` and `ig_threshold`) and are not covered
by its frozen cases; the ported particle filter runs them, but its agreement
with the reference implementation is not established for them.

The shared August settings (`dbn-hybrid-joint`,
`1d-causal-activation-v2-hybrid-joint` and `particle-filter-fixed`) were
archived outside the repository and remain in the history up to commit
`d33adde`. The system comparisons of August and of 1 October 2026 used them.

Every file is byte-bound by its bundle manifest. All current bundles use split
contract `e2e-537f350dbaf7e925`; consumers must select the same fold before
combining classifier and beat-tracking components.
