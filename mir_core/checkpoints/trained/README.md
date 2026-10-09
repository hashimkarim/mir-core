# Packaged trained models

This folder holds the trained five-fold model bundles that BeatLab,
ClassifierLab, SystemLab and the apps load. A bundle is one model family,
target and training condition. It has five fold checkpoints and a manifest. A
beat-tracking bundle also has six decoder settings. A decoder (postprocessor
in the code) turns the scores of the model into beat and downbeat times.

## Contents

| Path | Bundle | Source experiment and attempt |
| --- | --- | --- |
| `beatnet/latin_general/scratch/` | General Latin model | `btk-531c2f27e2d4e6dd`, `beat-final-online-v3-20260811T1715Z-latin-general` |
| `beatnet/brid/scratch/` | BRID specialist, trained from scratch | `btk-16d65fb5265baec2`, `beat-final-online-v3-20260811T1715Z-specialists-scratch` |
| `beatnet/brid/finetune_latin_general/` | BRID specialist, fine-tuned from the general model | `btk-331955aceed37404`, `beat-final-online-v3-20260811T1715Z-specialists-finetune` |
| `beatnet/candombe/scratch/` | Candombe specialist, trained from scratch | `btk-519cedc1e0634d01`, `beat-final-online-v3-20260811T1715Z-specialists-scratch` |
| `beatnet/candombe/finetune_latin_general/` | Candombe specialist, fine-tuned from the general model | `btk-db25e6e89ff49b3f`, `beat-final-online-v3-20260811T1715Z-specialists-finetune` |
| `beatnet/salsa/scratch/` | Salsa specialist, trained from scratch | `btk-745d23a56687ebe0`, `beat-final-online-v3-20260811T1715Z-specialists-scratch` |
| `beatnet/salsa/finetune_latin_general/` | Salsa specialist, fine-tuned from the general model | `btk-013273597182a11d`, `beat-final-online-v3-20260811T1715Z-specialists-finetune` |
| `beatnet/stock/baseline/postprocessors.json` | Stock decoder settings of the published BeatNet model | not a trained bundle |
| `classifier/latin_router/yamnet/` | Style classifier on YAMNet embeddings | `clf-839371022f8f797e`, `classifier-20260813T225050Z-0d27da6155` |
| `classifier/latin_router/efficientat/` | Style classifier on EfficientAT embeddings | `clf-b2fdbc86520b0d6c`, `classifier-20260814T045441Z-c14dd90c29` |

Inside a bundle:

| Path | Content |
| --- | --- |
| `manifest.json` | Schema `mir.trained-model-bundle/v2`. Binds every file by SHA-256 and size. |
| `checkpoints/seed_42_fold_<fold>.pt` | One checkpoint per fold, folds 0 to 4. |
| `postprocessors/<id>/params.json` | A stock decoder setting, the same for every fold. |
| `postprocessors/<id>/fold_<fold>.json` | A tuned decoder setting, one file per fold. |
| `postprocessors/<id>/selection.json` | Selected parameters and validation scores of every fold. |
| `postprocessors/<id>/source-manifest.json` | The selection procedure. |

Decoder setting ids:

| Decoder | Stock | Tuned |
| --- | --- | --- |
| Joint DBN | `stock-dbn` | `tuned-dbn` |
| 1D state space | `stock-1d` | `tuned-1d` |
| Particle filter | `stock-pf` | `tuned-pf` |

Stock settings are the original parameters. Each tuned file was selected on
that fold's validation songs only, so a test song never influences the setting
it is evaluated with. Classifier bundles have no beat decoder because their
causal routing policy is embedded in each checkpoint.

## How to use it

Use the Python API instead of constructing paths, and pass the fold:

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

Every file is byte-bound by its bundle manifest. All current bundles use split
contract `e2e-537f350dbaf7e925`. Consumers must select the same fold before
combining classifier and beat-tracking components.

After a change to a bundle, rebuild or check the manifests from the repository
root:

```bash
python scripts/rebuild_trained_model_manifests.py --check
```

## How the tuned settings were selected

On 6 October 2026, every setting that the August searches had tried for a
decoder, and that the ported decoder can run, was scored on live scores: each
fold's model on that fold's validation songs, through the uncached streaming
frontend of the ported engine. The setting with the highest joint RT-F1 at
70 ms was selected per fold.

Note of 2026-10-09: the `tuned-1d` settings were selected again on 7 October
2026 (commit `f661db8`). The selection of 6 October could consider 708 of the
1,330 valid 1D settings of the August searches. The ported decoder now runs
all of them, and the selection was repeated over the full list with the same
data, scoring and rule. The `source-manifest.json` files record `selected_at`
`2026-10-07` for `tuned-1d` and `2026-10-06` for `tuned-dbn` and `tuned-pf`.

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

## Status

Current on 2026-10-09. All bundles are deliberately marked `candidate`.
`default_postprocessor` is `tuned-dbn` for every beat-tracking bundle. The
source of truth for the scores of these models is W&B (entity
`hashimkarim-tu-delft`, projects `beatlab` and `classifierlab`, group equal to
the experiment hash) and the run folders on Prometheus; the selection records
are in
`thesis-docs/operations/validation/native-routing-20261001/recheck-20261006/all-decoders-per-fold/`.
