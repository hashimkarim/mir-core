# Packaged trained models

This directory contains the completed five-fold model candidates used by
BeatLab, ClassifierLab, SystemLab, and deployment applications. Checkpoints are
stored as:

```text
<model-family>/<target>/<condition>/checkpoints/seed_42_fold_<fold>.pt
```

Beat-tracking bundles attach both the original stock online postprocessors and
every completed tuned causal candidate under
`postprocessors/<candidate>/params.json`. Stock choices use `stock-*` names;
tuned choices use matching `tuned-*` names. The public id pattern is
`<kind>-<method>[-<variant>]`; historical experiment ids remain in each tuned
record as `source_id` provenance. Classifier bundles have no beat postprocessor
because their causal routing policy is embedded in each checkpoint.

The stable method ids are `stock-1d`/`tuned-1d`,
`stock-dbn`/`tuned-dbn`, and
`stock-particle-filter`/`tuned-particle-filter`. `tuned-1d` is present only for
bundles with the completed immediate causal-activation candidate; the older
12-frame past-snap candidate is not packaged as a runnable tuned option.

All bundles are deliberately marked `candidate`. Model and postprocessor
selection remains a separate scientific decision; the `default_postprocessor`
is only a convenient runnable default for validation smoke tests.

Use the stable Python API instead of constructing paths:

```python
from mir_core.checkpoints import (
    trained_checkpoint_path,
    trained_postprocessor_path,
)

checkpoint = trained_checkpoint_path(
    "beatnet", "candombe", "scratch", fold_index=2
)
tuned_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "tuned-dbn"
)
stock_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "stock-dbn"
)
classifier = trained_checkpoint_path(
    "classifier", "latin_router", "efficientat", fold_index=2
)
```

`TrainedModelBundle.postprocessors` contains both groups, while
`stock_postprocessors` and `tuned_postprocessors` expose them separately.
Choose the corresponding `stock-*` name explicitly to reproduce the original
behavior.

## Per-fold DBN settings

The four bundles of the routed system (`latin_general/scratch`,
`brid/finetune_latin_general`, `candombe/scratch` and
`salsa/finetune_latin_general`) also carry `tuned-dbn-per-fold`, which is
their default. It holds one joint-DBN parameter file per fold
(`postprocessors/tuned-dbn-per-fold/fold_<fold>.json`), selected on that
fold's validation songs only, so a test song never influences the setting it
is evaluated with. Pass the fold:

```python
parameters = trained_postprocessor_path(
    "beatnet", "salsa", "finetune_latin_general", fold_index=2
)
```

Calling without `fold_index` raises for a per-fold postprocessor; for a
shared one, `fold_index` is accepted and ignored, so fold-aware callers can
always pass it. `TrainedModelBundle.postprocessor_is_per_fold()` tells the two
apart, and `selection.json` records the validation scores of every fold.

The settings were selected on 6 October 2026 by scoring the settings that the
August searches had tried on live scores (the uncached streaming frontend of
the ported engine) and taking the highest joint RT-F1 at 70 ms per fold. The
August searches used stored features, whose frames are centred about 24 ms
later than the live ones. `tuned-dbn` remains the shared August selection, as
used by the system comparisons of August and 1 October 2026. The other three
bundles keep `tuned-dbn` as their default. The legacy combined stock catalog remains
available through `beatnet_stock_postprocessor_selection_path()`.

Every file is byte-bound by its bundle manifest. All current bundles use split
contract `e2e-537f350dbaf7e925`; consumers must select the same fold before
combining classifier and beat-tracking components.
