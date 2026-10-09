# mir-core

![RitmoMaestro](https://shieldcn.dev/badge/RitmoMaestro-shared%20package-7289da.svg?variant=secondary&logo=lu:Package)
![status](https://shieldcn.dev/badge/status-active-22c55e.svg?variant=secondary&logo=lu:CircleCheck)
![stack](https://shieldcn.dev/badge/stack-Python-3776ab.svg?variant=secondary&logo=python)

`mir-core` is the shared Python package of RitmoMaestro, a system that predicts
the beats and downbeats of Latin music live and turns them into vibrations for
dancers. The package holds the model definitions, audio features, decoders,
evaluation metrics, the split engine, the native export code and the trained
model bundles. A decoder (called postprocessor in the code) turns the scores
of a model into beat and downbeat times. The training code and the apps import
this package and do not copy its logic.

## Quick start

The conda environment `MIR` (defined in `mir-environment/environment.yml`)
installs this package in editable mode. For a new environment:

```bash
conda activate MIR
cd /home/hashim/thesis-workspace/mir-core
pip install -e '.[native]'              # the extra adds onnx and onnxruntime
python -m mir_env.verify_installation   # checks the installed packages
```

Read a packaged model and one of its decoder settings through the API, not
through hand-built paths:

```python
from mir_core.checkpoints import trained_checkpoint_path, trained_postprocessor_path

checkpoint = trained_checkpoint_path("beatnet", "salsa", "scratch", fold_index=2)
tuned = trained_postprocessor_path("beatnet", "salsa", "scratch", "tuned-dbn", fold_index=2)
stock = trained_postprocessor_path("beatnet", "salsa", "scratch", "stock-dbn")
```

Checks that exist in this repository (run them in `MIR`, from the repository
root):

```bash
PYTHONPATH=. pytest -q tests/test_trained_checkpoints.py
python scripts/rebuild_trained_model_manifests.py --check
PYTHONPATH=. python -m mir_core.testing.particle_filter_contract --output build/particle-filter-contract.json
```

## Layout

| Path | Content |
| --- | --- |
| `mir_core/` | The package. See the table below. |
| `mir_env/` | `verify_installation.py`, the environment check. |
| `native-dsp/` | C++ resampler with a Rust interface, shared with the apps. See [native-dsp/README.md](native-dsp/README.md). |
| `tools/` | `port_validation_plugin.py` and `validation/`: checks that compare the ported code in the sibling repositories with this package. |
| `scripts/` | `rebuild_trained_model_manifests.py`: rebuilds or checks the bundle manifests. |
| `tests/` | The pytest suite (34 test modules). |
| `docs/` | Design documents, and a plan and a design of April 2026. |
| `third_party/` | `madmom-NOTICE.md`: attribution and licence of the madmom-derived parts. |
| `.github/workflows/` | CI for the particle-filter contract and the resampler, and the project sync. |
| `pyproject.toml` | Package metadata, version `0.2.0`, optional extra `native`. |
| `build/`, `mir_core.egg-info/` | Local build output. Git ignores both. |

Subpackages of `mir_core/`:

| Subpackage | Content |
| --- | --- |
| `beats/` | Beat experiment hashes (`btk-...`), paper presets, beat schema. |
| `checkpoints/` | Published baseline weights and the trained model bundles. |
| `classifier/` | Style classifier runtime, calibration and metrics. |
| `datasets/` | Dataset loaders and the annotation reader. |
| `evaluation/` | Beat, downbeat, tempo and real-time metrics. |
| `models/` | BeatNet, BeatNet+, Böck TCN, BEAST, SpecTNT and the classifier models. |
| `native/` | ONNX export and sessions for streaming and batch models. |
| `postprocessing/` | The decoders: DBN, 1D state space, particle filter, peak picking. |
| `preprocessing/` | Audio feature extractors. |
| `runtime/` | Concurrency helpers for live execution. |
| `splitting/` | Split plans that keep leakage groups together. |
| `testing/` | The particle-filter contract and its fixture. |
| `training/` | PyTorch Lightning modules and layer freezing. |
| `utils/` | Hashing and signal helpers. |
| `hub.py` | Model registry and `load_model`. |

`mir_core/experiments/` and `mir_core/models/beat_this/` exist as empty
folders without tracked files.

## How it connects

- It needs the conda environment from `mir-environment`. The madmom, BeatNet
  and PyTorch dependencies are defined there, not in `pyproject.toml`.
- These repositories import `mir_core` (checked with `grep` on 2026-10-09):
  `mir-train-hpc`, `mir-desktop-app`, `mir-embedded-pp`, `mir-webapp`,
  `mir-android-app` and `mir-embedded-hmi`.
- `mir-train-hpc` trains the models. The trained bundles in
  `mir_core/checkpoints/trained/` come from its runs.
- The package itself, its tests and `scripts/` import no sibling repository.
  The files under `tools/` do: they import `beatlab`, `classifierlab` and
  `splitplan` from `mir-train-hpc` and `mir_desktop_app` from
  `mir-desktop-app`, so they need those checkouts on `PYTHONPATH`.

## Status

Status on 2026-10-09.

Current:

- Seven BeatNet bundles and two classifier bundles are packaged. All use
  split contract `e2e-537f350dbaf7e925` and are marked `candidate`.
- Every BeatNet bundle has six decoder settings: `stock-pf`, `tuned-pf`,
  `stock-dbn`, `tuned-dbn`, `stock-1d`, `tuned-1d`. A stock setting is one
  file for all folds. A tuned setting has one file per fold. The default is
  `tuned-dbn`.
- `origin/main` holds this state at commit `f661db8` (7 October 2026).

Kept as history:

- DanceBeat (the dance-one predictor) is dropped. Its code is only on the
  branch `archive/dancebeat`. `tests/test_native_beatnet.py` checks that the
  archived model names are rejected.
- The shared tuned settings of August 2026 were replaced by per-fold settings
  on 6 October 2026. They remain in the Git history up to commit `d33adde`.
- `docs/superpowers/` holds the plan and the design of the reorganisation of
  April 2026.

## More documentation

- [mir_core/checkpoints/trained/README.md](mir_core/checkpoints/trained/README.md):
  bundle layout, decoder settings and how the tuned settings were selected.
- [mir_core/beats/experiments/README.md](mir_core/beats/experiments/README.md):
  experiment hashes and paper presets.
- Baseline weights: [BeatNet](mir_core/checkpoints/beatnet/README.md),
  [BeatNet+](mir_core/checkpoints/beatnet_plus/README.md),
  [BEAST](mir_core/checkpoints/beast/README.md),
  [Böck TCN](mir_core/checkpoints/bocktcn/README.md).
- [docs/native-runtime.md](docs/native-runtime.md): native execution, what is
  exported and what is still missing.
- [docs/native-batch-frontends.md](docs/native-batch-frontends.md): audio
  frontends of the batch models as ONNX.
- [docs/particle-filter-rng-contract.md](docs/particle-filter-rng-contract.md):
  the frozen particle-filter contract.
- [native-dsp/README.md](native-dsp/README.md): the shared resampler.
- [third_party/madmom-NOTICE.md](third_party/madmom-NOTICE.md): madmom
  attribution and the licence of the packaged Böck TCN pickles.
