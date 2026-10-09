# mir-core Task Submodule Reorganisation — Design Spec

> **Status:** historical record (design approved on 2026-04-23)\
> **Date:** 2026-04-23\
> **Source of truth:** this file for the design as approved; the code in `mir_core/beats/`, `mir_core/classifier/` and `mir_core/utils/hashing.py` for the layout that exists today

**Scope:** `mir-core` package

---

## Goal

Reorganise `mir_core` so that task-specific concerns (experiment presets, evaluation metrics) live under named task submodules (`beats`, `classifier`), while shared utilities live in `mir_core.utils`. Every experiment hash encodes its task type via a 3-letter prefix, making it unambiguously identifiable without consulting any registry.

---

## Hash Identity Scheme

### Format

```
{task}-{16-hex-chars}

btk-a14aef639058a4c7    ← beat tracking experiment
clf-d9b3f21e88ac4d07    ← classifier experiment
```

Total length: 20 characters, always. The prefix is the authoritative task discriminator — no registry lookup needed to know what a hash refers to.

### Task prefixes

| Prefix | Task |
|---|---|
| `btk` | Beat tracking |
| `clf` | Music classifier |

### How hashes are computed

`stable_hash()` in `mir_core.utils.hashing` remains a pure SHA-256 utility (no task awareness). Each task submodule wraps it with its own `experiment_hash()`:

```python
# mir_core/beats/experiments/__init__.py
def experiment_hash(config: dict) -> str:
    return "btk-" + stable_hash(config)

# mir_core/classifier/experiments/__init__.py
def experiment_hash(config: dict) -> str:
    return "clf-" + stable_hash(config)
```

Preset JSON filenames include the full prefixed hash: `btk-a14aef639058a4c7.json`.

---

## New Package Structure

```
mir_core/
  utils/
    __init__.py
    hashing.py              ← canonical_json, stable_digest, stable_hash (moved from experiments/)

  beats/
    __init__.py
    experiments/
      __init__.py           ← Preset, load_presets, PRESETS, PRESETS_BY_KEY,
      presets.py            ←   get_by_hash, get_by_key, experiment_hash("btk-…")
      presets/
        btk-a14aef639058a4c7.json   ← rapini2024_salsaset_beatnet   (renamed)
        btk-875884b5ff6ad78a.json   ← rapini2024_salsaset_bocktcn   (renamed)  
        btk-8fec2f384db1fde8.json   ← heydari2021_beatnet           (renamed)
        btk-b75eba7a86c16914.json   ← davies2019_bocktcn            (renamed)
      README.md                     ← updated with new hash format
    evaluation/
      __init__.py           ← re-exports compute_beat_metrics, compute_downbeat_metrics
                               from mir_core.evaluation.metrics (no physical move yet)

  classifier/
    __init__.py
    experiments/
      __init__.py           ← same Preset/registry machinery, experiment_hash("clf-…")
      presets.py
      presets/              ← empty, .gitkeep (ready for classifier presets)
    evaluation/
      __init__.py           ← stub: accuracy, macro_f1, confusion_matrix (to be filled)

  experiments/              ← REMOVED entirely
  evaluation/               ← stays (physical home of beat metrics for now)
  models/                   ← unchanged
  datasets/                 ← unchanged
  preprocessing/            ← unchanged
  postprocessing/           ← unchanged
  training/                 ← unchanged
```

---

## File-by-File Changes

### New files

| File | Purpose |
|---|---|
| `mir_core/utils/__init__.py` | Package init |
| `mir_core/utils/hashing.py` | `canonical_json`, `stable_digest`, `stable_hash` (moved verbatim) |
| `mir_core/beats/__init__.py` | Package init |
| `mir_core/beats/experiments/__init__.py` | Beat preset registry + `experiment_hash()` |
| `mir_core/beats/experiments/presets.py` | `Preset` dataclass + `load_presets()` (copied from current `presets.py`, updated dir path) |
| `mir_core/beats/evaluation/__init__.py` | Re-exports beat metrics |
| `mir_core/classifier/__init__.py` | Package init |
| `mir_core/classifier/experiments/__init__.py` | Classifier preset registry + `experiment_hash()` |
| `mir_core/classifier/experiments/presets.py` | Same machinery as beats, `clf-` prefix |
| `mir_core/classifier/experiments/presets/.gitkeep` | Ready for first classifier preset |
| `mir_core/classifier/evaluation/__init__.py` | Stubs: `accuracy`, `macro_f1`, `confusion_matrix` |

### Renamed files (preset JSONs)

| Old name | New name |
|---|---|
| `mir_core/experiments/presets/a14aef639058a4c7.json` | `mir_core/beats/experiments/presets/btk-a14aef639058a4c7.json` |
| `mir_core/experiments/presets/875884b5ff6ad78a.json` | `mir_core/beats/experiments/presets/btk-875884b5ff6ad78a.json` |
| `mir_core/experiments/presets/8fec2f384db1fde8.json` | `mir_core/beats/experiments/presets/btk-8fec2f384db1fde8.json` |
| `mir_core/experiments/presets/b75eba7a86c16914.json` | `mir_core/beats/experiments/presets/btk-b75eba7a86c16914.json` |

Each renamed JSON also has its internal `"hash"` field updated to the prefixed form (e.g. `"hash": "btk-a14aef639058a4c7"`).

### Deleted

| File/Dir | Reason |
|---|---|
| `mir_core/experiments/__init__.py` | Replaced by `utils/hashing.py` + task submodules |
| `mir_core/experiments/presets.py` | Replaced by `beats/experiments/presets.py` |
| `mir_core/experiments/presets/` | Moved to `beats/experiments/presets/` |
| `mir_core/experiments/README.md` | Replaced by updated `beats/experiments/README.md` |

### Modified

| File | Change |
|---|---|
| `mir_core/__init__.py` | Update top-level re-exports: `stable_hash` etc. now from `mir_core.utils.hashing` |
| `mir_core/evaluation/metrics.py` | No changes — stays as physical home of beat metrics |
| `tests/test_bocktcn_arch.py` | No changes needed |
| `tests/test_presets.py` | Update imports to `mir_core.beats.experiments`; update hash format expectations |

---

## Updated Import Paths

### Before

```python
from mir_core.experiments import stable_hash, get_by_hash, get_by_key, PRESETS
from mir_core import stable_hash
```

### After

```python
from mir_core.utils.hashing import stable_hash
from mir_core.beats.experiments import get_by_hash, get_by_key, PRESETS, experiment_hash
from mir_core import stable_hash   # ← still works via __init__.py re-export
```

### In `beatlab/train_beat.py`

```python
# Before
from mir_core.experiments import get_by_hash, get_by_key
from .config import ..., stable_hash, ...

# After
from mir_core.beats.experiments import get_by_hash, get_by_key, experiment_hash
from mir_core.utils.hashing import stable_hash
```

---

## `Preset` Dataclass (unchanged)

The `Preset` dataclass definition is identical in both task submodules — fields: `key`, `hash`, `citation`, `config`, `notes`. The `hash` field now contains the prefixed form (`"btk-a14aef639058a4c7"`).

`load_presets()` is also identical in both submodules — it scans the local `presets/` directory and populates the registry. Each submodule's `experiment_hash()` function uses its own prefix.

---

## `mir_core.classifier.evaluation` — Stubs

The classifier evaluation module is created as stubs now (not implemented) so the import path is stable before classifierlab is built:

```python
# mir_core/classifier/evaluation/__init__.py
def accuracy(predictions, targets) -> float:
    raise NotImplementedError("classifier evaluation not yet implemented")

def macro_f1(predictions, targets) -> float:
    raise NotImplementedError("classifier evaluation not yet implemented")

def confusion_matrix(predictions, targets):
    raise NotImplementedError("classifier evaluation not yet implemented")
```

---

## Tests

### Updated: `tests/test_presets.py`

- Import from `mir_core.beats.experiments` instead of `mir_core.experiments`
- `PRESETS_DIR` now points to `mir_core/beats/experiments/presets/`
- Hash format check updated: `f.stem` must match `btk-[0-9a-f]{16}` pattern
- `test_preset_file_schema` validates `"hash"` field starts with `"btk-"`

### New: `tests/test_beats_experiments.py`

- `experiment_hash(config)` returns string starting with `"btk-"` and total length 20
- `get_by_hash("btk-a14aef639058a4c7")` returns correct preset
- `get_by_key("rapini2024_salsaset_beatnet")` returns correct preset

### New: `tests/test_utils_hashing.py`

- `stable_hash` is deterministic
- `stable_hash` is order-independent (different key order → same hash)
- `stable_hash` length parameter works

---

## Backward Compatibility

`mir_core.__init__.py` continues to re-export `stable_hash`, `stable_digest`, `canonical_json` at the top level — imported from `mir_core.utils.hashing`. Code that does `from mir_core import stable_hash` keeps working unchanged.

`mir_core.experiments` is **not** kept as a compatibility shim — it is fully removed. Consumers (`beatlab/train_beat.py`) are updated as part of this change.

---

## What Does NOT Change

- `mir_core/models/` — model code is architecture, not task schema. `bock_tcn`, `beatnet`, `classifier/` stay where they are.
- `mir_core/datasets/` — shared loading utilities, unchanged.
- `mir_core/preprocessing/` — unchanged.
- `mir_core/postprocessing/` — unchanged.
- `mir_core/training/` — unchanged.
- `mir_core/evaluation/metrics.py` — stays as physical home; `mir_core/beats/evaluation/` re-exports from it.

---

## Out of Scope

- Moving model files into task submodules (`beats/models/`, `classifier/models/`)
- Implementing classifier evaluation metrics (stubbed, not implemented)
- Adding classifier presets (directory created, no JSON files yet)
- Any changes to `mir-train-hpc` beyond updating import paths in `train_beat.py`
