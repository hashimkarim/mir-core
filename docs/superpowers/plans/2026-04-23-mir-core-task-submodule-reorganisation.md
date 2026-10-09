# mir-core Task Submodule Reorganisation Implementation Plan

> **Status:** historical record\
> **Date:** 2026-04-23\
> **Source of truth:** this file for the plan as written; the code in `mir_core/beats/`, `mir_core/classifier/` and `mir_core/utils/hashing.py` for the layout that exists today

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise `mir_core` into task submodules (`beats/`, `classifier/`) with `btk-`/`clf-`-prefixed experiment hashes, removing the flat `mir_core/experiments/` package.

**Architecture:** Shared hashing utilities move to `mir_core/utils/hashing.py`. Beat-specific code (preset registry, evaluation re-exports) moves to `mir_core/beats/`. A `mir_core/classifier/` stub is created with the same structure. All four existing preset JSONs are renamed with the `btk-` prefix. `mir_core/experiments/` is deleted and all consumers updated.

**Tech Stack:** Python 3.11+, pytest, PyYAML, two git repos (`mir-core` at `~/thesis-workspace/mir-core`, `mir-train-hpc` at `~/thesis-workspace/mir-train-hpc`).

---

## File Map

### mir-core

| Action | Path |
|--------|------|
| Create | `mir_core/utils/__init__.py` |
| Create | `mir_core/utils/hashing.py` |
| Create | `mir_core/beats/__init__.py` |
| Create | `mir_core/beats/experiments/__init__.py` |
| Create | `mir_core/beats/experiments/presets.py` |
| Create | `mir_core/beats/experiments/presets/btk-{hash}.json` × 4 |
| Create | `mir_core/beats/experiments/README.md` |
| Create | `mir_core/beats/evaluation/__init__.py` |
| Create | `mir_core/classifier/__init__.py` |
| Create | `mir_core/classifier/experiments/__init__.py` |
| Create | `mir_core/classifier/experiments/presets.py` |
| Create | `mir_core/classifier/experiments/presets/.gitkeep` |
| Create | `mir_core/classifier/evaluation/__init__.py` |
| Delete | `mir_core/experiments/` (entire directory) |
| Modify | `mir_core/__init__.py` |
| Modify | `tests/test_presets.py` |
| Create | `tests/test_utils_hashing.py` |
| Create | `tests/test_beats_experiments.py` |

### mir-train-hpc

| Action | Path |
|--------|------|
| Modify | `beatlab/config.py` |
| Modify | `beatlab/train_beat.py` |

---

## Task 1: Create `mir_core/utils/hashing.py`

**Files:**
- Create: `mir_core/utils/__init__.py`
- Create: `mir_core/utils/hashing.py`
- Create: `tests/test_utils_hashing.py`

- [ ] **Step 1.1 — Write failing tests**

```python
# tests/test_utils_hashing.py
"""Tests for mir_core.utils.hashing."""
from mir_core.utils.hashing import canonical_json, stable_digest, stable_hash


def test_stable_hash_is_deterministic():
    config = {"a": 1, "b": 2}
    assert stable_hash(config) == stable_hash(config)


def test_stable_hash_is_order_independent():
    config_a = {"a": 1, "b": 2}
    config_b = {"b": 2, "a": 1}
    assert stable_hash(config_a) == stable_hash(config_b)


def test_stable_hash_default_length():
    result = stable_hash({"x": 1})
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_stable_hash_custom_length():
    result = stable_hash({"x": 1}, length=8)
    assert len(result) == 8


def test_stable_digest_full_length():
    result = stable_digest({"x": 1})
    assert len(result) == 64  # full SHA-256 hex


def test_canonical_json_sorts_keys():
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b


def test_different_configs_produce_different_hashes():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
```

- [ ] **Step 1.2 — Run tests to confirm failure**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_utils_hashing.py -v
```

Expected: `ModuleNotFoundError: No module named 'mir_core.utils'`

- [ ] **Step 1.3 — Create `mir_core/utils/__init__.py`**

```python
# mir_core/utils/__init__.py
"""Shared utilities for mir-core."""
```

- [ ] **Step 1.4 — Create `mir_core/utils/hashing.py`**

```python
# mir_core/utils/hashing.py
"""Stable hashing utilities for experiment identity.

Functions:
    canonical_json  — deterministic JSON serialisation
    stable_digest   — full SHA-256 hex of any JSON-serialisable value
    stable_hash     — first N hex chars of SHA-256 (default 16)
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialise a value to deterministic JSON (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(value: Any, length: int | None = None) -> str:
    """Return a SHA-256 hex digest for a JSON-serialisable value."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the raw experiment hash for a config mapping (no task prefix).

    The returned string is `length` lowercase hex characters. Use
    mir_core.beats.experiments.experiment_hash() or
    mir_core.classifier.experiments.experiment_hash() to get a task-prefixed hash.
    """
    return stable_digest(dict(config), length=length)
```

- [ ] **Step 1.5 — Run tests**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_utils_hashing.py -v
```

Expected: 7 tests `PASSED`.

- [ ] **Step 1.6 — Run full suite to confirm no regressions**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/ -v
```

Expected: all existing tests still pass (`mir_core.experiments` still intact).

- [ ] **Step 1.7 — Commit**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/utils/__init__.py mir_core/utils/hashing.py tests/test_utils_hashing.py
git commit -m "feat: add mir_core.utils.hashing with canonical_json, stable_digest, stable_hash"
```

---

## Task 2: Create `mir_core/beats/experiments/` skeleton

**Files:**
- Create: `mir_core/beats/__init__.py`
- Create: `mir_core/beats/experiments/__init__.py`
- Create: `mir_core/beats/experiments/presets.py`
- Create: `mir_core/beats/experiments/presets/.gitkeep`
- Create: `tests/test_beats_experiments.py`

- [ ] **Step 2.1 — Write failing tests**

```python
# tests/test_beats_experiments.py
"""Tests for mir_core.beats.experiments."""
from mir_core.beats.experiments import (
    Preset,
    PRESETS,
    PRESETS_BY_KEY,
    experiment_hash,
    get_by_hash,
    get_by_key,
)


def test_experiment_hash_has_btk_prefix():
    config = {"model": {"name": "bock_tcn"}, "seed": 42}
    h = experiment_hash(config)
    assert h.startswith("btk-"), f"Expected 'btk-' prefix, got: {h}"


def test_experiment_hash_total_length():
    config = {"model": {"name": "bock_tcn"}}
    h = experiment_hash(config)
    assert len(h) == 20, f"Expected length 20, got {len(h)}: {h}"


def test_experiment_hash_hex_suffix():
    config = {"model": {"name": "bock_tcn"}}
    h = experiment_hash(config)
    suffix = h[4:]  # strip "btk-"
    assert all(c in "0123456789abcdef" for c in suffix), f"Non-hex suffix: {suffix}"


def test_experiment_hash_is_deterministic():
    config = {"model": {"name": "bock_tcn"}, "seed": 42}
    assert experiment_hash(config) == experiment_hash(config)


def test_experiment_hash_is_order_independent():
    config_a = {"a": 1, "b": 2}
    config_b = {"b": 2, "a": 1}
    assert experiment_hash(config_a) == experiment_hash(config_b)


def test_preset_dataclass_fields():
    p = Preset(
        key="test_key",
        hash="btk-abc123def456abcd",
        citation="Author 2024",
        config={"experiment": {"name": "test"}},
        notes=["note one"],
    )
    assert p.key == "test_key"
    assert p.hash == "btk-abc123def456abcd"
    assert isinstance(p.config, dict)
    assert isinstance(p.notes, list)


def test_load_presets_returns_dict():
    from mir_core.beats.experiments.presets import load_presets
    result = load_presets()
    assert isinstance(result, dict)


def test_presets_registry_is_dict():
    assert isinstance(PRESETS, dict)


def test_get_by_hash_unknown_returns_none():
    assert get_by_hash("btk-0000000000000000") is None


def test_get_by_key_unknown_returns_none():
    assert get_by_key("nonexistent_key") is None
```

- [ ] **Step 2.2 — Run tests to confirm failure**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py -v
```

Expected: `ModuleNotFoundError: No module named 'mir_core.beats'`

- [ ] **Step 2.3 — Create `mir_core/beats/__init__.py`**

```python
# mir_core/beats/__init__.py
"""Beat tracking submodule — experiments, evaluation."""
```

- [ ] **Step 2.4 — Create `mir_core/beats/experiments/presets.py`**

```python
# mir_core/beats/experiments/presets.py
"""Beat tracking experiment preset registry.

Each preset is stored as a JSON file in presets/ named by its full hash
(e.g. btk-a14aef639058a4c7.json). The filename stem IS the experiment hash.

Adding a new preset:
  1. Build the config dict with raw (unexpanded) env var strings.
  2. Compute hash (replace my_preset.json with your file):
     python -c "
     from mir_core.beats.experiments import experiment_hash
     import json
     print(experiment_hash(json.load(open('my_preset.json'))['config']))
     "
  3. Save as presets/{hash}.json with key, hash, citation, notes, config fields.
  4. Commit. No Python changes required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESETS_DIR: Path = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class Preset:
    key: str          # human identifier, e.g. "rapini2024_salsaset_beatnet"
    hash: str         # experiment_hash(config) — equals the JSON filename stem
    citation: str     # full citation string
    config: dict[str, Any]  # complete config (unexpanded env vars)
    notes: list[str]  # discrepancy notes and caveats


def load_presets() -> dict[str, Preset]:
    """Scan presets/ and load all .json files into a hash-keyed registry."""
    registry: dict[str, Preset] = {}
    if not PRESETS_DIR.is_dir():
        return registry
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            preset = Preset(
                key=data["key"],
                hash=path.stem,  # filename stem IS the hash (btk-…)
                citation=data["citation"],
                config=data["config"],
                notes=data.get("notes", []),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Malformed preset file {path.name}: {exc}") from exc
        registry[preset.hash] = preset
    return registry


# Module-level registry — loaded once at import time.
PRESETS: dict[str, Preset] = load_presets()

# Convenience lookup by human key (e.g. "rapini2024_salsaset_beatnet").
PRESETS_BY_KEY: dict[str, Preset] = {p.key: p for p in PRESETS.values()}


def get_by_hash(experiment_hash: str) -> Preset | None:
    """Return preset for a given experiment hash, or None."""
    return PRESETS.get(experiment_hash)


def get_by_key(key: str) -> Preset | None:
    """Return preset for a given human key, or None."""
    return PRESETS_BY_KEY.get(key)
```

- [ ] **Step 2.5 — Create `mir_core/beats/experiments/__init__.py`**

```python
# mir_core/beats/experiments/__init__.py
"""Beat tracking experiment registry.

All beat experiment hashes are prefixed with 'btk-' (20 chars total).

Usage:
    from mir_core.beats.experiments import experiment_hash, get_by_hash, get_by_key, PRESETS
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mir_core.utils.hashing import stable_hash
from .presets import (
    Preset,
    PRESETS,
    PRESETS_BY_KEY,
    load_presets,
    get_by_hash,
    get_by_key,
)

TASK_PREFIX = "btk"


def experiment_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the canonical beat tracking experiment hash.

    Format: 'btk-{16-hex-chars}' — always 20 characters.
    The 'btk' prefix identifies this as a beat tracking experiment.
    Pass the unexpanded config (env vars not substituted) for
    machine-independent hashes.
    """
    return f"{TASK_PREFIX}-{stable_hash(config, length=length)}"


__all__ = [
    "TASK_PREFIX",
    "experiment_hash",
    "Preset",
    "PRESETS",
    "PRESETS_BY_KEY",
    "load_presets",
    "get_by_hash",
    "get_by_key",
]
```

- [ ] **Step 2.6 — Create `mir_core/beats/experiments/presets/` directory**

```bash
cd ~/thesis-workspace/mir-core
mkdir -p mir_core/beats/experiments/presets
touch mir_core/beats/experiments/presets/.gitkeep
```

- [ ] **Step 2.7 — Run tests**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py -v
```

Expected: all 10 tests `PASSED`. (`PRESETS` is empty — JSON files not moved yet — so tests that only check the registry type/returns-None pass vacuously.)

- [ ] **Step 2.8 — Commit**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/beats/ tests/test_beats_experiments.py
git commit -m "feat: add mir_core.beats.experiments skeleton with btk- prefixed experiment_hash"
```

---

## Task 3: Move and rename preset JSON files

**Files:**
- Create: `mir_core/beats/experiments/presets/btk-{hash}.json` × 4
- Create: `mir_core/beats/experiments/README.md`

- [ ] **Step 3.1 — Run the rename script**

```bash
cd ~/thesis-workspace/mir-core
python - <<'EOF'
import json
from pathlib import Path

old_dir = Path("mir_core/experiments/presets")
new_dir = Path("mir_core/beats/experiments/presets")
new_dir.mkdir(parents=True, exist_ok=True)

for old_path in sorted(old_dir.glob("*.json")):
    old_hash = old_path.stem
    new_hash = f"btk-{old_hash}"
    data = json.loads(old_path.read_text())
    data["hash"] = new_hash        # update internal hash field
    new_path = new_dir / f"{new_hash}.json"
    new_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Created: {new_path}")
EOF
```

Expected output (4 lines):
```
Created: mir_core/beats/experiments/presets/btk-875884b5ff6ad78a.json
Created: mir_core/beats/experiments/presets/btk-8fec2f384db1fde8.json
Created: mir_core/beats/experiments/presets/btk-a14aef639058a4c7.json
Created: mir_core/beats/experiments/presets/btk-b75eba7a86c16914.json
```

- [ ] **Step 3.2 — Verify the registry loads 4 presets**

```bash
cd ~/thesis-workspace/mir-core
python -c "
from mir_core.beats.experiments import PRESETS, PRESETS_BY_KEY
print('Loaded', len(PRESETS), 'presets:')
for key, p in PRESETS_BY_KEY.items():
    print(f'  {p.hash}  {key}')
"
```

Expected: 4 lines, all hashes starting with `btk-`.

- [ ] **Step 3.3 — Write the updated README**

```bash
cd ~/thesis-workspace/mir-core
python -c "
from mir_core.beats.experiments import PRESETS_BY_KEY
for key in ['rapini2024_salsaset_beatnet','rapini2024_salsaset_bocktcn','heydari2021_beatnet','davies2019_bocktcn']:
    p = PRESETS_BY_KEY[key]
    print(f'| \`{p.hash}\` | \`{key}\` |')
"
```

Note the printed hashes, then write `mir_core/beats/experiments/README.md`:

```markdown
# mir_core/beats/experiments

Beat tracking experiment identity utilities and canonical paper preset registry.

## Hashing

`experiment_hash(config_dict)` — returns `btk-{SHA-256[:16]}`.  
Format: always 20 characters (`btk-` + 16 hex chars).  
Configs must NOT have env vars expanded before hashing.

## Presets

Each file in `presets/` is a self-contained canonical experiment definition.
The **filename stem is the experiment hash** (e.g. `btk-a14aef639058a4c7.json`).

### Available presets

| Hash | Key | Paper |
|------|-----|-------|
| `btk-a14aef639058a4c7` | `rapini2024_salsaset_beatnet` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-875884b5ff6ad78a` | `rapini2024_salsaset_bocktcn` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-8fec2f384db1fde8` | `heydari2021_beatnet` | Heydari et al. 2021 (ISMIR) |
| `btk-b75eba7a86c16914` | `davies2019_bocktcn` | Davies & Böck 2019 (EUSIPCO) |

Run `python -c "from mir_core.beats.experiments import PRESETS_BY_KEY; print(list(PRESETS_BY_KEY))"` to list current keys.

### Python API

```python
from mir_core.beats.experiments import (
    PRESETS, PRESETS_BY_KEY, get_by_hash, get_by_key, experiment_hash
)

preset = get_by_key("rapini2024_salsaset_beatnet")
preset.hash        # "btk-a14aef639058a4c7"
preset.citation    # full citation string
preset.config      # complete config dict (unexpanded)
preset.notes       # list of discrepancy / methodology notes

# Compute hash for a new config
h = experiment_hash(config_dict)  # "btk-{16 hex chars}"
```

### Adding a preset

1. Build the config dict with raw (unexpanded) env var strings.
2. Compute hash:
   ```bash
   python -c "from mir_core.beats.experiments import experiment_hash; import json; print(experiment_hash(json.load(open('my_preset.json'))['config']))"
   ```
3. Write `presets/{hash}.json` with `key`, `hash`, `citation`, `notes`, `config` fields.
4. Commit. No Python changes required.

## BeatNet preprocessing discrepancy

**Paper text** (Heydari et al. 2021): 93 ms window, 46 ms hop → ~22 fps.  
**Official implementation** (all published results): win=1408 samples (64 ms), hop=441 samples (20 ms) → **50 fps**.  
All BeatNet presets use 50 fps to match published numbers.
```

- [ ] **Step 3.4 — Run tests**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py -v
```

Expected: all 10 tests `PASSED` (including preset-count-dependent ones now that JSONs are present).

- [ ] **Step 3.5 — Commit**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/beats/experiments/presets/ mir_core/beats/experiments/README.md
git commit -m "feat: add four btk-prefixed beat preset JSON files and README

Renamed from bare hashes (a14aef…) to btk-prefixed (btk-a14aef…).
Internal 'hash' field updated to match filename stem."
```

---

## Task 4: Create `mir_core/beats/evaluation/`

**Files:**
- Create: `mir_core/beats/evaluation/__init__.py`

- [ ] **Step 4.1 — Write failing import test**

Add this to `tests/test_beats_experiments.py`:

```python
def test_beats_evaluation_imports():
    from mir_core.beats.evaluation import (
        compute_beat_metrics,
        compute_downbeat_metrics,
    )
    assert callable(compute_beat_metrics)
    assert callable(compute_downbeat_metrics)
```

- [ ] **Step 4.2 — Run to confirm failure**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py::test_beats_evaluation_imports -v
```

Expected: `ModuleNotFoundError: No module named 'mir_core.beats.evaluation'`

- [ ] **Step 4.3 — Create `mir_core/beats/evaluation/__init__.py`**

```python
# mir_core/beats/evaluation/__init__.py
"""Beat tracking evaluation metrics.

Re-exports from mir_core.evaluation.metrics. Import from here for
task-scoped imports:

    from mir_core.beats.evaluation import compute_beat_metrics
"""
from mir_core.evaluation.metrics import (
    compute_beat_metrics,
    compute_downbeat_metrics,
    evaluate_beats,
    evaluate_downbeats,
    compute_per_track_metrics,
    compute_ibi_stats,
)

__all__ = [
    "compute_beat_metrics",
    "compute_downbeat_metrics",
    "evaluate_beats",
    "evaluate_downbeats",
    "compute_per_track_metrics",
    "compute_ibi_stats",
]
```

- [ ] **Step 4.4 — Run tests**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py -v
```

Expected: all 11 tests `PASSED`.

- [ ] **Step 4.5 — Commit**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/beats/evaluation/__init__.py tests/test_beats_experiments.py
git commit -m "feat: add mir_core.beats.evaluation re-export module"
```

---

## Task 5: Create `mir_core/classifier/` stubs

**Files:**
- Create: `mir_core/classifier/__init__.py`
- Create: `mir_core/classifier/experiments/__init__.py`
- Create: `mir_core/classifier/experiments/presets.py`
- Create: `mir_core/classifier/experiments/presets/.gitkeep`
- Create: `mir_core/classifier/evaluation/__init__.py`

- [ ] **Step 5.1 — Write failing import test**

```python
# Add to tests/test_beats_experiments.py:

def test_classifier_experiments_imports():
    from mir_core.classifier.experiments import experiment_hash, PRESETS
    assert callable(experiment_hash)
    assert isinstance(PRESETS, dict)


def test_classifier_experiment_hash_has_clf_prefix():
    from mir_core.classifier.experiments import experiment_hash
    config = {"model": {"name": "genre_classifier"}, "seed": 42}
    h = experiment_hash(config)
    assert h.startswith("clf-"), f"Expected 'clf-' prefix, got: {h}"
    assert len(h) == 20


def test_classifier_evaluation_stubs_importable():
    from mir_core.classifier.evaluation import accuracy, macro_f1, confusion_matrix
    assert callable(accuracy)
    assert callable(macro_f1)
    assert callable(confusion_matrix)
```

- [ ] **Step 5.2 — Run to confirm failure**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py::test_classifier_experiments_imports -v
```

Expected: `ModuleNotFoundError: No module named 'mir_core.classifier'`

- [ ] **Step 5.3 — Create `mir_core/classifier/__init__.py`**

```python
# mir_core/classifier/__init__.py
"""Music classifier submodule — experiments, evaluation.

Stubs ready for classifierlab training scripts.
"""
```

- [ ] **Step 5.4 — Create `mir_core/classifier/experiments/presets.py`**

```python
# mir_core/classifier/experiments/presets.py
"""Classifier experiment preset registry (stub — no presets yet).

Follows identical structure to mir_core.beats.experiments.presets.
Populate presets/ with clf-{hash}.json files when classifier experiments are defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESETS_DIR: Path = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class Preset:
    key: str
    hash: str
    citation: str
    config: dict[str, Any]
    notes: list[str]


def load_presets() -> dict[str, Preset]:
    """Returns empty dict until classifier presets are added."""
    return {}


PRESETS: dict[str, Preset] = {}
PRESETS_BY_KEY: dict[str, Preset] = {}


def get_by_hash(experiment_hash: str) -> Preset | None:
    return None


def get_by_key(key: str) -> Preset | None:
    return None
```

- [ ] **Step 5.5 — Create `mir_core/classifier/experiments/__init__.py`**

```python
# mir_core/classifier/experiments/__init__.py
"""Classifier experiment registry.

All classifier experiment hashes are prefixed with 'clf-' (20 chars total).

Usage:
    from mir_core.classifier.experiments import experiment_hash
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mir_core.utils.hashing import stable_hash
from .presets import (
    Preset,
    PRESETS,
    PRESETS_BY_KEY,
    load_presets,
    get_by_hash,
    get_by_key,
)

TASK_PREFIX = "clf"


def experiment_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the canonical classifier experiment hash.

    Format: 'clf-{16-hex-chars}' — always 20 characters.
    """
    return f"{TASK_PREFIX}-{stable_hash(config, length=length)}"


__all__ = [
    "TASK_PREFIX",
    "experiment_hash",
    "Preset",
    "PRESETS",
    "PRESETS_BY_KEY",
    "load_presets",
    "get_by_hash",
    "get_by_key",
]
```

- [ ] **Step 5.6 — Create presets directory and evaluation stubs**

```bash
cd ~/thesis-workspace/mir-core
mkdir -p mir_core/classifier/experiments/presets
touch mir_core/classifier/experiments/presets/.gitkeep
```

```python
# mir_core/classifier/evaluation/__init__.py
"""Classifier evaluation metrics (stubs — not yet implemented).

Will provide: accuracy, macro_f1, confusion_matrix when classifierlab is built.
"""


def accuracy(predictions, targets) -> float:
    """Classification accuracy. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")


def macro_f1(predictions, targets) -> float:
    """Macro-averaged F1 score. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")


def confusion_matrix(predictions, targets):
    """Confusion matrix. Not yet implemented."""
    raise NotImplementedError("classifier evaluation not yet implemented")
```

- [ ] **Step 5.7 — Run tests**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_beats_experiments.py -v
```

Expected: all 14 tests `PASSED`.

- [ ] **Step 5.8 — Commit**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/classifier/ tests/test_beats_experiments.py
git commit -m "feat: add mir_core.classifier stubs (clf- prefix, evaluation stubs)"
```

---

## Task 6: Update `mir_core/__init__.py` and `beatlab/config.py`

**Files:**
- Modify: `mir_core/__init__.py`
- Modify: `~/thesis-workspace/mir-train-hpc/beatlab/config.py`

- [ ] **Step 6.1 — Update `mir_core/__init__.py`**

Find the line:
```python
from .experiments import canonical_json, stable_digest, stable_hash, experiment_hash
```

Replace it with:
```python
from .utils.hashing import canonical_json, stable_digest, stable_hash
```

Note: the top-level `experiment_hash` alias is removed — callers should now use `mir_core.beats.experiments.experiment_hash` or `mir_core.classifier.experiments.experiment_hash`. The raw `stable_hash` remains at the top level for convenience.

- [ ] **Step 6.2 — Verify top-level re-exports still work**

```bash
cd ~/thesis-workspace/mir-core
python -c "from mir_core import stable_hash, stable_digest, canonical_json; print('OK')"
```

Expected: `OK`

- [ ] **Step 6.3 — Update `beatlab/config.py`**

In `~/thesis-workspace/mir-train-hpc/beatlab/config.py`, find:
```python
from mir_core.experiments import stable_hash
```

Replace with:
```python
from mir_core.utils.hashing import stable_hash
```

- [ ] **Step 6.4 — Verify beatlab config import works**

```bash
cd ~/thesis-workspace/mir-train-hpc
PYTHONPATH="$(pwd):$HOME/thesis-workspace/mir-core" conda run -n MIR \
    python -c "from beatlab.config import stable_hash, expand_env, load_config; print('OK')"
```

Expected: `OK`

- [ ] **Step 6.5 — Run full mir-core test suite**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/ -v
```

Expected: all tests pass (`mir_core.experiments` still exists, so `test_presets.py` still works).

- [ ] **Step 6.6 — Commit both repos**

```bash
cd ~/thesis-workspace/mir-core
git add mir_core/__init__.py
git commit -m "refactor: update mir_core.__init__ to import hashing from utils.hashing"

cd ~/thesis-workspace/mir-train-hpc
git add beatlab/config.py
git commit -m "fix: update beatlab.config to import stable_hash from mir_core.utils.hashing"
```

---

## Task 7: Delete `mir_core/experiments/`, update all consumers

**Files:**
- Delete: `mir_core/experiments/` (entire directory)
- Modify: `tests/test_presets.py`
- Modify: `beatlab/train_beat.py`

- [ ] **Step 7.1 — Update `tests/test_presets.py`**

Replace the entire file content:

```python
# tests/test_presets.py
"""Tests for the beat tracking canonical experiment preset registry."""
import json
from pathlib import Path
import pytest
from mir_core.beats.experiments.presets import Preset, PRESETS, load_presets, PRESETS_DIR


def test_preset_dataclass_fields():
    p = Preset(
        key="test_key",
        hash="btk-abc123def456abcd",
        citation="Author 2024",
        config={"experiment": {"name": "test"}},
        notes=["note one"],
    )
    assert p.key == "test_key"
    assert p.hash == "btk-abc123def456abcd"
    assert isinstance(p.config, dict)
    assert isinstance(p.notes, list)


def test_load_presets_returns_dict():
    result = load_presets()
    assert isinstance(result, dict)


def test_presets_dir_exists():
    assert PRESETS_DIR.is_dir(), f"presets/ directory not found at {PRESETS_DIR}"


def test_preset_file_schema():
    """Every .json file in presets/ must have key, citation, config, notes."""
    for f in PRESETS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        assert "key" in data, f"{f.name} missing 'key'"
        assert "citation" in data, f"{f.name} missing 'citation'"
        assert "config" in data, f"{f.name} missing 'config'"
        assert "notes" in data, f"{f.name} missing 'notes'"
        # filename stem must be btk-{16 hex chars}
        assert f.stem.startswith("btk-"), (
            f"{f.name}: filename must start with 'btk-', got '{f.stem}'"
        )
        assert len(f.stem) == 20, (
            f"{f.name}: hash must be 20 chars, got {len(f.stem)}"
        )
        assert all(c in "0123456789abcdef" for c in f.stem[4:]), (
            f"{f.name}: hash suffix must be hex, got '{f.stem[4:]}'"
        )
        assert data.get("hash") == f.stem, (
            f"{f.name}: 'hash' field '{data.get('hash')}' must match filename stem '{f.stem}'"
        )


def test_presets_registry_from_exports():
    """PRESETS exported from mir_core.beats.experiments must equal load_presets()."""
    from mir_core.beats.experiments import PRESETS as top_level
    from mir_core.beats.experiments.presets import PRESETS as direct
    assert top_level is direct
```

- [ ] **Step 7.2 — Run updated test_presets.py (expects PASS)**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/test_presets.py -v
```

Expected: all 5 tests `PASSED` (imports now from `mir_core.beats.experiments`).

- [ ] **Step 7.3 — Update `beatlab/train_beat.py` imports**

In `~/thesis-workspace/mir-train-hpc/beatlab/train_beat.py`, find:
```python
from mir_core.experiments import get_by_hash, get_by_key
```

Replace with:
```python
from mir_core.beats.experiments import get_by_hash, get_by_key, experiment_hash as btk_experiment_hash
```

Then find the experiment hash computation in `main()`:
```python
    # Hash must be computed on unexpanded config to be machine-independent.
    experiment_hash = (
        preset_hash
        or args.experiment_hash
        or stable_hash(unexpanded_config)
    )
```

Replace with:
```python
    # Hash must be computed on unexpanded config to be machine-independent.
    # btk_experiment_hash returns "btk-{16-hex-chars}" — task-prefixed.
    experiment_hash = (
        preset_hash
        or args.experiment_hash
        or btk_experiment_hash(unexpanded_config)
    )
```

Also remove `stable_hash` from the `.config` import line if it's still there:
```python
# Find:
from .config import expand_env, require_mapping, stable_hash, write_config
# Replace with:
from .config import expand_env, require_mapping, write_config
```

- [ ] **Step 7.4 — Smoke-test train_beat CLI**

```bash
cd ~/thesis-workspace/mir-train-hpc
PYTHONPATH="$(pwd):$HOME/thesis-workspace/mir-core" conda run -n MIR \
    python -m beatlab.train_beat --help 2>&1 | head -5

PYTHONPATH="$(pwd):$HOME/thesis-workspace/mir-core" conda run -n MIR python - <<'EOF'
import sys
sys.argv = ['t', '--preset', 'rapini2024_salsaset_beatnet', '--run-dir', '/tmp/btk_test']
from beatlab.train_beat import parse_args, resolve_config
args = parse_args()
config, h, key = resolve_config(args)
assert h.startswith("btk-"), f"Expected btk- prefix, got: {h}"
assert len(h) == 20, f"Expected length 20, got: {len(h)}"
print(f"PASS: key={key} hash={h}")
EOF
```

Expected: `PASS: key=rapini2024_salsaset_beatnet hash=btk-a14aef639058a4c7`

- [ ] **Step 7.5 — Delete `mir_core/experiments/`**

```bash
cd ~/thesis-workspace/mir-core
git rm -r mir_core/experiments/
```

- [ ] **Step 7.6 — Run full test suite**

```bash
cd ~/thesis-workspace/mir-core
python -m pytest tests/ -v
```

Expected: all tests pass. No references to `mir_core.experiments` remain.

- [ ] **Step 7.7 — Commit both repos**

```bash
cd ~/thesis-workspace/mir-core
git add tests/test_presets.py
git commit -m "refactor: remove mir_core.experiments, update test_presets to use beats.experiments

All consumers now use mir_core.beats.experiments or mir_core.utils.hashing.
mir_core.experiments deleted entirely."

cd ~/thesis-workspace/mir-train-hpc
git add beatlab/train_beat.py
git commit -m "fix: update train_beat.py to use mir_core.beats.experiments

get_by_hash, get_by_key now from mir_core.beats.experiments.
experiment_hash() now returns btk-prefixed hash (was bare stable_hash)."
```

---

## Self-Review

**Spec coverage:**
- [x] `mir_core/utils/hashing.py` — Task 1
- [x] `btk-` prefix on beat hashes, `experiment_hash()` — Task 2
- [x] Preset JSONs renamed with `btk-` prefix, internal hash field updated — Task 3
- [x] `mir_core/beats/evaluation/` re-export — Task 4
- [x] `mir_core/classifier/` stubs with `clf-` prefix — Task 5
- [x] `mir_core/__init__.py` updated — Task 6
- [x] `beatlab/config.py` updated — Task 6
- [x] `mir_core/experiments/` deleted — Task 7
- [x] `test_presets.py` updated — Task 7
- [x] `beatlab/train_beat.py` updated — Task 7
- [x] `from mir_core import stable_hash` still works — Task 6

**Type consistency:**
- `experiment_hash()` signature is identical in `beats/experiments/__init__.py` and `classifier/experiments/__init__.py` — consistent
- `Preset` dataclass fields are identical in both submodules — consistent
- `get_by_hash` / `get_by_key` return `Preset | None` in both — consistent
- `btk_experiment_hash` alias in `train_beat.py` matches the import — consistent
