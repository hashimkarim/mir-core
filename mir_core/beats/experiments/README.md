# mir_core/beats/experiments

This folder gives every beat-tracking experiment its identity. It computes the
experiment hash (`btk-...`) from a configuration and keeps a small registry of
presets that reproduce the experiment setups of published papers.

## Contents

| Path | Content |
| --- | --- |
| `__init__.py` | `experiment_hash(config)` and the re-exported registry functions. |
| `presets.py` | The `Preset` dataclass and the loader that scans `presets/`. |
| `presets/` | One JSON file per preset. The file name is the experiment hash. |

Available presets:

| Hash | Key | Paper |
| --- | --- | --- |
| `btk-724a5e4d0d1e8edf` | `rapini2024_salsaset_beatnet` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-68a1f54a14e999c9` | `rapini2024_salsaset_bocktcn` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-65bd103f0edc1432` | `heydari2021_beatnet` | Heydari et al. 2021 (ISMIR) |
| `btk-6e688b9bba34efba` | `davies2019_bocktcn` | Davies & Böck 2019 (EUSIPCO) |

## How to use it

### Hashing

`experiment_hash(config_dict)` returns `btk-` followed by the first 16
hexadecimal characters of a SHA-256 digest: always 20 characters. Hash the
configuration before environment variables are expanded, so the hash is the
same on every machine.

### Presets

Each file in `presets/` is a self-contained experiment definition with the
fields `key`, `hash`, `citation`, `notes`, `category` and `config`.

These presets reproduce historical paper-specific dataset universes and carry
`data.allow_legacy_split: true` explicitly. New multi-lab thesis experiments
must use the universal `data.split_plan` runtime in `mir-train-hpc`; the opt-in
prevents a paper preset from silently being mistaken for that shared protocol.

List the current keys:

```bash
python -c "from mir_core.beats.experiments import PRESETS_BY_KEY; print(list(PRESETS_BY_KEY))"
```

Python API:

```python
from mir_core.beats.experiments import (
    PRESETS, PRESETS_BY_KEY, get_by_hash, get_by_key, experiment_hash
)

preset = get_by_key("rapini2024_salsaset_beatnet")
preset.hash        # "btk-724a5e4d0d1e8edf"
preset.citation    # full citation string
preset.config      # complete config dict (unexpanded)
preset.notes       # list of discrepancy / methodology notes

# Compute hash for a new config
h = experiment_hash(config_dict)  # "btk-{16 hex chars}"
```

### Adding a preset

1. Build the config dict with raw (unexpanded) environment variable strings.
2. Compute the hash:
   ```bash
   python -c "from mir_core.beats.experiments import experiment_hash; import json; print(experiment_hash(json.load(open('my_preset.json'))['config']))"
   ```
3. Write `presets/{hash}.json` with the `key`, `hash`, `citation`, `notes` and
   `config` fields.
4. Commit. No Python change is required.

### BeatNet preprocessing discrepancy

The paper text (Heydari et al. 2021) gives a 93 ms window and a 46 ms hop,
about 22 fps. The official implementation, used for all published results,
has a window of 1408 samples (64 ms) and a hop of 441 samples (20 ms): 50 fps.
All BeatNet presets use 50 fps to match the published numbers.

## Status

Current on 2026-10-09. The four preset files, hashes and keys match this
table. The presets describe paper setups; the thesis runs use the matrix
configurations of `mir-train-hpc`, not these presets.
