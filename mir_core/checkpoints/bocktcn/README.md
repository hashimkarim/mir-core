# Bock TCN Baseline Checkpoints

These resources package the canonical madmom TCN beat model ensemble from:

`madmom/madmom/models/beats/2019/beats_tcn_[1-8].pkl`

The selector names map to the eight ensemble members:

| Selector | Source file |
| --- | --- |
| `baseline` | `beats_tcn_1.pkl` |
| `baseline_alt0` | `beats_tcn_2.pkl` |
| `baseline_alt1` | `beats_tcn_3.pkl` |
| `baseline_alt2` | `beats_tcn_4.pkl` |
| `baseline_alt3` | `beats_tcn_5.pkl` |
| `baseline_alt4` | `beats_tcn_6.pkl` |
| `baseline_alt5` | `beats_tcn_7.pkl` |
| `baseline_alt6` | `beats_tcn_8.pkl` |

These are madmom pickle resources for the published inference ensemble, not
native PyTorch `BockTCN` state dictionaries.

## Portable original ensemble

`mir_core.models.bock_tcn.LegacyBockTCN()` loads all eight resources only after
checking their pinned SHA-256 digests. Pass a list of packaged selectors to
export individual members. This inference-only conversion preserves the
original 2019 beat and 300-class tempo heads; it is separate from the trainable
hybrid `BockTCN`. The native batch family is `bocktcn_legacy`.

Use `ensure_batch_model_onnx(model, model_name="bocktcn_legacy",
model_config=model.port_config, checkpoint_sha256=model.checkpoint_sha256,
input_shape=(1, 1, 37, 81))`. Its input comes from the existing `bocktcn` native
waveform frontend at 44100 Hz and 100 fps. The model repeats the boundary
feature frames twice, returns one beat activation per original frame, and
averages the members in the original order. Tempo is a whole-sequence output.
This is a noncausal batch model; chunking changes its context.

The conversion follows madmom's SciPy convolution path: each kernel uses
double accumulation before float32 channel summation. The export gate and
`tests/test_native_legacy_bock.py` compare all eight trained members and the
full waveform ensemble against the original madmom processors with the
existing fixed native model tolerance (`rtol=2e-5`, `atol=2e-6`). Exporting
modified weights or source metadata under this historical identity is rejected.
The original resource licensing and attribution above still apply.
