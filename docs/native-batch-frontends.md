# Native batch-model audio frontends

`mir_core.native.batch_frontend` exports the canonical feature extractors for
BockTCN, BEAST, and SpecTNT as independently versioned ONNX artifacts. This
closes the portable graph boundary from a decoded waveform to the tensor
accepted by each model artifact; Python is needed to create and parity-check an
artifact, but not to run a distributed artifact from Rust or C++.

## ABI boundary

All three graphs expose one named input and one named output:

- `waveform`: finite mono `float32` with shape `[batch, samples]`
- `features`: the model-ready `float32` tensor described below

Audio decoding, channel mixing, and resampling remain host operations. The
host must supply the sample rate recorded in `input_contract.sample_rate`.
They are deliberately not approximated inside the graph because decoder and
resampler choices are part of the input contract and are not equivalent across
native hosts.

| Family | Exact reference contract | Output | Frame count | Model bridge |
| --- | --- | --- | --- | --- |
| BockTCN | madmom centered 2048-sample STFT, NumPy symmetric Hann, 81-band logarithmic filterbank, natural log | `[batch, 1, frames, 81]` | `ceil(samples / 441)` | Direct; model input time is dynamic and requires at least five frames |
| BEAST | librosa periodic-Hann 4096-point power mel spectrogram and `power_to_db(ref=np.max, top_db=80)` | `[batch, frames, 128]` | `floor(samples / 1024) + 1` | Extract the complete waveform first, then provide exact fixed-size feature windows to the model graph |
| SpecTNT | torchaudio periodic-Hann 512-point power STFT, reflect padding, six-channel harmonic filterbank, power dB | `[batch, 6, 128, frames]` | `floor(samples / 256) + 1` | Provide exact fixed-size feature windows to the model graph |

BEAST is exact only as a whole-waveform batch frontend. Its reference
normalization uses the largest mel power anywhere in each waveform, so the
manifest sets `streaming_safe` to `false` and records this blocker. Applying
the graph independently to chunks would change feature values and is not a
supported streaming approximation.

SpecTNT requires at least 257 samples because its centered STFT uses reflect
padding. The harmonic bandwidth parameter defaults to `bw_q=1.0`. A config may
instead point `checkpoint` at a state mapping containing exactly one
`hstft.bw_Q`; the scalar and checkpoint content hash are frozen into the
artifact identity. A checkpoint locator is never included in that identity.

## Export and run

Export only a frontend from Python:

```python
from mir_core.native import ensure_batch_frontend_onnx

artifact = ensure_batch_frontend_onnx(
    model_name="bock_tcn",
    preprocessing_config={"fps": 100, "frame_size": 2048, "num_bands": 12},
)
print(artifact.model_path, artifact.manifest_path)
```

Export a trained batch model and its matching frontend in one command from
`mir-train-hpc`:

```bash
python -m beatlab.native_export path/to/config.yaml \
  --checkpoint path/to/best.pt \
  --include-audio-frontend
```

The report keeps model and frontend parity results separate and includes a
validated `composition` record. BockTCN composes directly. BEAST and SpecTNT
model artifacts have fixed feature-time geometry, so the host owns windowing
and must supply the exact recorded frame count; the artifact does not invent a
padding or remainder policy.

Run a distributed artifact without constructing madmom, librosa, or
torchaudio preprocessors:

```python
import numpy as np
from mir_core.native import (
    OnnxBatchFrontendSession,
    load_batch_frontend_artifact,
)

artifact = load_batch_frontend_artifact("frontend.onnx.json")
session = OnnxBatchFrontendSession(artifact)
features = session.infer(np.zeros((1, 44100), dtype=np.float32))
```

Rust and C++ hosts can use ONNX Runtime directly with the same named ABI and
manifest shape rules. Loaders must verify `onnx.sha256` and `onnx.size_bytes`
before creating a session.

The Rust host also exposes an optimized direct implementation for BockTCN. It
loads and validates the same `mir.native-batch-frontend/v1` artifact, then
executes the centered STFT, logarithmic filterbank, and log compression with
RustFFT instead of evaluating the portable ONNX graph:

```python
from mir_native_runtime import BockTcnFrontendSession

session = BockTcnFrontendSession("frontend.onnx")
features = session.infer(waveforms)
```

On the development host, 220,500 input samples took 5.560 ms median with the
direct Rust host, 13.643 ms with madmom, and 31.884 ms with the ONNX frontend.
That is a 2.45x speedup over madmom and a 5.74x speedup over the portable graph;
maximum absolute error against madmom was `5.96e-7`. The ONNX artifact remains
the cross-platform fallback, and the direct host fails closed if its exact
BockTCN contract is not present.

## Artifact and parity contract

The manifest schema is `mir.native-batch-frontend/v1`, ABI version 1. Its
content-addressed identity includes the normalized family, resolved
preprocessing-config hash, constant-tensor hash, source-contract hash, ONNX
opset, exporter Torch version, and nonzero export shape. The manifest also
records the final ONNX hash and byte size. Cache publication is atomic and a
hash mismatch causes re-export instead of reuse.

Every export is accepted only after ONNX Runtime CPU matches the real reference
preprocessor on three deterministic, nonzero probes, including a dynamic input
length and a batch of two random waveforms. Unsupported preprocessing keys,
fractional BockTCN hops, ambiguous SpecTNT bandwidth state, and incompatible
model/frontend feature geometry fail closed.

For graph-only timing (excluding decode, downmix, and resampling):

```python
from mir_core.native import OnnxBatchFrontendSession, benchmark_batch_frontend

result = benchmark_batch_frontend(OnnxBatchFrontendSession(artifact))
print(result["mean_ms"], result["compute_realtime_factor"])
```
