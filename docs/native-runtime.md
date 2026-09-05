# Native MIR runtime

## Decision

Keep training and checkpoint selection in PyTorch. Export deployment graphs to
ONNX with an explicit, versioned streaming ABI, then execute those graphs in a
native runtime. ONNX Runtime uses a C++ inference engine today; the same `.onnx`
artifact can later be loaded by a standalone C++ host or a Rust host without
translating learned weights or maintaining a second model implementation.

Hand-porting each layer is not the first step. It would duplicate model logic,
make checkpoint compatibility fragile, and still leave feature extraction and
causal decoding outside the port. Numerical parity at every boundary is a
release requirement.

## Implemented coverage

DanceBeatNet support below describes the optional export contract. Its Python
model is developed on a separate branch and is not required to use these native
exports; Dance-specific model tests skip when that module is unavailable.

BeatNet, BeatNet+, DanceBeatNet, and a selected MultiHeadBeatNet genre head now
export one causal feature frame per graph invocation. The LSTM hidden and cell
tensors are explicit inputs and outputs. BeatNet graphs return canonical
`[all_beats, downbeats]` activations, independent of the training model's
three-class tensor layout. DanceBeatNet also returns individually named
`beats`, `downbeats`, and `dancebeats` sigmoid heads; its canonical two-channel
output uses the configured beat or dance accent target. MultiHead export
requires an explicit genre label and hashes that selection into the artifact.
A content-addressed manifest binds every graph to the source checkpoint hash,
normalized model configuration, export ABI, Torch exporter version, ONNX hash,
tensor names, shapes, and dtype.

Recurrent exports now identify their computation policy as
`torch.onnx.legacy.fp64-ordered-fp32-state-v1`. Checkpoint weights are unchanged.
The convolution, projection, LSTM gates and output normalization compute in
float64 using an explicit pairwise reduction tree and shared range-reduced
polynomial nonlinearities. The graph contains no Conv, LSTM, GEMM, MatMul,
Exp, Tanh, Sigmoid or ReduceSum operators whose internal operation order or
math implementation can vary by provider. Activations and persisted hidden/cell
state cast directly to float32 once per frame. There is no state-rounding grid.
The external ABI stays float32. These primitive tensor operations replace
provider-specific kernels, whose small differences
can otherwise send a downstream particle filter down a different trajectory.
Android and desktop use this same exporter implementation. The policy is part
of the content-addressed identity, so existing float32 graphs are not reused.

`tools/validation/check_precise_streams.py` compares complete 4,096-frame
streams across all 38 promoted/stock BeatNet checkpoints. It requires bitwise
equality of deployed activations and every persisted state value, exact decoder
events and reset against an independent NumPy implementation of the specified
operation sequence. It also evaluates the original PyTorch model in float64
with float32 state boundaries, retaining its differences under the original
numerical tolerance.
The original float32 model, checkpoint/split hashes, activation tolerance and
legacy event comparisons are retained separately. Deployment agreement does
not establish equivalent held-out musical accuracy to the old computation
policy. Fixed-operation computation changes graph size and compute cost; measure
the intended route count on the target device before making latency claims.

Streaming exports are accepted only after a deterministic 12-frame replay
matches eager PyTorch execution of the deployment graph on CPU. The gate advances PyTorch and ONNX Runtime hidden/cell
states independently and checks canonical activations, both recurrent outputs,
and every DanceBeatNet named head on every frame. Its versioned successful
record is bound to the ONNX digest in the manifest. Python cache reuse and the
Rust loader both reject missing, failed, stale, incomplete, or over-tolerance
parity records; a failed export removes the candidate graph and manifest.

The Python host is deliberately thin: feature arrays cross into ONNX Runtime,
the recurrent graph runs in native C++, and two activation floats come back.
Cached exports live outside the repository under
`$MIR_NATIVE_MODEL_CACHE`, `$XDG_CACHE_HOME/mir/native-models`, or
`~/.cache/mir/native-models`.

The separate batch ABI supports BockTCN, BEAST, and SpecTNT. Every batch export
is executed through ONNX Runtime and compared numerically with its deployment
graph before the manifest is written. BEAST's gate covers linspace plus seeded
random inputs at batch sizes one and two. BockTCN has dynamic batch and time axes;
its odd, stride-one dilated `Conv1d` layers are copied with equivalent numeric
padding because ONNX Runtime rejects dilated `SAME_UPPER` convolutions. BEAST
has a dynamic batch axis with fixed time/context geometry. Its deployment copy
expresses contextual-block assembly, the layer-to-layer context shift, and
output stitching as functional tensor operations; this replaces exporter-
hostile in-place slice writes without changing the eager model's result.
SpecTNT has a dynamic batch axis and fixed feature geometry matching its
learned spectral/temporal tensors. Manifests record each shape contract, every
named output and its semantics, source hashes, the ONNX hash, and parity errors.

The classifier ABI covers every architecture in
`mir_core.models.classifier.architectures`: MelCNN, MFCCCNN,
MelCNNAttention, BeatNetLogSpectCNN, EmbeddingStatsMLP,
FramewiseEmbeddingMLP, and BeatNetConvClassifier. It also exports the
GenreClassifier wrapper, preserving label order and calibration temperature.
Classifier graphs accept dynamic batch and context-time axes, return named raw
`logits` and calibrated `probabilities`, and must pass the same deterministic
nonzero-input ONNX Runtime gate before caching.

Promoted EfficientAT and YAMNet checkpoints package their trained downstream
heads, and those heads are parity-tested against the shipped fold artifacts.
Their upstream feature sources are now separate, independently gated native
artifacts: EfficientAT uses ONNX Runtime and YAMNet uses a builtins-only
TensorFlow Lite flatbuffer. Each artifact binds the exact source/configuration
and can be loaded without importing the reference model. Keeping frontend and
head identities separate prevents a successful frontend conversion from
weakening classifier-head acceptance.

The canonical batch preprocessors also have waveform-to-feature ONNX exports.
BockTCN reproduces the madmom magnitude/filterbank/log frontend, BEAST
reproduces the librosa mel/power-to-dB frontend, and SpecTNT reproduces its
torchaudio harmonic STFT, including a checkpoint-bound learned `hstft.bw_Q`.
The input boundary is finite mono float32 audio at the manifest's canonical
sample rate; decode, channel mixing, and resampling remain host responsibilities.
BEAST is marked `streaming_safe=false` because its `ref=np.max` dB normalization
depends on the maximum over the complete waveform.

The Rust host has an optimized direct BockTCN frontend behind that same checked
batch-frontend manifest. It replaces only the artifact's ONNX STFT/filterbank
graph with RustFFT DSP while preserving its tensor contract. A 220,500-sample
development-host replay measured 2.45x faster than madmom and 5.74x faster than
the portable ONNX graph, with `5.96e-7` maximum absolute error. The ONNX graph
remains the portable fallback for non-Rust hosts.

For the live BeatNet families, the Rust host additionally owns the rolling
LOG_SPECT frontend, explicit recurrent state, and ONNX Runtime session. It can
run a combined audio-hop session or share one frontend across persistent routed
model sessions. The generic Rust tensor host validates and executes batch,
classifier, EfficientAT, and batch-frontend artifacts without reimplementing
the learned layers. A live `beatnet_log_spect` classifier also selects the Rust
causal-window frontend when available; its canonical-contract gate runs before
audio processing, and automatic fallback remains Python-only and startup-only.

## Scope and order

| Component | Current deployment behavior | Native path |
| --- | --- | --- |
| BeatNet | Stateful causal desktop model | Streaming ONNX plus Rust audio/frontend/state host implemented |
| BeatNet+ | Stateful causal desktop model | Streaming ONNX plus Rust audio/frontend/state host implemented |
| DanceBeatNet | Stateful model with nested beat/downbeat/dancebeat heads | Streaming ONNX/Rust path implemented; promoted full-track gate awaits a trained artifact |
| MultiHeadBeatNet | Shared convolution with genre-specific recurrent heads | Explicitly selected, hash-bound genre-head streaming export implemented |
| BockTCN | Batch/evaluation path; no faithful desktop stream runner | Batch ONNX implemented with dynamic time; causal receptive-field cache still needed |
| BEAST | Batch transformer plus experimental streaming layers | Batch ONNX implemented with dynamic batch and fixed context/time geometry; short-sequence branch remains unsupported |
| SpecTNT | Batch spectro-temporal transformer | Batch ONNX implemented with fixed feature geometry; streaming/cache semantics still needed |
| BockTCN/BEAST/SpecTNT frontends | Python scientific-library preprocessing | Parity-gated waveform ONNX implemented; BockTCN also has a faster direct RustFFT host; BEAST remains whole-waveform-only |
| EfficientAT/YAMNet router | Promoted embedding head plus frozen upstream frontend | All heads exported; EfficientAT ONNX and YAMNet TFLite frontends implemented |
| Joint and beat-only online DBNs | Python/madmom causal forward filters | Exact C++ f64 topology construction and allocation-free data plane; supported native startup does not import madmom |
| 1D state space | Python/NumPy | Exact allocation-free C++ implementation |
| Particle filter | Python/NumPy with bursty resampling | Exact C++ implementation for the explicit portable RNG contract |

“Exported” does not mean “accepted.” Each family must pass the applicable
feature, activation/state, or output-tensor parity gate, event-trace parity
after its decoder, reset/discontinuity tests, content/provenance checks, and
latency benchmarks. Unsupported settings fail before processing rather than
silently selecting a similar algorithm.

The packaged BockTCN baselines are madmom network pickles, not PyTorch
`BockTCN` state dictionaries, so they cannot be attached to this graph by a
normal state-dict load. The packaged BEAST baseline loads strictly into the
paper configuration and passes the native export gate. Directly tracing its
source forward remains unsafe: the convolution frontend agrees, but the
contextual encoder diverges because Python-controlled overlapping slice writes
are not preserved by the legacy exporter. The functional deployment graph is
therefore part of the BEAST artifact contract. Its fixed-shape requirement also
avoids the source model's currently broken `time <= block_size` branch. A graph
that merely exports but fails parity is not a native port.

## What native execution can and cannot fix

Native execution reduces per-hop compute, Python scheduling, memory allocation,
and latency variance. It does not remove the 20 ms input hop, model receptive
field, causal state-acquisition time, decoder confidence delay, audio driver
buffering, network delay, or physical actuator rise time. Those are measured
separately; a faster graph must not be reported as lower musical-event latency
unless end-to-end event timing also improves.

## Desktop selection

The desktop runtime uses `MIR_DESKTOP_MODEL_BACKEND=auto` by default. With the
optional ONNX Runtime dependency installed, `auto`/CPU selects native inference;
explicit CUDA remains on PyTorch until a CUDA-provider parity and tail-latency
gate is added. Use `MIR_DESKTOP_MODEL_BACKEND=torch` for the reference path or
`MIR_DESKTOP_MODEL_BACKEND=onnxruntime` to require the native CPU path.

`MIR_DESKTOP_DATA_PLANE=auto|python|rust` controls whether the rolling
LOG_SPECT frontend and recurrent-session orchestration stay in Python or move
into the checked Rust extension. `auto` chooses Rust only for a supported CPU
artifact and exact canonical frontend contract. `MIR_DESKTOP_POSTPROCESSOR_BACKEND`
similarly selects `auto|python|native` for the C++ causal decoders. Automatic
fallback is initialization-only; a native runtime error after state has
advanced is terminal for that stream.

The desktop GUI itself remains the existing Comfy-themed PyQt control plane.
It consumes losslessly buffered telemetry at 10 Hz while the native data plane
runs at the 50 Hz audio cadence. A second Rust/C++ GUI would not make ONNX
kernels faster and is deferred until UI profiling or a Python-free packaging
requirement justifies maintaining another application shell.
