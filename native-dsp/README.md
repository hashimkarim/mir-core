# Shared native audio DSP

The C++ C ABI and safe Rust facade implement the same mono float32 SoXR HQ
resampler used by the Python reference. Android/JVM uses the same C++ source
through JNI. This is one shared implementation, with separately exercised
language and platform interfaces.

```sh
cmake -S native-dsp -B native-dsp/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-dsp/build --parallel 2
cargo build --manifest-path native-dsp/rust/Cargo.toml --locked --release --examples
```

`include/mir_dsp.h` documents ABI 1 and `mir.soxr-hq/mono-f32-v1`.
`mir_dsp::Resampler` dynamically loads a caller-selected trusted library.
Input blocks must be finite; empty blocks are valid. `finish` drains the filter
tail, and `reset` starts a fresh stream. Output timing and filter delay are part
of the contract: callers must accept variable-size output blocks and explicitly
flush at end of audio. Each instance owns its state and must be used exclusively.

`tools/resampling_parity.py` compares C++ and Rust to Python SoXR over rate pairs,
chunk sizes, impulses, tones, flush, reset and alias-rejection probes, and exports
the shared Android/JVM reference fixtures. Run it in conda `MIR`; use `--help`
for paths. Phone validation requires a separately negotiated ADB slot.

The vendored archive is unmodified Python-SoXR 0.5.0.post1 source with its exact
embedded libsoxr revision. CMake verifies the archive SHA-256 before extraction.
See `third_party/README.md` and the included upstream licenses; distribution
must preserve libsoxr's LGPL obligations. The build uses a separate shared soxr
library. No model weights are included here.
