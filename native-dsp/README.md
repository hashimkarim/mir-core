# Shared native audio DSP

This folder holds the resampler that the apps share with the Python reference.
The C++ library with a C interface and the safe Rust facade implement the same
mono float32 SoXR HQ resampler used by the Python reference. Android/JVM uses
the same C++ source through JNI. This is one shared implementation, with
separately exercised language and platform interfaces.

## Contents

| Path | Content |
| --- | --- |
| `CMakeLists.txt` | Builds the shared library `mir_dsp` and the replay program `mir_resampler_replay`. |
| `include/mir_dsp.h` | The C interface. |
| `src/resampler.cpp` | The resampler. Reports ABI version 1 and the contract `mir.soxr-hq/mono-f32-v1`. |
| `src/resampler_jni.cpp` | The JNI interface for Android/JVM. |
| `rust/` | The crate `mir-dsp` with `mir_dsp::Resampler` and the example `resampler_replay`. |
| `tests/resampler_replay.cpp` | Source of the C++ replay program. |
| `tools/resampling_parity.py` | Compares C++ and Rust with Python SoXR and writes the reference fixtures. |
| `third_party/` | The unmodified Python-SoXR 0.5.0.post1 source archive and its licences. See `third_party/README.md`. |
| `build/`, `rust/target/` | Local build output. Git ignores both. |

## How to use it

Build from the repository root:

```sh
cmake -S native-dsp -B native-dsp/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-dsp/build --parallel 2
cargo build --manifest-path native-dsp/rust/Cargo.toml --locked --release --examples
```

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

## Status

Current on 2026-10-09. The workflow `.github/workflows/native-dsp-parity.yml`
builds both interfaces and runs the parity tool on changes to this folder.
