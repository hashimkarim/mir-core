# libsoxr provenance

`soxr-0.5.0.post1.tar.gz` is the unmodified Python-SoXR source distribution:

- Source: https://pypi.org/project/soxr/0.5.0.post1/#files
- Archive SHA-256: `7092b9f3e8a416044e1fa138c8172520757179763b85dc53aa9504f4813cff73`
- The `libsoxr/` subtree identifies as `0.1.3-11-gedbdb40`, matching conda MIR's Python-SoXR reference.
- Only the libsoxr subtree is built; no Python interpreter is used at runtime.
- `COPYING.LGPL` and `LICENSE-PFFFT.txt` preserve the upstream terms. Additional notices remain in the source archive.

The CMake build verifies the archive hash and links libsoxr as a separate shared
library. Distribute the source archive and these notices with distributions of
the native library. The source is unchanged; MIR wrappers are outside the archive.
