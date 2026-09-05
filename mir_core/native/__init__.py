"""Native deployment helpers for MIR models.

Training remains in PyTorch.  This package defines the stable artifact boundary
used by native runtimes so the same exported graph can be consumed from Python,
C++, Rust, or an embedded deployment toolchain.
"""

from .beatnet import (
    NATIVE_STREAMING_SCHEMA,
    NativeModelArtifact,
    OnnxBeatNetStreamingSession,
    ensure_streaming_beatnet_onnx,
    export_streaming_beatnet_onnx,
    onnxruntime_available,
    resolve_streaming_backend,
)

__all__ = [
    "NATIVE_STREAMING_SCHEMA",
    "NativeModelArtifact",
    "OnnxBeatNetStreamingSession",
    "ensure_streaming_beatnet_onnx",
    "export_streaming_beatnet_onnx",
    "onnxruntime_available",
    "resolve_streaming_backend",
]
