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
from .batch import (
    NATIVE_BATCH_SCHEMA,
    NativeBatchModelArtifact,
    OnnxBatchModelSession,
    UnsupportedNativeBatchModelError,
    ensure_batch_model_onnx,
    export_batch_model_onnx,
)
from .batch_frontend import (
    BATCH_FRONTEND_ONNX_OPSET_VERSION,
    NATIVE_BATCH_FRONTEND_SCHEMA,
    NativeBatchFrontendArtifact,
    OnnxBatchFrontendSession,
    SUPPORTED_BATCH_FRONTENDS,
    UnsupportedNativeBatchFrontendError,
    benchmark_batch_frontend,
    ensure_batch_frontend_onnx,
    export_batch_frontend_onnx,
    load_batch_frontend_artifact,
)

__all__ = [
    "BATCH_FRONTEND_ONNX_OPSET_VERSION",
    "NATIVE_BATCH_FRONTEND_SCHEMA",
    "NATIVE_BATCH_SCHEMA",
    "NATIVE_STREAMING_SCHEMA",
    "NativeBatchFrontendArtifact",
    "NativeBatchModelArtifact",
    "NativeModelArtifact",
    "OnnxBatchFrontendSession",
    "OnnxBatchModelSession",
    "OnnxBeatNetStreamingSession",
    "SUPPORTED_BATCH_FRONTENDS",
    "UnsupportedNativeBatchFrontendError",
    "UnsupportedNativeBatchModelError",
    "benchmark_batch_frontend",
    "ensure_batch_frontend_onnx",
    "ensure_batch_model_onnx",
    "ensure_streaming_beatnet_onnx",
    "export_batch_frontend_onnx",
    "export_batch_model_onnx",
    "export_streaming_beatnet_onnx",
    "load_batch_frontend_artifact",
    "onnxruntime_available",
    "resolve_streaming_backend",
]
