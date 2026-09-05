"""Small signal helpers kept independent of the optional madmom package."""

from __future__ import annotations

import numpy as np


def smooth_signal(signal: np.ndarray, kernel: np.ndarray | int | None) -> np.ndarray:
    """Smooth along axis zero with madmom-compatible Hamming convolution.

    This is the complete behavior used by mir-core's tempo helpers. Keeping the
    tiny primitive local avoids importing the whole madmom stack merely to call
    ``madmom.audio.signal.smooth``.
    """

    if kernel is None:
        return signal
    if isinstance(kernel, (int, np.integer)):
        if kernel == 0:
            return signal
        if kernel > 1:
            kernel = np.hamming(int(kernel))
        else:
            raise ValueError(f"can't create a smoothing kernel of size {kernel:d}")
    elif not isinstance(kernel, np.ndarray):
        raise ValueError(f"can't smooth signal with {kernel}")

    if signal.ndim == 1:
        return np.convolve(signal, kernel, "same")
    if signal.ndim == 2:
        from scipy.signal import convolve2d

        return convolve2d(signal, kernel[:, np.newaxis], "same")
    raise ValueError("signal must be either 1D or 2D")


__all__ = ["smooth_signal"]
