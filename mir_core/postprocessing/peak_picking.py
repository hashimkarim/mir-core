"""
Convenience functions for beat detection and tempo estimation.

Functions:
    detect_beats  — one-call beat detection (creates a DBNBeatTracker internally).
    detect_tempo  — tempo estimation from activation histogram via peak picking.
    peak_picking  — simple threshold + min-interval peak picking (no DBN).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

from mir_core.utils.signal import smooth_signal


def detect_beats(
    activations: np.ndarray,
    fps: int = 100,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: float = 100.0,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Convenience function to detect beats from activations.

    Args:
        activations: Beat activation function
        fps: Frames per second
        min_bpm: Minimum tempo
        max_bpm: Maximum tempo
        transition_lambda: DBN transition parameter
        threshold: Detection threshold

    Returns:
        Array of beat times in seconds
    """
    from .dbn import DBNBeatTracker

    tracker = DBNBeatTracker(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=fps,
        transition_lambda=transition_lambda,
        threshold=threshold,
    )
    return tracker(activations)


def detect_tempo(
    activations: np.ndarray,
    hist_smooth: int = 11,
    min_bpm: float = 10.0,
) -> np.ndarray:
    """
    Detect tempo from activation function.

    Args:
        activations: Tempo activation function (histogram over BPM)
        hist_smooth: Smoothing window for histogram
        min_bpm: Minimum BPM to consider

    Returns:
        Array of (tempo, strength) tuples, sorted by strength
    """
    min_bpm_idx = int(np.floor(min_bpm))
    tempi = np.arange(min_bpm_idx, len(activations))
    bins = activations[min_bpm_idx:]

    # Smooth histogram
    if hist_smooth > 0:
        bins = smooth_signal(bins, hist_smooth)

    # Interpolate for finer resolution
    interpolation_fn = interp1d(tempi, bins, 'quadratic')
    tempi_fine = np.arange(tempi[0], tempi[-1], 0.001)
    bins_fine = interpolation_fn(tempi_fine)

    # Find peaks
    peaks = argrelmax(bins_fine, mode='wrap')[0]

    if len(peaks) == 0:
        return np.array([[0.0, 0.0]])

    # Sort by strength
    sorted_peaks = peaks[np.argsort(bins_fine[peaks])[::-1]]
    strengths = bins_fine[sorted_peaks]
    strengths = strengths / np.sum(strengths)  # Normalize

    # Return top tempi with strengths
    result = np.array([
        [tempi_fine[p], s]
        for p, s in zip(sorted_peaks[:2], strengths[:2])
    ])

    return result


def peak_picking(
    activations: np.ndarray,
    threshold: float = 0.5,
    fps: int = 100,
    min_interval: float = 0.2,
) -> np.ndarray:
    """
    Simple peak picking for beat detection.

    Alternative to DBN for simpler/faster processing.

    Args:
        activations: Activation function
        threshold: Detection threshold
        fps: Frames per second
        min_interval: Minimum interval between beats (seconds)

    Returns:
        Array of beat times
    """
    from scipy.signal import find_peaks

    min_distance = int(min_interval * fps)

    peaks, _ = find_peaks(
        activations,
        height=threshold,
        distance=min_distance,
    )

    return peaks / fps
