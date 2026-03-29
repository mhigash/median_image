"""Processing operations on multiple images (image stacks).

All functions are pure — no GUI dependencies.

Progress is reported via an optional callback:
    progress_cb(current: int, total: int, filename: str) -> bool
Return False from the callback to request cancellation.
"""

import os

import cv2
import numpy as np


def load_stack(paths, progress_cb=None):
    """Load images from paths into a list of numpy arrays.

    Args:
        paths: List of file paths.
        progress_cb: Optional callable(current, total, filename) -> bool.
                     Return False to cancel; returns None on cancellation.

    Returns:
        List of BGR numpy arrays with matching shape, or None if cancelled.
    """
    images = []
    ref_shape = None
    n = len(paths)
    for i, path in enumerate(paths):
        if progress_cb is not None:
            if not progress_cb(i, n, os.path.basename(path)):
                return None
        img = cv2.imread(path)
        if img is None:
            continue
        if ref_shape is None:
            ref_shape = img.shape
        if img.shape == ref_shape:
            images.append(img)
    return images


def make_median(images):
    """Compute per-pixel median across images.

    Args:
        images: List of BGR numpy arrays with the same shape.

    Returns:
        Median image as uint8 numpy array.
    """
    stack = np.array(images, dtype=np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def make_mean(images):
    """Compute per-pixel mean across images.

    Args:
        images: List of BGR numpy arrays with the same shape.

    Returns:
        Mean image as uint8 numpy array.
    """
    stack = np.array(images, dtype=np.float32)
    return np.mean(stack, axis=0).astype(np.uint8)


def compute_anomaly_maps(images, method="mean", threshold=2.0, normalize=True,
                         progress_cb=None):
    """Compute per-pixel anomaly heatmaps for each image relative to the stack.

    Args:
        images: List of BGR numpy arrays with the same shape.
        method: 'mean' for Z-score (mean/std), 'median' for MAD-based.
        threshold: Number of σ that maps to mid-range (128) in the heatmap.
                   Only used when normalize=True.
        normalize: If True, divide by spread and scale by threshold so the
                   heatmap spans the full colour range relative to σ.
                   If False, use raw absolute difference |image − reference|
                   clipped to 0–255, showing true pixel-value deviation.
        progress_cb: Optional callable(current, total, filename) -> bool.

    Returns:
        List of BGR heatmap arrays (one per input image), or None if cancelled.
    """
    stack = np.array(images, dtype=np.float32)

    if method == "mean":
        ref = np.mean(stack, axis=0)
        spread = np.std(stack, axis=0)
    else:
        ref = np.median(stack, axis=0)
        # MAD (Median Absolute Deviation) is a robust spread estimator immune to outliers.
        # For a normal distribution, MAD = 0.6745 * sigma, because the 75th percentile
        # of N(0,1) is Phi^{-1}(0.75) = 0.6745 — meaning 50% of values fall within
        # ±0.6745 sigma of the median. Multiplying by 1/0.6745 = 1.4826 rescales MAD
        # to match sigma, so anomaly z-scores are on the same scale as the mean/std path.
        spread = np.median(np.abs(stack - ref), axis=0) * 1.4826

    n = len(images)
    heatmaps = []
    for i, img in enumerate(images):
        if progress_cb is not None:
            if not progress_cb(i, n, ""):
                return None

        diff = np.abs(img.astype(np.float32) - ref)

        if normalize:
            # Divide per-channel before collapsing, then scale by threshold
            score = diff / (spread + 1e-6)
            if score.ndim == 3:
                score = score.max(axis=2)
            score_norm = np.clip(score / (threshold * 2) * 255, 0, 255).astype(np.uint8)
        else:
            # Raw absolute pixel difference; collapse then clip to 0–255
            if diff.ndim == 3:
                diff = diff.max(axis=2)
            score_norm = np.clip(diff, 0, 255).astype(np.uint8)

        heatmaps.append(cv2.applyColorMap(score_norm, cv2.COLORMAP_JET))

    return heatmaps
