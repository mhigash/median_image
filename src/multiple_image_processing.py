"""Processing operations on multiple images (image stacks)."""

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QFileDialog


def _load_stack(parent, paths, title):
    """Load images with a progress dialog.

    Returns:
        (images, cancelled) where images is a list of numpy arrays
        and cancelled is True if the user cancelled.
    """
    n = len(paths)
    progress = QProgressDialog("Loading images...", "Cancel", 0, n + 1, parent)
    progress.setWindowTitle(title)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    images = []
    ref_shape = None
    for i, path in enumerate(paths):
        if progress.wasCanceled():
            progress.close()
            return [], True
        progress.setLabelText(
            f"Loading image {i + 1}/{n}: {os.path.basename(path)}")
        QApplication.processEvents()

        img = cv2.imread(path)
        if img is None:
            progress.setValue(i + 1)
            continue
        if ref_shape is None:
            ref_shape = img.shape
        if img.shape == ref_shape:
            images.append(img)
        progress.setValue(i + 1)

    progress.setValue(n + 1)
    progress.close()
    return images, False


def make_median_image(parent, paths):
    """Compute per-pixel median across images and prompt the user to save.

    Args:
        parent: Parent widget for dialogs.
        paths: List of image file paths.

    Returns:
        A status message string describing the outcome.
    """
    images, cancelled = _load_stack(parent, paths, "Making Median Image")
    if cancelled:
        return "Median image cancelled"
    if not images:
        return "No valid images to process"

    stack = np.array(images, dtype=np.uint8)
    result = np.median(stack, axis=0).astype(np.uint8)

    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Median Image", "median.png",
        "Images (*.png *.jpg *.bmp)")
    if not save_path:
        return "Median image not saved"

    cv2.imwrite(save_path, result)
    return f"Median image saved — {len(images)} images, {save_path}"


def make_mean_image(parent, paths):
    """Compute per-pixel mean across images and prompt the user to save.

    Args:
        parent: Parent widget for dialogs.
        paths: List of image file paths.

    Returns:
        A status message string describing the outcome.
    """
    images, cancelled = _load_stack(parent, paths, "Making Mean Image")
    if cancelled:
        return "Mean image cancelled"
    if not images:
        return "No valid images to process"

    stack = np.array(images, dtype=np.float32)
    result = np.mean(stack, axis=0).astype(np.uint8)

    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Mean Image", "mean.png",
        "Images (*.png *.jpg *.bmp)")
    if not save_path:
        return "Mean image not saved"

    cv2.imwrite(save_path, result)
    return f"Mean image saved — {len(images)} images, {save_path}"
