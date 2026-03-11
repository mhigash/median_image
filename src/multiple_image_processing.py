"""Processing operations on multiple images (image stacks)."""

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QFileDialog


def make_median_image(parent, paths):
    """Compute per-pixel median across images and prompt the user to save.

    Args:
        parent: Parent widget for dialogs.
        paths: List of image file paths.

    Returns:
        A status message string describing the outcome.
    """
    n = len(paths)
    progress = QProgressDialog("Loading images...", "Cancel", 0, n + 1, parent)
    progress.setWindowTitle("Making Median Image")
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    # Load all images, using the first valid image for reference shape
    images = []
    ref_shape = None
    for i, path in enumerate(paths):
        if progress.wasCanceled():
            progress.close()
            return "Median image cancelled"
        progress.setLabelText(
            f"Loading image {i + 1}/{n}: {os.path.basename(path)}")
        QApplication.processEvents()

        img = cv2.imread(path)
        if img is None:
            progress.setValue(i + 1)
            continue
        if ref_shape is None:
            ref_shape = img.shape
        # Only include images with matching dimensions
        if img.shape == ref_shape:
            images.append(img)
        progress.setValue(i + 1)

    if not images:
        progress.close()
        return "No valid images to process"

    progress.setLabelText("Computing median...")
    QApplication.processEvents()

    stack = np.array(images, dtype=np.uint8)
    median_img = np.median(stack, axis=0).astype(np.uint8)

    progress.setValue(n + 1)
    progress.close()

    # Ask user where to save
    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Median Image", "median.png",
        "Images (*.png *.jpg *.bmp)")
    if not save_path:
        return "Median image not saved"

    cv2.imwrite(save_path, median_img)
    return f"Median image saved — {len(images)} images, {save_path}"
