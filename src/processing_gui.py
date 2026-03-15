"""Qt GUI wrappers for multiple_image_processing functions.

Each public function handles progress dialogs and file dialogs, then
delegates computation to the pure functions in multiple_image_processing.
"""

import os

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QProgressDialog,
    QApplication,
    QFileDialog,
    QDialog,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
)

from multiple_image_processing import (
    load_stack,
    make_median,
    make_mean,
    compute_anomaly_maps,
)


def _progress_dialog(parent, title, n):
    """Create and return a modal progress dialog."""
    dlg = QProgressDialog("Loading images...", "Cancel", 0, n, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    return dlg


def _load_with_progress(parent, paths, title):
    """Load image stack with a modal progress dialog.

    Returns:
        (images, cancelled): images is a list of arrays or [] on failure;
        cancelled is True if the user pressed Cancel.
    """
    n = len(paths)
    progress = _progress_dialog(parent, title, n)

    def cb(current, total, filename):
        if progress.wasCanceled():
            return False
        progress.setLabelText(f"Loading image {current + 1}/{total}: {filename}")
        progress.setValue(current)
        QApplication.processEvents()
        return True

    images = load_stack(paths, cb)
    progress.close()

    if images is None:
        return [], True
    return images, False


def make_median_image(parent, paths):
    """Load stack, compute median image, prompt user to save.

    Returns:
        Status message string.
    """
    images, cancelled = _load_with_progress(parent, paths, "Making Median Image")
    if cancelled:
        return "Median image cancelled"
    if not images:
        return "No valid images to process"

    result = make_median(images)

    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Median Image", "median.png", "Images (*.png *.jpg *.bmp)")
    if not save_path:
        return "Median image not saved"

    cv2.imwrite(save_path, result)
    return f"Median image saved — {len(images)} images, {save_path}"


def make_mean_image(parent, paths):
    """Load stack, compute mean image, prompt user to save.

    Returns:
        Status message string.
    """
    images, cancelled = _load_with_progress(parent, paths, "Making Mean Image")
    if cancelled:
        return "Mean image cancelled"
    if not images:
        return "No valid images to process"

    result = make_mean(images)

    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Mean Image", "mean.png", "Images (*.png *.jpg *.bmp)")
    if not save_path:
        return "Mean image not saved"

    cv2.imwrite(save_path, result)
    return f"Mean image saved — {len(images)} images, {save_path}"


class _AnomalyConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detect Anomalies")

        layout = QFormLayout(self)

        self._method_box = QComboBox()
        self._method_box.addItem("Mean + Std Dev (Z-score)", "mean")
        self._method_box.addItem("Median + MAD (robust)", "median")
        layout.addRow("Reference method:", self._method_box)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.5, 20.0)
        self._threshold_spin.setSingleStep(0.5)
        self._threshold_spin.setValue(2.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setSuffix("  σ")
        layout.addRow("Threshold:", self._threshold_spin)

        self._folder = None
        self._folder_label = QLabel("No folder selected")
        self._folder_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(browse_btn)
        folder_row.addWidget(self._folder_label, 1)
        layout.addRow("Output folder:", folder_row)

        self._normalize_check = QCheckBox("Normalize by threshold (σ-scaled)")
        self._normalize_check.setChecked(True)
        self._normalize_check.toggled.connect(self._on_normalize_toggled)
        layout.addRow("", self._normalize_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._ok_btn = buttons.button(QDialogButtonBox.Ok)
        self._ok_btn.setEnabled(False)

    def _on_normalize_toggled(self, checked):
        self._threshold_spin.setEnabled(checked)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        self._folder = folder
        self._folder_label.setText(folder)
        self._ok_btn.setEnabled(True)

    def method(self):
        return self._method_box.currentData()

    def threshold(self):
        return self._threshold_spin.value()

    def output_folder(self):
        return self._folder

    def normalize(self):
        return self._normalize_check.isChecked()


def detect_anomalies(parent, paths):
    """Show config dialog, load stack, compute and save anomaly heatmaps.

    Returns:
        Status message string.
    """
    dlg = _AnomalyConfigDialog(parent)
    if dlg.exec() != QDialog.Accepted:
        return "Anomaly detection cancelled"

    method = dlg.method()
    threshold = dlg.threshold()
    normalize = dlg.normalize()
    output_folder = dlg.output_folder()

    images, cancelled = _load_with_progress(parent, paths, "Detecting Anomalies")
    if cancelled:
        return "Anomaly detection cancelled"
    if not images:
        return "No valid images to process"

    n = len(images)
    progress = _progress_dialog(parent, "Detecting Anomalies", n)
    progress.setLabelText("Computing anomaly maps...")

    def cb(current, total, _):
        if progress.wasCanceled():
            return False
        progress.setLabelText(
            f"Computing anomaly map {current + 1}/{total}: "
            f"{os.path.basename(paths[current])}")
        progress.setValue(current)
        QApplication.processEvents()
        return True

    heatmaps = compute_anomaly_maps(images, method, threshold, normalize, cb)
    progress.close()

    if heatmaps is None:
        return "Anomaly detection cancelled"

    # Clear output folder then save
    for entry in os.listdir(output_folder):
        entry_path = os.path.join(output_folder, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)

    for i, heatmap in enumerate(heatmaps):
        cv2.imwrite(os.path.join(output_folder, f"anomaly_{i + 1:03d}.png"), heatmap)

    method_label = "Z-score" if method == "mean" else "MAD"
    return (
        f"Anomaly detection done — {n} maps saved to {output_folder} "
        f"(method: {method_label}, threshold: {threshold:.1f}σ)"
    )
