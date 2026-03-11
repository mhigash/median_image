"""Template Matching Wizard — guides through the full matching workflow."""

import glob
import os

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWizard,
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QDoubleSpinBox,
    QFormLayout,
    QScrollArea,
    QProgressDialog,
    QApplication,
)

from image_viewer import ImageViewer
from template_matcher import TemplateMatcher, cv_image_to_qpixmap


class TemplateSelectPage(QWizardPage):
    """Page 1: Select a template image and optionally draw an ROI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Template Image")
        self.setSubTitle("Choose an image and optionally draw a region of interest.")

        self._image_path = None
        self._cv_image = None

        layout = QVBoxLayout(self)

        btn_layout = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse)
        btn_layout.addWidget(self._browse_btn)
        self._path_label = QLabel("No file selected")
        self._path_label.setWordWrap(True)
        btn_layout.addWidget(self._path_label, 1)
        layout.addLayout(btn_layout)

        self._viewer = ImageViewer()
        self._viewer.set_roi_mode(True)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._viewer)
        self._scroll.setWidgetResizable(False)
        layout.addWidget(self._scroll, 1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Template Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)",
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self._image_path = path
        self._cv_image = img
        self._path_label.setText(os.path.basename(path))
        self._viewer.set_image(cv_image_to_qpixmap(img))
        self._viewer.zoom_fit(self._scroll.viewport().size())
        self.completeChanged.emit()

    def isComplete(self):
        return self._cv_image is not None

    def get_template(self):
        """Return the template image (cropped to ROI if drawn)."""
        if self._cv_image is None:
            return None
        roi = self._viewer.get_roi_rect()
        if roi.isNull() or roi.width() < 2 or roi.height() < 2:
            return self._cv_image.copy()
        x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
        return self._cv_image[y : y + h, x : x + w].copy()


class SearchImagesPage(QWizardPage):
    """Page 2: Select one or more search images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Search Images")
        self.setSubTitle("Add the images to search for template matches.")

        layout = QVBoxLayout(self)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self._add_files)
        btn_layout.addWidget(add_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._list = QListWidget()
        layout.addWidget(self._list, 1)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Search Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)",
        )
        if not paths:
            return
        for p in paths:
            self._list.addItem(p)
        self.completeChanged.emit()

    def _clear(self):
        self._list.clear()
        self.completeChanged.emit()

    def isComplete(self):
        return self._list.count() > 0

    def get_paths(self):
        return [self._list.item(i).text() for i in range(self._list.count())]


class ParametersPage(QWizardPage):
    """Page 3: Configure matching threshold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Parameters")
        self.setSubTitle("Adjust the matching threshold.")

        layout = QFormLayout(self)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0.0, 1.0)
        self._spin.setSingleStep(0.05)
        self._spin.setValue(0.80)
        self._spin.setDecimals(2)
        layout.addRow("Threshold:", self._spin)

    def get_threshold(self):
        return self._spin.value()


class DestinationPage(QWizardPage):
    """Page 4: Select output folder for saved match crops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select Destination")
        self.setSubTitle("Choose a folder to save cropped match images.")

        self._folder = None

        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        btn_layout.addWidget(browse_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._folder_label = QLabel("No folder selected")
        self._folder_label.setWordWrap(True)
        layout.addWidget(self._folder_label)
        layout.addStretch()

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        self._folder = folder
        self._folder_label.setText(folder)
        self.completeChanged.emit()

    def isComplete(self):
        return self._folder is not None

    def get_folder(self):
        return self._folder


class MatchingWizard(QWizard):
    """Four-step wizard for batch template matching."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Template Matching Wizard")
        self.resize(700, 500)

        self._template_page = TemplateSelectPage()
        self._search_page = SearchImagesPage()
        self._params_page = ParametersPage()
        self._dest_page = DestinationPage()

        self.addPage(self._template_page)
        self.addPage(self._search_page)
        self.addPage(self._params_page)
        self.addPage(self._dest_page)

        self.match_count = 0

    def accept(self):
        template = self._template_page.get_template()
        if template is None:
            super().accept()
            return

        search_paths = self._search_page.get_paths()
        threshold = self._params_page.get_threshold()
        folder = self._dest_page.get_folder()
        if not folder:
            super().accept()
            return

        matcher = TemplateMatcher()

        # Remove all files in destination folder before saving
        for entry in os.listdir(folder):
            entry_path = os.path.join(folder, entry)
            if os.path.isfile(entry_path):
                os.remove(entry_path)

        progress = QProgressDialog(
            "Matching...", "Cancel", 0, len(search_paths), self)
        progress.setWindowTitle("Running Template Matching")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        counter = 0
        total = 0
        for i, img_path in enumerate(search_paths):
            if progress.wasCanceled():
                break
            progress.setLabelText(
                f"Processing image {i + 1}/{len(search_paths)}: "
                f"{os.path.basename(img_path)}")
            QApplication.processEvents()

            img = cv2.imread(img_path)
            if img is None:
                progress.setValue(i + 1)
                continue
            _, boxes = matcher.run(img, template, threshold)
            for x, y, w, h in boxes:
                counter += 1
                crop = img[y : y + h, x : x + w].copy()
                filename = os.path.join(folder, f"match_{counter:03d}.png")
                cv2.imwrite(filename, crop)
                total += 1
            progress.setValue(i + 1)

        progress.close()
        self.match_count = total
        super().accept()
