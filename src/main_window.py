"""MainWindow for the Template Matching application."""

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QScrollArea,
    QFileDialog,
    QToolBar,
    QDoubleSpinBox,
    QStatusBar,
    QWidget,
    QSplitter,
    QVBoxLayout,
)

from image_viewer import ImageViewer
from template_matcher import TemplateMatcher, cv_image_to_qpixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Matching")
        self.resize(1200, 700)

        # State
        self._cv_image = None  # Template source image (BGR)
        self._template = None  # Cropped template (BGR)
        self._search_image = None  # Search target image (BGR)
        self._matcher = TemplateMatcher()

        # Dual viewer with splitter
        splitter = QSplitter(Qt.Horizontal)
        left_panel, self._template_scroll, self._template_viewer = \
            self._create_viewer_panel("Template")
        right_panel, self._search_scroll, self._search_viewer = \
            self._create_viewer_panel("Search")
        self._template_viewer.set_roi_mode(True)
        self._search_viewer.set_roi_mode(False)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        self.setCentralWidget(splitter)

        # Main toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open Template Image", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)

        save_tpl_action = QAction("Save Template", self)
        save_tpl_action.triggered.connect(self._save_template)
        toolbar.addAction(save_tpl_action)

        open_search_action = QAction("Open Search Image", self)
        open_search_action.triggered.connect(self._open_search_image)
        toolbar.addAction(open_search_action)

        run_action = QAction("Run Matching", self)
        run_action.triggered.connect(self._run_matching)
        toolbar.addAction(run_action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Threshold: "))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.80)
        self._threshold_spin.setDecimals(2)
        toolbar.addWidget(self._threshold_spin)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — open an image to begin")

    def _create_viewer_panel(self, label_text):
        """Create a panel with label, zoom toolbar, and image viewer.

        Returns (panel_widget, scroll_area, viewer).
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        viewer = ImageViewer()
        scroll = QScrollArea()
        scroll.setWidget(viewer)
        scroll.setWidgetResizable(False)

        # Panel toolbar with label and zoom buttons
        tb = QToolBar()
        tb.setMovable(False)
        tb.addWidget(QLabel(f" {label_text} "))
        tb.addSeparator()

        zoom_in = QAction("Zoom In (+)", panel)
        zoom_in.triggered.connect(lambda: (viewer.zoom_in(), self._update_zoom_status()))
        tb.addAction(zoom_in)

        zoom_out = QAction("Zoom Out (-)", panel)
        zoom_out.triggered.connect(lambda: (viewer.zoom_out(), self._update_zoom_status()))
        tb.addAction(zoom_out)

        zoom_fit = QAction("Fit", panel)
        zoom_fit.triggered.connect(
            lambda: (viewer.zoom_fit(scroll.viewport().size()), self._update_zoom_status()))
        tb.addAction(zoom_fit)

        zoom_reset = QAction("100%", panel)
        zoom_reset.triggered.connect(lambda: (viewer.zoom_reset(), self._update_zoom_status()))
        tb.addAction(zoom_reset)

        layout.addWidget(tb)
        layout.addWidget(scroll)
        return panel, scroll, viewer

    def _update_zoom_status(self):
        t_pct = int(self._template_viewer.get_scale() * 100)
        s_pct = int(self._search_viewer.get_scale() * 100)
        self.statusBar().showMessage(f"Template: {t_pct}%  |  Search: {s_pct}%")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.statusBar().showMessage("Failed to load image")
            return
        self._cv_image = img
        self._template_viewer.set_image(cv_image_to_qpixmap(img))
        self.statusBar().showMessage("Image loaded — drag to select a template region")

    def _save_template(self):
        if self._cv_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        roi = self._template_viewer.get_roi_rect()
        if roi.isNull() or roi.width() < 2 or roi.height() < 2:
            self._template = self._cv_image.copy()
            h, w = self._template.shape[:2]
        else:
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            self._template = self._cv_image[y : y + h, x : x + w].copy()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "template.png", "Images (*.png *.jpg *.bmp)"
        )
        if path:
            cv2.imwrite(path, self._template)
            self.statusBar().showMessage(f"Template saved ({w}x{h}) — now open a search image")
        else:
            self.statusBar().showMessage(f"Template captured ({w}x{h}) — now open a search image")

    def _open_search_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Search Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.statusBar().showMessage("Failed to load search image")
            return
        self._search_image = img
        self._search_viewer.set_image(cv_image_to_qpixmap(img))
        self.statusBar().showMessage("Search image loaded — click Run Matching")

    def _run_matching(self):
        if self._template is None and self._cv_image is not None:
            roi = self._template_viewer.get_roi_rect()
            if roi.isNull() or roi.width() < 2 or roi.height() < 2:
                self._template = self._cv_image.copy()
            else:
                x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
                self._template = self._cv_image[y : y + h, x : x + w].copy()
        if self._template is None:
            self.statusBar().showMessage("No template — open an image first")
            return
        if self._search_image is None:
            self.statusBar().showMessage("No search image loaded")
            return

        threshold = self._threshold_spin.value()
        display, boxes = self._matcher.run(self._search_image, self._template, threshold)

        self._search_viewer.set_image(cv_image_to_qpixmap(display))

        self.statusBar().showMessage(f"Found {len(boxes)} match(es) at threshold {threshold:.2f}")
