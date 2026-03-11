"""MainWindow for the Template Matching application."""

import glob
import os

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QScrollArea,
    QFileDialog,
    QToolBar,
    QStatusBar,
    QWidget,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QDialog,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFormLayout,
)

from image_viewer import ImageViewer
from pixel_profile_dialog import PixelProfileDialog
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
        self._match_boxes = []

        # Image stack state
        self._stack_paths = []
        self._stack_index = 0
        self._stack_panel = None
        self._stack_viewer = None
        self._stack_scroll = None
        self._profile_dialog = None

        # Dual viewer with splitter
        self._splitter = QSplitter(Qt.Horizontal)
        left_panel, self._template_scroll, self._template_viewer = \
            self._create_viewer_panel("Template")
        right_panel, self._search_scroll, self._search_viewer = \
            self._create_viewer_panel("Search")
        self._template_viewer.set_roi_mode(True)
        self._search_viewer.set_roi_mode(False)
        self._splitter.addWidget(left_panel)
        self._splitter.addWidget(right_panel)
        self._splitter.setSizes([600, 600])

        # Central stacked widget (page 0 = template matching, page 1 = image stack)
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self._splitter)
        self.setCentralWidget(self._central_stack)

        # Menu bar (embedded in window, not native macOS menu)
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu("&File")

        open_action = QAction("Open &Template Image...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        open_search_action = QAction("Open &Search Image...", self)
        open_search_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_search_action.triggered.connect(self._open_search_image)
        file_menu.addAction(open_search_action)

        open_folder_action = QAction("Open Image &Folder...", self)
        open_folder_action.setShortcut(QKeySequence("Ctrl+D"))
        open_folder_action.triggered.connect(self._open_image_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        self._back_to_template_action = QAction("&Back to Template Matching", self)
        self._back_to_template_action.triggered.connect(self._switch_to_template_mode)
        self._back_to_template_action.setEnabled(False)
        file_menu.addAction(self._back_to_template_action)

        file_menu.addSeparator()

        save_tpl_action = QAction("Save Template &As...", self)
        save_tpl_action.setShortcut(QKeySequence("Ctrl+S"))
        save_tpl_action.triggered.connect(self._save_template)
        file_menu.addAction(save_tpl_action)

        file_menu.addSeparator()

        save_matches_action = QAction("Save &Matches...", self)
        save_matches_action.triggered.connect(self._save_matches)
        file_menu.addAction(save_matches_action)

        matching_menu = menubar.addMenu("&Matching")

        run_action = QAction("&Run Matching...", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._run_matching)
        matching_menu.addAction(run_action)

        # Main toolbar (actions shared with menu)
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        toolbar.addAction(open_search_action)
        toolbar.addAction(open_folder_action)
        toolbar.addAction(save_tpl_action)
        toolbar.addAction(save_matches_action)
        toolbar.addSeparator()
        toolbar.addAction(run_action)

        # State for threshold (remembered across dialogs)
        self._threshold = 0.80

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
        self._switch_to_template_mode()
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
        self._switch_to_template_mode()
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

        # Show threshold dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Run Matching")
        layout = QFormLayout(dlg)

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setValue(self._threshold)
        spin.setDecimals(2)
        layout.addRow("Threshold:", spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        self._threshold = spin.value()
        display, boxes = self._matcher.run(self._search_image, self._template, self._threshold)
        self._match_boxes = boxes

        self._search_viewer.set_image(cv_image_to_qpixmap(display))

        self.statusBar().showMessage(f"Found {len(boxes)} match(es) at threshold {self._threshold:.2f}")

    def _save_matches(self):
        if self._search_image is None:
            self.statusBar().showMessage("No search image loaded")
            return
        if not self._match_boxes:
            self.statusBar().showMessage("No matches to save — run matching first")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        # Find the highest existing match_NNN.png number in the folder
        existing = glob.glob(os.path.join(folder, "match_[0-9][0-9][0-9].png"))
        start = 0
        for path in existing:
            base = os.path.basename(path)
            num = int(base[6:9])  # "match_NNN.png" -> NNN
            if num > start:
                start = num
        for i, (x, y, w, h) in enumerate(self._match_boxes, start=start + 1):
            crop = self._search_image[y : y + h, x : x + w].copy()
            filename = os.path.join(folder, f"match_{i:03d}.png")
            cv2.imwrite(filename, crop)
        self.statusBar().showMessage(
            f"Saved {len(self._match_boxes)} match image(s) to {folder}"
        )

    def _open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not folder:
            return
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.sort()
        if not paths:
            self.statusBar().showMessage("No images found in selected folder")
            return
        self._stack_paths = paths
        self._stack_index = 0
        if self._stack_panel is None:
            self._stack_panel, self._stack_scroll, self._stack_viewer = \
                self._create_viewer_panel("Stack")
            self._stack_viewer.set_roi_mode(False)
            self._stack_viewer.navigate.connect(self._navigate_stack)
            self._stack_viewer.pixel_clicked.connect(self._on_stack_pixel_clicked)
            self._central_stack.addWidget(self._stack_panel)
        self._central_stack.setCurrentIndex(1)
        self._back_to_template_action.setEnabled(True)
        self._show_stack_image()
        self._stack_viewer.setFocus()

    def _switch_to_template_mode(self):
        self._central_stack.setCurrentIndex(0)
        self._back_to_template_action.setEnabled(False)

    def _show_stack_image(self):
        path = self._stack_paths[self._stack_index]
        img = cv2.imread(path)
        if img is None:
            self.statusBar().showMessage(f"Failed to load {os.path.basename(path)}")
            return
        self._stack_viewer.set_image(cv_image_to_qpixmap(img))
        total = len(self._stack_paths)
        name = os.path.basename(path)
        self.statusBar().showMessage(
            f"Image {self._stack_index + 1}/{total} \u2014 {name}"
        )

    def _navigate_stack(self, direction):
        """Handle arrow key navigation in stack mode. direction: -1 or +1."""
        if self._central_stack.currentIndex() != 1 or not self._stack_paths:
            return
        new_index = self._stack_index + direction
        if 0 <= new_index < len(self._stack_paths):
            self._stack_index = new_index
            self._show_stack_image()

    def _on_stack_pixel_clicked(self, point):
        """Show pixel profile dialog for the clicked pixel across all stack images."""
        if not self._stack_paths:
            return
        if self._profile_dialog is None or not self._profile_dialog.isVisible():
            self._profile_dialog = PixelProfileDialog(self)
        self._profile_dialog.update_profile(point.x(), point.y(), self._stack_paths)
        self._profile_dialog.show()
        self._profile_dialog.raise_()
