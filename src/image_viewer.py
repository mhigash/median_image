"""ImageViewer widget — displays an image with ROI selection, match overlays, zoom and pan."""

from PySide6.QtCore import Qt, QRect, QPoint, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QCursor
from PySide6.QtWidgets import QWidget, QScrollArea


class ImageViewer(QWidget):
    """Widget that displays an image and supports ROI selection, match overlays, zoom and pan."""

    _HANDLE_SIZE = 8  # px, full side length of handle squares
    _HANDLE_NAMES = [
        "top_left", "top_center", "top_right",
        "middle_left", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    _CURSOR_MAP = {
        "top_left": Qt.SizeFDiagCursor,
        "top_center": Qt.SizeVerCursor,
        "top_right": Qt.SizeBDiagCursor,
        "middle_left": Qt.SizeHorCursor,
        "middle_right": Qt.SizeHorCursor,
        "bottom_left": Qt.SizeBDiagCursor,
        "bottom_center": Qt.SizeVerCursor,
        "bottom_right": Qt.SizeFDiagCursor,
        "move": Qt.SizeAllCursor,
    }

    _MIN_SCALE = 0.05
    _MAX_SCALE = 20.0
    _ZOOM_FACTOR = 1.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._roi_rect = QRect()
        self._match_rects = []
        self._roi_mode = True
        self._drag_mode = None  # None | "creating" | "moving" | "handle_XX"
        self._drag_origin = QPoint()
        self._drag_rect_origin = QRect()
        self._scale = 1.0
        self._panning = False
        self._pan_start = QPoint()
        self._pan_scroll_origin = QPoint()
        self._space_held = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # --- coordinate helpers ---

    def _to_image(self, widget_pos):
        """Convert widget-space position to image-space QPoint."""
        return QPoint(int(widget_pos.x() / self._scale),
                      int(widget_pos.y() / self._scale))

    def _to_widget_pt(self, image_pos):
        """Convert image-space position to widget-space QPointF."""
        return QPointF(image_pos.x() * self._scale,
                       image_pos.y() * self._scale)

    def _rect_to_widget(self, image_rect):
        """Convert image-space QRect to widget-space QRectF."""
        return QRectF(image_rect.x() * self._scale,
                      image_rect.y() * self._scale,
                      image_rect.width() * self._scale,
                      image_rect.height() * self._scale)

    # --- zoom/scale ---

    def _update_size(self):
        """Resize widget to match pixmap * scale."""
        if self._pixmap.isNull():
            return
        w = int(self._pixmap.width() * self._scale)
        h = int(self._pixmap.height() * self._scale)
        self.setFixedSize(w, h)
        self.update()

    def get_scale(self):
        return self._scale

    def set_scale(self, scale, anchor_widget=None):
        """Set zoom scale, optionally keeping the image point under anchor_widget fixed."""
        scale = max(self._MIN_SCALE, min(self._MAX_SCALE, scale))
        if scale == self._scale:
            return
        old_scale = self._scale

        scroll = self._get_scroll_area()
        # Compute image point under anchor before resize
        img_anchor = None
        viewport_anchor = None
        if anchor_widget is not None and scroll is not None:
            viewport_anchor = anchor_widget
            img_anchor = QPointF(anchor_widget.x() / old_scale,
                                 anchor_widget.y() / old_scale)

        self._scale = scale
        self._update_size()

        # Adjust scrollbars so the image point stays under the cursor
        if img_anchor is not None and scroll is not None:
            new_widget_x = img_anchor.x() * scale
            new_widget_y = img_anchor.y() * scale
            # viewport_anchor is the widget-space position that was under the cursor
            # We need the viewport position of that point
            vp_x = viewport_anchor.x() - scroll.horizontalScrollBar().value()
            vp_y = viewport_anchor.y() - scroll.verticalScrollBar().value()
            scroll.horizontalScrollBar().setValue(int(new_widget_x - vp_x))
            scroll.verticalScrollBar().setValue(int(new_widget_y - vp_y))

    def zoom_in(self):
        self.set_scale(self._scale * self._ZOOM_FACTOR)

    def zoom_out(self):
        self.set_scale(self._scale / self._ZOOM_FACTOR)

    def zoom_reset(self):
        self.set_scale(1.0)

    def zoom_fit(self, viewport_size):
        if self._pixmap.isNull():
            return
        sx = viewport_size.width() / self._pixmap.width()
        sy = viewport_size.height() / self._pixmap.height()
        self.set_scale(min(sx, sy))

    # --- scroll area helper ---

    def _get_scroll_area(self):
        """Walk parent chain to find the enclosing QScrollArea."""
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QScrollArea):
                return w
            w = w.parentWidget()
        return None

    # --- public API ---

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._match_rects = []
        self._roi_rect = QRect()
        self._update_size()

    def set_match_rects(self, rects):
        self._match_rects = rects
        self.update()

    def get_roi_rect(self):
        return self._roi_rect

    def get_pixmap(self):
        return self._pixmap

    def set_roi_mode(self, enabled):
        self._roi_mode = enabled

    # --- handle helpers ---

    def _handle_positions(self):
        """Return dict mapping handle name -> center QPoint in image coords."""
        r = self._roi_rect
        if r.isNull():
            return {}
        cx = (r.left() + r.right()) // 2
        cy = (r.top() + r.bottom()) // 2
        return {
            "top_left": QPoint(r.left(), r.top()),
            "top_center": QPoint(cx, r.top()),
            "top_right": QPoint(r.right(), r.top()),
            "middle_left": QPoint(r.left(), cy),
            "middle_right": QPoint(r.right(), cy),
            "bottom_left": QPoint(r.left(), r.bottom()),
            "bottom_center": QPoint(cx, r.bottom()),
            "bottom_right": QPoint(r.right(), r.bottom()),
        }

    def _handle_positions_widget(self):
        """Return dict mapping handle name -> center QPointF in widget coords."""
        result = {}
        for name, img_pt in self._handle_positions().items():
            result[name] = self._to_widget_pt(img_pt)
        return result

    def _hit_test(self, widget_pos):
        """Return handle name, 'move', or None. Tests in widget space."""
        if self._roi_rect.isNull():
            return None
        # Hit test in widget space; handle visual size is constant screen pixels
        hs = self._HANDLE_SIZE // 2
        for name, center in self._handle_positions_widget().items():
            if abs(widget_pos.x() - center.x()) <= hs and abs(widget_pos.y() - center.y()) <= hs:
                return name
        roi_widget = self._rect_to_widget(self._roi_rect)
        if roi_widget.contains(QPointF(widget_pos.x(), widget_pos.y())):
            return "move"
        return None

    # --- painting ---

    def paintEvent(self, event):
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        s = self._scale

        # Draw scaled pixmap
        target = QRect(0, 0, int(self._pixmap.width() * s), int(self._pixmap.height() * s))
        painter.drawPixmap(target, self._pixmap)

        # Use painter transform so we draw in image coordinates.
        # Cosmetic pens keep constant screen-pixel width at any zoom.
        painter.scale(s, s)

        # Draw current ROI selection
        if not self._roi_rect.isNull() and self._roi_mode:
            pen = QPen(Qt.blue, 2, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(self._roi_rect))

            # Draw resize handles (constant screen size)
            hs = self._HANDLE_SIZE / 2.0 / s  # half-size in image coords
            handle_pen = QPen(Qt.blue, 1)
            handle_pen.setCosmetic(True)
            painter.setPen(handle_pen)
            painter.setBrush(QBrush(Qt.white))
            for center in self._handle_positions().values():
                painter.drawRect(QRectF(center.x() - hs, center.y() - hs,
                                        hs * 2, hs * 2))

        # Draw match rectangles
        pen = QPen(Qt.green, 2, Qt.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for rect in self._match_rects:
            painter.drawRect(QRectF(rect))

        painter.end()

    # --- key events (pan with space) ---

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            if not self._panning:
                self.setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    # --- mouse events ---

    def mousePressEvent(self, event):
        pos = event.position().toPoint()

        # Pan: space+left or middle button
        if (event.button() == Qt.MiddleButton or
                (event.button() == Qt.LeftButton and self._space_held)):
            scroll = self._get_scroll_area()
            if scroll is not None:
                self._panning = True
                self._pan_start = event.globalPosition().toPoint()
                self._pan_scroll_origin = QPoint(
                    scroll.horizontalScrollBar().value(),
                    scroll.verticalScrollBar().value(),
                )
                self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton or not self._roi_mode or self._pixmap.isNull():
            return

        hit = self._hit_test(pos)

        if hit and hit != "move":
            # Start handle resize
            self._drag_mode = f"handle_{hit}"
            self._drag_origin = self._to_image(pos)
            self._drag_rect_origin = QRect(self._roi_rect)
        elif hit == "move":
            # Start move
            self._drag_mode = "moving"
            self._drag_origin = self._to_image(pos)
            self._drag_rect_origin = QRect(self._roi_rect)
        else:
            # Start new selection
            self._drag_mode = "creating"
            self._drag_origin = self._to_image(pos)
            self._roi_rect = QRect()
            self._match_rects = []
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        # Handle panning
        if self._panning:
            scroll = self._get_scroll_area()
            if scroll is not None:
                delta = event.globalPosition().toPoint() - self._pan_start
                scroll.horizontalScrollBar().setValue(
                    self._pan_scroll_origin.x() - delta.x())
                scroll.verticalScrollBar().setValue(
                    self._pan_scroll_origin.y() - delta.y())
            return

        if self._drag_mode is None:
            # Update cursor based on hover
            if self._space_held:
                self.setCursor(Qt.OpenHandCursor)
            elif self._roi_mode and not self._roi_rect.isNull():
                hit = self._hit_test(pos)
                if hit and hit in self._CURSOR_MAP:
                    self.setCursor(QCursor(self._CURSOR_MAP[hit]))
                else:
                    self.setCursor(Qt.ArrowCursor)
            return

        img_pos = self._to_image(pos)

        if self._drag_mode == "creating":
            self._roi_rect = QRect(self._drag_origin, img_pos).normalized()
            self.update()

        elif self._drag_mode == "moving":
            delta = img_pos - self._drag_origin
            self._roi_rect = self._drag_rect_origin.translated(delta)
            self.update()

        elif self._drag_mode.startswith("handle_"):
            handle = self._drag_mode[len("handle_"):]
            r = QRect(self._drag_rect_origin)
            dx = img_pos.x() - self._drag_origin.x()
            dy = img_pos.y() - self._drag_origin.y()

            if "left" in handle:
                r.setLeft(r.left() + dx)
            if "right" in handle:
                r.setRight(r.right() + dx)
            if "top" in handle:
                r.setTop(r.top() + dy)
            if "bottom" in handle:
                r.setBottom(r.bottom() + dy)

            self._roi_rect = r.normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        # End panning
        if self._panning and (event.button() == Qt.MiddleButton or
                               event.button() == Qt.LeftButton):
            self._panning = False
            if self._space_held:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if event.button() != Qt.LeftButton or self._drag_mode is None:
            return

        img_pos = self._to_image(event.position().toPoint())

        if self._drag_mode == "creating":
            self._roi_rect = QRect(self._drag_origin, img_pos).normalized()

        # Clamp to image bounds
        if not self._pixmap.isNull():
            self._roi_rect = self._roi_rect.intersected(self._pixmap.rect())

        self._drag_mode = None
        self.update()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Zoom with anchor at mouse position
            angle = event.angleDelta().y()
            if angle > 0:
                new_scale = self._scale * self._ZOOM_FACTOR
            elif angle < 0:
                new_scale = self._scale / self._ZOOM_FACTOR
            else:
                return
            anchor = QPointF(event.position().x(), event.position().y())
            self.set_scale(new_scale, anchor)
            # Notify parent window to update status bar
            main_win = self.window()
            if hasattr(main_win, '_update_zoom_status'):
                main_win._update_zoom_status()
            event.accept()
        else:
            # Propagate to scroll area
            event.ignore()
