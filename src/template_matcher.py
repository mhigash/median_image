"""Template matching using OpenCV."""

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPixmap


def cv_image_to_qpixmap(cv_img):
    """Convert an OpenCV BGR image to QPixmap."""
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def non_max_suppression(boxes, overlap_thresh=0.3):
    """Suppress overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    area = boxes_arr[:, 2] * boxes_arr[:, 3]

    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return [boxes[i] for i in pick]


class TemplateMatcher:
    """Performs template matching on a search image."""

    def run(self, search_image, template, threshold):
        """Run template matching.

        Args:
            search_image: BGR numpy array of the image to search in.
            template: BGR numpy array of the template to find.
            threshold: float, correlation threshold (0.0 - 1.0).

        Returns:
            (display_image, boxes) where display_image is a BGR numpy array
            with matches drawn, and boxes is a list of (x, y, w, h) tuples.
        """
        tpl_h, tpl_w = template.shape[:2]

        result = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        boxes = [(int(x), int(y), tpl_w, tpl_h) for y, x in zip(*locations)]
        boxes = non_max_suppression(boxes, overlap_thresh=0.3)

        display = search_image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return display, boxes
