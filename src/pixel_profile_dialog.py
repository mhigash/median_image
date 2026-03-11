"""PixelProfileDialog — shows pixel value line plot and histogram across an image stack."""

import cv2
import numpy as np
from scipy.stats import norm
from PySide6.QtWidgets import QDialog, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PixelProfileDialog(QDialog):
    """Modeless dialog displaying pixel value profile across image stack."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Profile")
        self.resize(600, 400)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def update_profile(self, x, y, paths):
        """Read pixel (x, y) from every image in paths and update plots."""
        values = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                values.append(None)
                continue
            if y < img.shape[0] and x < img.shape[1]:
                values.append(img[y, x].copy())
            else:
                values.append(None)

        # Filter out None entries
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            return
        valid_values = np.array([values[i] for i in valid_indices])

        self._figure.clear()
        is_color = valid_values.ndim == 2 and valid_values.shape[1] >= 3

        ax_line = self._figure.add_subplot(1, 2, 1)
        ax_hist = self._figure.add_subplot(1, 2, 2)

        bin_edges = range(0, 257, 4)
        bin_width = 4
        fit_x = np.linspace(0, 255, 256)

        if is_color:
            colors = ['blue', 'green', 'red']
            labels = ['B', 'G', 'R']
            for ch in range(3):
                ch_vals = valid_values[:, ch].astype(float)
                ax_line.plot(valid_indices, ch_vals, color=colors[ch], label=labels[ch], linewidth=0.8)
                ax_hist.hist(ch_vals, bins=bin_edges, color=colors[ch],
                             alpha=0.4, label=labels[ch])
                mu, std = ch_vals.mean(), ch_vals.std()
                if std > 0:
                    fit_y = norm.pdf(fit_x, mu, std) * len(ch_vals) * bin_width
                    ax_hist.plot(fit_x, fit_y, color=colors[ch], linewidth=1.2)
        else:
            vals = valid_values.flatten().astype(float) if valid_values.ndim == 1 else valid_values[:, 0].astype(float)
            ax_line.plot(valid_indices, vals, color='gray', linewidth=0.8)
            ax_hist.hist(vals, bins=bin_edges, color='gray', alpha=0.6)
            mu, std = vals.mean(), vals.std()
            if std > 0:
                fit_y = norm.pdf(fit_x, mu, std) * len(vals) * bin_width
                ax_hist.plot(fit_x, fit_y, color='black', linewidth=1.2)

        ax_line.set_xlabel("Image index")
        ax_line.set_ylabel("Pixel value")
        ax_line.set_ylim(-5, 260)
        ax_line.set_title(f"Pixel ({x}, {y})")
        if is_color:
            ax_line.legend(fontsize='small')

        ax_hist.set_xlabel("Pixel value")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Histogram")
        if is_color:
            ax_hist.legend(fontsize='small')

        self._canvas.draw()
