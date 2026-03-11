# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Template Matching Application — a PySide6/OpenCV desktop app with a dual-panel image viewer for template matching.

## Running

```
source venv/bin/activate
python src/main.py
```

## Environment

- Python 3.9.6 (system CommandLineTools)
- Virtual environment: `venv/`
- Dependencies: PySide6, OpenCV (opencv-python), NumPy

## Source Structure

- `src/main.py` — Entry point
- `src/main_window.py` — `MainWindow` (dual-panel UI, toolbar, file dialogs)
- `src/image_viewer.py` — `ImageViewer` widget (image display, ROI selection, zoom/pan)
- `src/template_matcher.py` — `TemplateMatcher` class, `cv_image_to_qpixmap`, `non_max_suppression`

## Functionalities

- **Dual-panel viewer**: Horizontal `QSplitter` with Template (left) and Search (right) panels, each with independent zoom controls
- **ROI selection**: Draw, move, and resize a region-of-interest on the template image using drag handles
- **Template capture**: Save a cropped ROI as a template image; if no ROI is selected, the entire image is used
- **Template matching**: OpenCV `matchTemplate` (TM_CCOEFF_NORMED) with configurable threshold and non-max suppression; results drawn on the search image
- **Zoom/Pan**: Per-panel Zoom In/Out/Fit/100% toolbar buttons, Ctrl+wheel zoom with anchor, space+drag or middle-mouse-drag pan
- **Cosmetic pens**: ROI rectangle, handles, and match rectangles use cosmetic pens for constant screen-pixel width at any zoom level
