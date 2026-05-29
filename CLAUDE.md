# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Planktoscope is a PyQt5 desktop tool for interactively previewing and tuning microscope image segmentation parameters, then batch-cutting individual plankton organisms from composite microscope images. Originally extracted from the ALGAVISION project.

## Running

```bash
python plankton_viewer.py
```

## Dependencies

```bash
pip install PyQt5 opencv-python numpy imageio
```

## Architecture

**plankton_viewer.py** — Main application (`PlanktoscopeSegmentViewer`). A PyQt5 `QMainWindow` with:
- Left panel: folder selection, binarization method selector (Adaptive Threshold / Otsu / Global Threshold / Canny Edge), method-specific parameters, common parameters (min particle size in pixels, blur kernel, morphology iterations), "Show binarized image" toggle, batch cut button. Parameter visibility adapts to the selected method.
- Center panel: image display with green bounding box overlay showing detected particles, plus prev/next navigation.
- Detection pipeline: grayscale → Gaussian blur → adaptive threshold → morphological close → contour detection → area filtering. Implemented in the standalone `detect_contours()` function so it can be reused outside the GUI.
- Batch mode: iterates all images in a folder, applies current parameters, crops particles, saves as `{basename}_{NNN}.png` to a chosen output folder.
- Falls back to `imageio` when OpenCV cannot read a format (e.g. some TIF variants).

**imagecut.py** — Standalone cutting functions (not imported by the viewer):
- `Microscope_Cut(path_image, th_size)` — same detection pipeline as above but returns cropped images directly. Hard-coded blur/block/C/morph parameters.
- `Flowcam_Cut(tif_path)` — simpler pipeline for FlowCAM images: binary threshold on background, then bounding-box crop per contour.

## Configuration

- **planktoscope.cfg** — JSON file auto-saved on exit and auto-loaded on startup. Stores all detection parameters (method, th_size, blur_kernel, adapt_block, adapt_C, morph_iter, global_thresh, canny_low, canny_high) and the last-used image folder. Located alongside the script.

## Key Design Notes

- `detect_contours()` in the viewer is the parameterized, reusable version of the hard-coded logic in `imagecut.Microscope_Cut()`. If modifying detection behavior, update both or migrate callers to `detect_contours()`.
- Image format support is defined in `IMG_FORMATS` set (tif, jpg, png, bmp, gif, jfif).
- All OpenCV images use BGR color space internally; conversion to RGB happens only at the Qt display boundary (`cv2_to_qpixmap`).
