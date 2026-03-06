# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM3 is a Python wrapper around Ultralytics' SAM3 (Segment Anything Model 3) for text-prompted and exemplar-based image segmentation. The project is a single-module tool, not a packaged library.

## Running

```bash
# Run the main script (includes usage examples in __main__)
python sam3.py

# Download model weights
python download.py
```

运行环境为 conda 环境 `yoloV8`：

```bash
conda activate yoloV8
```

There is no formal build system, no `requirements.txt`, no test suite, and no linting setup. Dependencies (ultralytics, opencv-python, numpy) already installed in the conda env.

The model file `sam3.pt` (~3.5GB) must be present in the project root.

## Architecture

**`sam3.py`** — the entire codebase is a single file with two components:

1. **`SAM3Segmentor` class** — Wraps `ultralytics.models.sam.SAM3SemanticPredictor` with:
   - `predict()` — text-prompted segmentation. Accepts image path or BGR numpy array + text prompts. Returns `(masks, boxes)` where masks are `(N, H, W)` and boxes are `(N, 4)` xyxy format, or `(None, None)`.
   - `predict_with_exemplar()` — bbox exemplar-based segmentation. Same return format.
   - Image feature caching via `_current_image` — avoids re-encoding when running multiple prompts on the same image. Use `force_reload=False` to reuse cached features.

2. **`draw_masks_on_image()` function** — Visualization utility that overlays colored masks, contours, bounding boxes, and text labels onto an image. All image I/O uses BGR format (OpenCV convention).

**`download.py`** — one-liner to download the SAM3 model weights via Ultralytics.

## Key Conventions

- All images are BGR format (OpenCV). numpy array inputs must be BGR.
- When a numpy array is passed as input, it is saved to a temp file internally (SAM3SemanticPredictor requires a file path).
- Docstrings and comments are in Chinese.
- The `3/` directory contains 200+ test JPG images; `result_*.jpg` files are output artifacts.
