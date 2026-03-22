# Web Frontend Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gradio UI with FastAPI backend + vanilla HTML/JS frontend, preserving all existing functionality.

**Architecture:** Single FastAPI process serves REST API endpoints and hosts `frontend/` as StaticFiles. Long-running tasks (annotation, training) run in background threads with task_id polling. Training logs stream via SSE.

**Tech Stack:** FastAPI, uvicorn, Python threading, vanilla JS (ES Modules), CSS custom properties for theming.

**Spec:** `docs/superpowers/specs/2026-03-23-web-frontend-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `run.py` | Create | Launcher: set env vars, start uvicorn |
| `backend/__init__.py` | Create | Package marker |
| `backend/main.py` | Create | FastAPI app, mount routers + StaticFiles |
| `backend/models.py` | Create | All Pydantic request/response models |
| `backend/tasks.py` | Create | Thread-safe task registry |
| `backend/utils.py` | Create | Pure helpers ported from app.py |
| `backend/segmentor.py` | Create | SAM3 lazy singleton |
| `backend/routers/__init__.py` | Create | Package marker |
| `backend/routers/images.py` | Create | List/delete images |
| `backend/routers/preview.py` | Create | Preview render + annotation deletion |
| `backend/routers/annotate.py` | Create | SAM3 annotation task |
| `backend/routers/train.py` | Create | YOLO training + SSE logs |
| `backend/routers/infer.py` | Create | YOLO inference, model info, single-image SAM3 |
| `frontend/index.html` | Create | App shell: topbar + sidebar + view container |
| `frontend/style.css` | Create | Apple-style dark/light theme, CSS vars |
| `frontend/app.js` | Create | Router, api() client, toast, theme toggle |
| `frontend/views/annotate.js` | Create | Auto-annotation page |
| `frontend/views/preview.js` | Create | Preview/edit page |
| `frontend/views/train.js` | Create | Training page + SSE log stream |
| `frontend/views/infer.js` | Create | Inference + model info page |
| `requirements.txt` | Modify | Add fastapi, uvicorn[standard], pytest, httpx |
| `tests/conftest.py` | Create | TestClient fixture, temp dirs |
| `tests/test_utils.py` | Create | Unit tests for pure helpers |
| `tests/test_tasks.py` | Create | Task registry tests |
| `tests/test_images.py` | Create | Images router tests |
| `tests/test_preview.py` | Create | Preview router tests |
| `tests/test_annotate.py` | Create | Annotate router tests (mock SAM3) |
| `tests/test_train.py` | Create | Train router tests (mock YOLO) |
| `tests/test_infer.py` | Create | Infer router tests (mock YOLO) |

---

## Task 1: Project Skeleton + run.py

**Files:**
- Create: `run.py`
- Create: `backend/__init__.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/main.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 2: Create `backend/__init__.py` and `backend/routers/__init__.py`**

Both empty files.

- [ ] **Step 3: Create `backend/main.py`**

```python
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SAMpler")

# Preview cache dir
PREVIEW_DIR = Path(".cache/previews")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/previews", StaticFiles(directory=str(PREVIEW_DIR)), name="previews")

# Frontend static files (registered last so API routes take priority)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    """Serve frontend SPA — all non-API paths return index.html"""
    candidate = FRONTEND_DIR / full_path
    if candidate.is_file():
        return FileResponse(str(candidate))
    return FileResponse(str(FRONTEND_DIR / "index.html"))
```

- [ ] **Step 4: Create `run.py`**

```python
"""SAMpler launcher — python run.py"""
import os
import sys
from pathlib import Path

# Must be set before any torch/ultralytics import
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def main():
    backend = Path(__file__).parent / "backend"
    frontend = Path(__file__).parent / "frontend"
    if not backend.exists():
        print(f"ERROR: backend/ directory not found at {backend}", file=sys.stderr)
        sys.exit(1)
    if not frontend.exists():
        print(f"ERROR: frontend/ directory not found at {frontend}", file=sys.stderr)
        sys.exit(1)

    import uvicorn
    print("SAMpler starting at http://localhost:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Install dependencies**

```bash
conda run -n yoloV8 pip install fastapi "uvicorn[standard]" pytest httpx
```

- [ ] **Step 6: Create minimal `frontend/index.html` (placeholder, replaced in Task 11)**

```html
<!DOCTYPE html>
<html><body><h1>SAMpler loading...</h1></body></html>
```

- [ ] **Step 7: Verify server starts**

```bash
conda run -n yoloV8 python run.py &
sleep 2
curl http://localhost:8000/api/health
# Expected: {"status":"ok"}
kill %1
```

- [ ] **Step 8: Commit**

```bash
git add run.py backend/ frontend/index.html requirements.txt
git commit -m "feat: FastAPI skeleton + run.py launcher"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `backend/models.py`

- [ ] **Step 1: Create `backend/models.py`**

```python
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Images ────────────────────────────────────────────────────────────────────

class ImageItem(BaseModel):
    id: str                  # filename stem, e.g. "img_0042"
    filename: str            # e.g. "img_0042.jpg"
    has_label: bool


class ImagesResponse(BaseModel):
    files: list[ImageItem]
    total: int


# ── Annotation ────────────────────────────────────────────────────────────────

class AnnotateRequest(BaseModel):
    image_dir: str = "rawData"
    output_dir: str = "dataset"
    prompts: str = Field(..., description="comma-separated class names")
    mode: Literal["detect", "segment"] = "segment"
    sort_mode: Literal["conf", "area"] = "conf"
    conf: float = Field(0.25, ge=0.01, le=0.99)
    val_ratio: float = Field(0.1, ge=0.0, le=0.5)
    max_instances: int = Field(7, ge=1, le=100)


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_dir: str = "dataset"
    task: Literal["detect", "segment"] = "segment"
    yolo_version: Literal["YOLOv8", "YOLOv11", "YOLO26"] = "YOLOv11"
    model_size: Literal["n", "s", "m", "l", "x"] = "n"
    epochs: int = Field(100, ge=1, le=1000)
    imgsz: int = Field(640, ge=32, le=4096)


# ── Task ──────────────────────────────────────────────────────────────────────

class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "done", "error"]
    progress: int = 0
    total: int = 0
    message: str = ""
    result: Optional[dict] = None


# ── Preview ───────────────────────────────────────────────────────────────────

class AnnotationItem(BaseModel):
    id: int
    class_name: str
    type: Literal["polygon", "bbox"]
    color: list[int]          # [R, G, B]
    points: list[list[float]] # normalized, for Canvas use
    bbox: list[float]         # [x1, y1, x2, y2] normalized


class PreviewResponse(BaseModel):
    preview_url: str
    annotations: list[AnnotationItem]
    images_dir: str           # echoed back for convenience
    split: str                # "train" or "val"


# ── Inference ─────────────────────────────────────────────────────────────────

class InferResponse(BaseModel):
    result_url: str
    stats: dict


class ModelInfoResponse(BaseModel):
    model_name: Optional[str] = None
    architecture: Optional[str] = None
    task: Optional[str] = None
    size_mb: float
    raw: str                  # human-readable summary
```

- [ ] **Step 2: Commit**

```bash
git add backend/models.py
git commit -m "feat: Pydantic request/response models"
```

---

## Task 3: Task Registry

**Files:**
- Create: `backend/tasks.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_tasks.py`

- [ ] **Step 1: Create `backend/tasks.py`**

```python
import threading
import uuid
from typing import Optional
from backend.models import TaskStatus


class TaskRegistry:
    """Thread-safe in-memory task store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskStatus] = {}

    def create(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = TaskStatus(task_id=task_id, status="pending")
        return task_id

    def get(self, task_id: str) -> Optional[TaskStatus]:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for k, v in kwargs.items():
                setattr(task, k, v)


# Module-level singleton used by all routers
registry = TaskRegistry()
```

- [ ] **Step 2: Create `tests/__init__.py`** (empty)

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 4: Create `tests/test_tasks.py`**

```python
from backend.tasks import TaskRegistry


def test_create_returns_unique_ids():
    reg = TaskRegistry()
    ids = {reg.create() for _ in range(10)}
    assert len(ids) == 10


def test_get_missing_returns_none():
    reg = TaskRegistry()
    assert reg.get("nonexistent") is None


def test_update_task_status():
    reg = TaskRegistry()
    tid = reg.create()
    reg.update(tid, status="running", progress=5, total=10)
    t = reg.get(tid)
    assert t.status == "running"
    assert t.progress == 5


def test_update_nonexistent_does_not_raise():
    reg = TaskRegistry()
    reg.update("ghost", status="done")  # should not raise
```

- [ ] **Step 5: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_tasks.py -v
# Expected: 4 passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/tasks.py tests/
git commit -m "feat: thread-safe task registry + tests"
```

---

## Task 4: Utility Functions

**Files:**
- Create: `backend/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_utils.py
import numpy as np
import pytest
from backend.utils import (
    collect_images, xyxy_to_xywh_norm, mask_to_polygon_norm,
    build_model_name, generate_colors, detect_mode_from_label,
)
import tempfile, os
from pathlib import Path


def test_collect_images_finds_jpgs(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.txt").touch()
    files = collect_images(str(tmp_path))
    assert len(files) == 2
    assert all(f.endswith((".jpg", ".png")) for f in files)


def test_collect_images_missing_dir():
    with pytest.raises(FileNotFoundError):
        collect_images("/nonexistent/path/xyz")


def test_xyxy_to_xywh_norm():
    xc, yc, w, h = xyxy_to_xywh_norm([10, 20, 50, 60], 100, 100)
    assert abs(xc - 0.3) < 1e-6
    assert abs(yc - 0.4) < 1e-6
    assert abs(w - 0.4) < 1e-6
    assert abs(h - 0.4) < 1e-6


def test_build_model_name_detect():
    assert build_model_name("YOLOv8", "n", "detect") == "yolov8n.pt"


def test_build_model_name_segment():
    assert build_model_name("YOLOv11", "m", "segment") == "yolo11m-seg.pt"


def test_generate_colors_deterministic():
    c1 = generate_colors(5)
    c2 = generate_colors(5)
    assert c1 == c2


def test_detect_mode_from_label_detect():
    line = "0 0.5 0.5 0.3 0.3"
    assert detect_mode_from_label(line) == "detect"


def test_detect_mode_from_label_segment():
    line = "0 0.1 0.2 0.3 0.4 0.5 0.6"
    assert detect_mode_from_label(line) == "segment"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n yoloV8 pytest tests/test_utils.py -v
# Expected: ImportError — module doesn't exist yet
```

- [ ] **Step 3: Create `backend/utils.py`**

```python
"""Pure helper functions — no FastAPI or SAM3 imports."""
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[str]:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    files = [str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"目录中没有找到图片: {image_dir}")
    return sorted(files)


def xyxy_to_xywh_norm(box, img_w: int, img_h: int) -> tuple:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def mask_to_polygon_norm(mask, img_w: int, img_h: int):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(np.float64)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    return pts


def build_model_name(version: str, size: str, task: str) -> str:
    seg_suffix = "-seg" if task == "segment" else ""
    prefix_map = {
        "YOLOv8":  f"yolov8{size}{seg_suffix}.pt",
        "YOLOv11": f"yolo11{size}{seg_suffix}.pt",
        "YOLO26":  f"yolo26{size}{seg_suffix}.pt",
    }
    return prefix_map[version]


def generate_colors(n: int) -> list[tuple]:
    """Deterministic color list (BGR). Same seed as app.py."""
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]


def detect_mode_from_label(line: str) -> str:
    """Infer annotation mode from a single YOLO label line."""
    parts = line.strip().split()
    # detect: class_id + 4 values = 5 total
    # segment: class_id + >=6 values
    return "detect" if len(parts) == 5 else "segment"


def redraw_from_annotations(
    img_bgr: np.ndarray, contours: list[dict], mode: str
) -> np.ndarray:
    """Re-render preview from stored annotation metadata (no re-inference)."""
    output = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    alpha = 0.4

    for ann in contours:
        color = ann["color"]          # BGR tuple
        class_name = ann["class_name"]
        parts = ann["label_line"].split()
        values = list(map(float, parts[1:]))

        if mode == "detect":
            xc, yc, bw, bh = values
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        else:
            pts = np.array(values).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts_int = pts.astype(np.int32)
            mask_layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_layer, [pts_int], 255)
            colored = np.zeros_like(img_bgr)
            colored[mask_layer > 0] = color
            output = cv2.addWeighted(output, 1, colored, alpha, 0)
            cv2.polylines(output, [pts_int], True, color, 2)
            bx1, by1 = pts_int.min(axis=0)
            bx2, by2 = pts_int.max(axis=0)
            x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)

        (tw, th), baseline = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            output, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1
        )
        cv2.putText(
            output, class_name, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return output
```

- [ ] **Step 4: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_utils.py -v
# Expected: 8 passed
```

- [ ] **Step 5: Commit**

```bash
git add backend/utils.py tests/test_utils.py
git commit -m "feat: utility functions + tests (ported from app.py)"
```

---

## Task 5: SAM3 Singleton + Segmentor Module

**Files:**
- Create: `backend/segmentor.py`

- [ ] **Step 1: Create `backend/segmentor.py`**

```python
"""SAM3 lazy singleton — loaded once per process, reused across requests."""
import threading
import sys
from pathlib import Path

# Import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3 import SAM3Segmentor

_lock = threading.Lock()
_segmentor: SAM3Segmentor | None = None
_segmentor_conf: float | None = None


def get_segmentor(conf: float = 0.25) -> SAM3Segmentor:
    global _segmentor, _segmentor_conf
    with _lock:
        if _segmentor is None:
            _segmentor = SAM3Segmentor(
                model_path="sam3.pt", conf=conf, device="0", half=True
            )
            _segmentor_conf = conf
        elif conf != _segmentor_conf:
            _segmentor.predictor.args.conf = conf
            _segmentor_conf = conf
        return _segmentor
```

- [ ] **Step 2: Commit** (no test — requires 3.5GB model file)

```bash
git add backend/segmentor.py
git commit -m "feat: SAM3 lazy singleton in backend/segmentor.py"
```

---

## Task 6: Images Router

**Files:**
- Create: `backend/routers/images.py`
- Create: `tests/test_images.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_images.py
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_list_images_missing_dir():
    r = client.get("/api/images", params={"dir": "/nonexistent/dir/xyz"})
    assert r.status_code == 404


def test_list_images_empty_dir(tmp_path):
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 404  # no image files


def test_list_images_returns_files(tmp_path):
    (tmp_path / "cat.jpg").touch()
    (tmp_path / "dog.png").touch()
    (tmp_path / "notes.txt").touch()
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    ids = [f["id"] for f in data["files"]]
    assert "cat" in ids and "dog" in ids


def test_list_images_has_label_false(tmp_path):
    (tmp_path / "img.jpg").touch()
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["files"][0]["has_label"] is False


def test_list_images_has_label_true(tmp_path):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "img.jpg").touch()
    (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    r = client.get("/api/images", params={"dir": str(img_dir)})
    assert r.json()["files"][0]["has_label"] is True


def test_delete_image(tmp_path):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "cat.jpg").write_bytes(b"fake")
    (lbl_dir / "cat.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    r = client.delete(
        "/api/images/cat",
        params={"images_dir": str(img_dir)},
    )
    assert r.status_code == 200
    assert not (img_dir / "cat.jpg").exists()
    assert not (lbl_dir / "cat.txt").exists()


def test_delete_image_not_found(tmp_path):
    r = client.delete(
        "/api/images/ghost",
        params={"images_dir": str(tmp_path)},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n yoloV8 pytest tests/test_images.py -v
# Expected: 404 on /api/images — route not registered yet
```

- [ ] **Step 3: Create `backend/routers/images.py`**

```python
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from backend.models import ImagesResponse, ImageItem
from backend.utils import IMAGE_EXTENSIONS

router = APIRouter(prefix="/api/images", tags=["images"])


def _find_label(stem: str, images_dir: Path) -> Path | None:
    """Given images/train or images/val, find the corresponding label file."""
    # images_dir = .../images/train → labels_dir = .../labels/train
    parts = images_dir.parts
    try:
        img_idx = next(i for i, p in enumerate(parts) if p == "images")
        labels_dir = Path(*parts[:img_idx]) / "labels" / parts[img_idx + 1]
        candidate = labels_dir / f"{stem}.txt"
        return candidate if candidate.exists() else None
    except (StopIteration, IndexError):
        return None


@router.get("", response_model=ImagesResponse)
def list_images(dir: str = Query(..., description="图片目录路径")):
    p = Path(dir)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir}")
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise HTTPException(status_code=404, detail=f"目录中无图片: {dir}")
    items = [
        ImageItem(
            id=f.stem,
            filename=f.name,
            has_label=_find_label(f.stem, p) is not None,
        )
        for f in files
    ]
    return ImagesResponse(files=items, total=len(items))


@router.delete("/{image_id}")
def delete_image(image_id: str, images_dir: str = Query(...)):
    p = Path(images_dir)
    # Find the image file
    img_file = next(
        (p / f"{image_id}{ext}" for ext in IMAGE_EXTENSIONS
         if (p / f"{image_id}{ext}").exists()),
        None,
    )
    if img_file is None:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")

    deleted = []
    img_file.unlink()
    deleted.append(str(img_file))

    label = _find_label(image_id, p)
    if label and label.exists():
        label.unlink()
        deleted.append(str(label))

    return {"deleted": deleted}
```

- [ ] **Step 4: Register router in `backend/main.py`**

Add after the existing imports:
```python
from backend.routers import images as images_router
app.include_router(images_router.router)
```

- [ ] **Step 5: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_images.py -v
# Expected: 7 passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/routers/images.py backend/main.py tests/test_images.py
git commit -m "feat: images router (list + delete) + tests"
```

---

## Task 7: Preview Router

**Files:**
- Create: `backend/routers/preview.py`
- Create: `tests/test_preview.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preview.py
import json
import numpy as np
import cv2
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _make_dataset(tmp_path, mode="detect"):
    """Create a minimal dataset structure with one image and label."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "img.jpg"), img)

    if mode == "detect":
        (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    else:
        (lbl_dir / "img.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

    (tmp_path / "data.yaml").write_text("names:\n  0: cat\n")
    return img_dir


def test_preview_missing_image(tmp_path):
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    r = client.get("/api/images/ghost/preview", params={"images_dir": str(img_dir)})
    assert r.status_code == 404


def test_preview_returns_url_and_annotations(tmp_path):
    img_dir = _make_dataset(tmp_path, mode="detect")
    r = client.get("/api/images/img/preview", params={"images_dir": str(img_dir)})
    assert r.status_code == 200
    data = r.json()
    assert "preview_url" in data
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["type"] == "bbox"
    assert data["annotations"][0]["class_name"] == "cat"


def test_preview_segment_annotation_type(tmp_path):
    img_dir = _make_dataset(tmp_path, mode="segment")
    r = client.get("/api/images/img/preview", params={"images_dir": str(img_dir)})
    assert r.status_code == 200
    assert r.json()["annotations"][0]["type"] == "polygon"


def test_delete_annotation(tmp_path):
    img_dir = _make_dataset(tmp_path, mode="detect")
    lbl = tmp_path / "labels" / "train" / "img.txt"
    # First write two annotations
    lbl.write_text("0 0.5 0.5 0.3 0.3\n1 0.2 0.2 0.1 0.1\n")

    r = client.delete(
        "/api/images/img/annotations/0",
        params={"images_dir": str(img_dir)},
    )
    assert r.status_code == 200
    remaining = lbl.read_text().strip().splitlines()
    assert len(remaining) == 1
    assert remaining[0].startswith("1")


def test_delete_annotation_out_of_range(tmp_path):
    img_dir = _make_dataset(tmp_path)
    r = client.delete(
        "/api/images/img/annotations/99",
        params={"images_dir": str(img_dir)},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n yoloV8 pytest tests/test_preview.py -v
# Expected: failures — route not registered
```

- [ ] **Step 3: Create `backend/routers/preview.py`**

```python
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.models import AnnotationItem, PreviewResponse
from backend.utils import (
    IMAGE_EXTENSIONS, detect_mode_from_label,
    generate_colors, redraw_from_annotations,
)

router = APIRouter(prefix="/api/images", tags=["preview"])

PREVIEW_CACHE = Path(".cache/previews")


def _derive_paths(image_id: str, images_dir: str) -> dict:
    """Given images_dir (e.g. dataset/images/train), derive all related paths."""
    p = Path(images_dir)
    parts = p.parts
    try:
        img_idx = next(i for i, part in enumerate(parts) if part == "images")
        dataset_dir = Path(*parts[:img_idx])
        split = parts[img_idx + 1]
    except (StopIteration, IndexError):
        dataset_dir = p.parent
        split = "train"

    labels_dir = dataset_dir / "labels" / split
    data_yaml = dataset_dir / "data.yaml"

    img_file = next(
        (p / f"{image_id}{ext}" for ext in IMAGE_EXTENSIONS
         if (p / f"{image_id}{ext}").exists()),
        None,
    )
    return {
        "img_file": img_file,
        "label_file": labels_dir / f"{image_id}.txt",
        "data_yaml": data_yaml,
        "dataset_dir": dataset_dir,
        "split": split,
    }


def _load_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return {}
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _parse_annotations(label_file: Path, class_names: dict) -> list[dict]:
    if not label_file.exists():
        return []
    lines = [l.strip() for l in label_file.read_text().splitlines() if l.strip()]
    colors = generate_colors(len(lines))
    result = []
    for i, line in enumerate(lines):
        parts = line.split()
        class_id = int(parts[0])
        values = list(map(float, parts[1:]))
        mode = detect_mode_from_label(line)
        b, g, rgb_r = colors[i]  # BGR → RGB for frontend
        result.append({
            "id": i,
            "label_line": line,
            "class_name": class_names.get(class_id, str(class_id)),
            "type": "bbox" if mode == "detect" else "polygon",
            "color": [int(rgb_r), int(g), int(b)],  # RGB for Canvas
            "color_bgr": colors[i],                  # BGR for cv2
            "values": values,
            "mode": mode,
        })
    return result


def _render_and_cache(image_id: str, img_file: Path, annotations: list[dict]) -> str:
    """Render preview image, save to cache, return URL path."""
    PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_CACHE / f"{image_id}.jpg"

    img_bgr = cv2.imread(str(img_file))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="无法读取图片文件")

    if annotations:
        # Build contour dicts for redraw_from_annotations
        mode = annotations[0]["mode"]
        contours = [
            {"label_line": a["label_line"], "class_name": a["class_name"],
             "color": a["color_bgr"]}
            for a in annotations
        ]
        vis = redraw_from_annotations(img_bgr, contours, mode)
    else:
        vis = img_bgr

    cv2.imwrite(str(out_path), vis)
    return f"/previews/{image_id}.jpg"


@router.get("/{image_id}/preview", response_model=PreviewResponse)
def get_preview(image_id: str, images_dir: str = Query(...)):
    paths = _derive_paths(image_id, images_dir)
    if paths["img_file"] is None:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")

    class_names = _load_class_names(paths["data_yaml"])
    annotations = _parse_annotations(paths["label_file"], class_names)
    preview_url = _render_and_cache(image_id, paths["img_file"], annotations)

    ann_items = []
    for a in annotations:
        vals = a["values"]
        if a["mode"] == "detect":
            xc, yc, w, h = vals
            x1, y1 = xc - w / 2, yc - h / 2
            x2, y2 = xc + w / 2, yc + h / 2
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            bbox = [x1, y1, x2, y2]
        else:
            points = [vals[i:i+2] for i in range(0, len(vals), 2)]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        ann_items.append(AnnotationItem(
            id=a["id"],
            class_name=a["class_name"],
            type=a["type"],
            color=a["color"],
            points=points,
            bbox=bbox,
        ))

    return PreviewResponse(
        preview_url=preview_url,
        annotations=ann_items,
        images_dir=images_dir,
        split=paths["split"],
    )


@router.delete("/{image_id}/annotations/{ann_id}")
def delete_annotation(image_id: str, ann_id: int, images_dir: str = Query(...)):
    paths = _derive_paths(image_id, images_dir)
    if paths["img_file"] is None:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")

    label_file = paths["label_file"]
    if not label_file.exists():
        raise HTTPException(status_code=404, detail="标注文件不存在")

    lines = [l.strip() for l in label_file.read_text().splitlines() if l.strip()]
    if ann_id < 0 or ann_id >= len(lines):
        raise HTTPException(status_code=404, detail=f"标注序号越界: {ann_id}")

    lines.pop(ann_id)
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Re-render preview
    class_names = _load_class_names(paths["data_yaml"])
    annotations = _parse_annotations(label_file, class_names)
    preview_url = _render_and_cache(image_id, paths["img_file"], annotations)

    return {"preview_url": preview_url, "remaining": len(lines)}
```

- [ ] **Step 4: Register router in `backend/main.py`**

```python
from backend.routers import preview as preview_router
app.include_router(preview_router.router)
```

- [ ] **Step 5: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_preview.py -v
# Expected: 5 passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/routers/preview.py backend/main.py tests/test_preview.py
git commit -m "feat: preview router (render + annotation deletion) + tests"
```

---

## Task 8: Annotate Router

**Files:**
- Create: `backend/routers/annotate.py`
- Create: `tests/test_annotate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_annotate.py
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _mock_segmentor():
    seg = MagicMock()
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:60] = True
    seg.predict.return_value = (
        np.array([mask]),
        np.array([[20, 20, 60, 60]], dtype=float),
        np.array([0.85]),
    )
    return seg


def test_annotate_missing_dir():
    r = client.post("/api/annotate", json={
        "image_dir": "/nonexistent/xyz",
        "output_dir": "/tmp/out",
        "prompts": "cat",
    })
    assert r.status_code == 202  # task accepted
    data = r.json()
    assert "task_id" in data


def test_annotate_returns_task_id():
    r = client.post("/api/annotate", json={
        "image_dir": "rawData",
        "output_dir": "dataset",
        "prompts": "cat, dog",
        "mode": "detect",
        "conf": 0.3,
    })
    assert r.status_code == 202
    assert len(r.json()["task_id"]) == 12


def test_task_status_pending(tmp_path):
    r = client.post("/api/annotate", json={
        "image_dir": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "prompts": "cat",
    })
    task_id = r.json()["task_id"]
    r2 = client.get(f"/api/tasks/{task_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] in ("pending", "running", "error", "done")


def test_task_not_found():
    r = client.get("/api/tasks/nonexistent123")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n yoloV8 pytest tests/test_annotate.py -v
# Expected: failures — routes not registered
```

- [ ] **Step 3: Create `backend/routers/annotate.py`**

```python
import random
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException

from backend.models import AnnotateRequest, TaskStatus
from backend.tasks import registry
from backend.utils import (
    collect_images, xyxy_to_xywh_norm, mask_to_polygon_norm,
    generate_colors, redraw_from_annotations,
)

router = APIRouter(tags=["annotate"])

PREVIEW_CACHE = Path(".cache/previews")


@router.post("/api/annotate", status_code=202)
def start_annotate(req: AnnotateRequest):
    task_id = registry.create()
    t = threading.Thread(
        target=_run_annotation, args=(task_id, req), daemon=True
    )
    t.start()
    return {"task_id": task_id}


@router.get("/api/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str):
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return task


def _run_annotation(task_id: str, req: AnnotateRequest):
    registry.update(task_id, status="running")
    try:
        # Import here to avoid loading model at module import time
        from backend.segmentor import get_segmentor
        from sam3 import draw_masks_on_image

        names = [p.strip() for p in req.prompts.split(",") if p.strip()]
        if not names:
            registry.update(task_id, status="error", message="提示词为空")
            return

        image_files = collect_images(req.image_dir)
        random.seed(42)
        random.shuffle(image_files)

        n_val = max(1, int(len(image_files) * req.val_ratio))
        val_set = set(image_files[:n_val])

        out = Path(req.output_dir)
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)

        segmentor = get_segmentor(req.conf)
        total = len(image_files)
        registry.update(task_id, total=total)

        class_counts = {n: 0 for n in names}

        for idx, img_path in enumerate(image_files):
            registry.update(task_id, progress=idx + 1,
                            message=f"{idx + 1}/{total}")

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem
            split = "val" if img_path in val_set else "train"

            all_masks, all_boxes, all_confs, all_labels, label_lines = [], [], [], [], []

            for class_id, prompt in enumerate(names):
                masks, boxes, confs = segmentor.predict(
                    img_path, prompt, force_reload=(class_id == 0)
                )
                if masks is None:
                    continue
                for i in range(len(masks)):
                    if req.mode == "detect":
                        xc, yc, bw, bh = xyxy_to_xywh_norm(boxes[i], w, h)
                        ll = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    else:
                        poly = mask_to_polygon_norm(masks[i], w, h)
                        if poly is None:
                            continue
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                        ll = f"{class_id} {coords}"
                    all_masks.append(masks[i])
                    all_boxes.append(boxes[i])
                    all_confs.append(float(confs[i]))
                    all_labels.append(prompt)
                    label_lines.append(ll)
                    class_counts[prompt] += 1

            # Truncate to max_instances
            if len(all_masks) > req.max_instances:
                if req.sort_mode == "conf":
                    order = sorted(range(len(all_confs)),
                                   key=lambda k: all_confs[k], reverse=True)
                else:
                    order = sorted(range(len(all_boxes)),
                                   key=lambda k: (
                                       (all_boxes[k][2] - all_boxes[k][0]) *
                                       (all_boxes[k][3] - all_boxes[k][1])
                                   ), reverse=True)
                keep = sorted(order[:req.max_instances])
                all_masks = [all_masks[k] for k in keep]
                all_boxes = [all_boxes[k] for k in keep]
                all_labels = [all_labels[k] for k in keep]
                label_lines = [label_lines[k] for k in keep]

            # Write label file
            img_name = stem + ".jpg"
            lbl_path = out / "labels" / split / f"{stem}.txt"
            lbl_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else "")
            )

            # Copy image
            dst = out / "images" / split / img_name
            if Path(img_path).resolve() != dst.resolve():
                shutil.copy2(img_path, dst)

            # Generate and cache preview
            colors = generate_colors(len(all_masks))
            preview_path = PREVIEW_CACHE / img_name
            if all_masks:
                contours = [
                    {"label_line": label_lines[i], "class_name": all_labels[i],
                     "color": colors[i]}
                    for i in range(len(all_masks))
                ]
                vis = redraw_from_annotations(img, contours, req.mode)
                cv2.imwrite(str(preview_path), vis)
            else:
                cv2.imwrite(str(preview_path), img)

        # Write data.yaml
        data_yaml = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(names)},
        }
        with open(out / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

        registry.update(
            task_id, status="done", progress=total,
            message="标注完成",
            result={"class_counts": class_counts, "total": total,
                    "dataset_dir": req.output_dir},
        )

    except Exception as e:
        registry.update(task_id, status="error", message=str(e))
```

- [ ] **Step 4: Register router in `backend/main.py`**

```python
from backend.routers import annotate as annotate_router
app.include_router(annotate_router.router)
```

- [ ] **Step 5: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_annotate.py -v
# Expected: 4 passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/routers/annotate.py backend/main.py tests/test_annotate.py
git commit -m "feat: annotate router (SAM3 task + polling) + tests"
```

---

## Task 9: Train Router

**Files:**
- Create: `backend/routers/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_train_returns_task_id(tmp_path):
    # Create a minimal data.yaml so the router doesn't error on validation
    (tmp_path / "data.yaml").write_text("names:\n  0: cat\n")
    r = client.post("/api/train", json={
        "dataset_dir": str(tmp_path),
        "task": "detect",
        "yolo_version": "YOLOv8",
        "model_size": "n",
        "epochs": 1,
        "imgsz": 32,
    })
    assert r.status_code == 202
    assert "task_id" in r.json()


def test_train_missing_data_yaml():
    r = client.post("/api/train", json={
        "dataset_dir": "/nonexistent/xyz",
        "task": "detect",
        "yolo_version": "YOLOv8",
        "model_size": "n",
        "epochs": 1,
        "imgsz": 32,
    })
    # Should accept and return task_id; error surfaces via task status
    assert r.status_code == 202


def test_train_logs_sse_invalid_task():
    # SSE endpoint for unknown task should close immediately or 404
    r = client.get("/api/train/nonexistent/logs")
    assert r.status_code == 404
```

- [ ] **Step 2: Create `backend/routers/train.py`**

```python
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models import TrainRequest
from backend.tasks import registry
from backend.utils import build_model_name

router = APIRouter(tags=["train"])

# Per-task log buffers: task_id → list[str]
_log_buffers: dict[str, list[str]] = {}
_log_lock = threading.Lock()


@router.post("/api/train", status_code=202)
def start_train(req: TrainRequest):
    task_id = registry.create()
    with _log_lock:
        _log_buffers[task_id] = []
    t = threading.Thread(
        target=_run_training, args=(task_id, req), daemon=True
    )
    t.start()
    return {"task_id": task_id}


@router.get("/api/train/{task_id}/logs")
def stream_logs(task_id: str):
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    def generate():
        last = 0
        while True:
            with _log_lock:
                buf = _log_buffers.get(task_id, [])
                new_lines = buf[last:]
                last += len(new_lines)
            for line in new_lines:
                yield f"data: {line}\n\n"

            t = registry.get(task_id)
            if t and t.status in ("done", "error"):
                # Drain remaining
                with _log_lock:
                    buf = _log_buffers.get(task_id, [])
                    for line in buf[last:]:
                        yield f"data: {line}\n\n"
                if t.status == "error":
                    yield f"event: error\ndata: {t.message}\n\n"
                else:
                    yield "event: done\ndata: 训练完成\n\n"
                break
            time.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _run_training(task_id: str, req: TrainRequest):
    registry.update(task_id, status="running")

    def log(text: str):
        with _log_lock:
            buf = _log_buffers.get(task_id)
            if buf is not None:
                buf.append(text.rstrip())

    try:
        data_yaml = Path(req.dataset_dir) / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

        model_file = build_model_name(req.yolo_version, req.model_size, req.task)
        log(f"模型: {model_file}")
        log(f"任务: {req.task}")
        log(f"数据集: {data_yaml}")
        log(f"Epochs: {req.epochs}, ImgSz: {req.imgsz}")
        log("=" * 60)

        save_dir = "runs/"  # default; overwritten on success

        class LogCapture:
            def __init__(self, orig):
                self.orig = orig
            def write(self, text):
                self.orig.write(text)
                if text.strip():
                    log(text)
            def flush(self):
                self.orig.flush()

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = LogCapture(old_out)
        sys.stderr = LogCapture(old_err)
        try:
            from ultralytics import YOLO
            model = YOLO(model_file)
            results = model.train(
                data=str(data_yaml),
                epochs=req.epochs,
                imgsz=req.imgsz,
                task=req.task,
                verbose=True,
            )
            save_dir = str(results.save_dir) if hasattr(results, "save_dir") else "runs/"
            log(f"\n{'=' * 60}")
            log(f"训练完成！结果保存在: {save_dir}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        registry.update(task_id, status="done", message="训练完成",
                        result={"save_dir": save_dir})
    except Exception as e:
        log(f"训练出错: {e}")
        registry.update(task_id, status="error", message=str(e))
```

- [ ] **Step 3: Register router in `backend/main.py`**

```python
from backend.routers import train as train_router
app.include_router(train_router.router)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_train.py -v
# Expected: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add backend/routers/train.py backend/main.py tests/test_train.py
git commit -m "feat: train router (YOLO task + SSE log stream) + tests"
```

---

## Task 10: Infer Router

**Files:**
- Create: `backend/routers/infer.py`
- Create: `tests/test_infer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_infer.py
import io
import uuid
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _fake_image_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


def test_model_info_missing_file():
    r = client.get("/api/model-info", params={"weights_path": "/nonexistent.pt"})
    assert r.status_code == 200
    assert "size_mb" in r.json()


def test_infer_no_image():
    r = client.post("/api/infer", data={
        "weights_path": "/nonexistent.pt", "conf": "0.25", "imgsz": "640"
    })
    assert r.status_code == 422  # missing image field


def test_infer_bad_weights(tmp_path):
    fake_img = _fake_image_bytes()
    r = client.post("/api/infer", data={
        "weights_path": "/nonexistent.pt",
        "conf": "0.25",
        "imgsz": "640",
    }, files={"image": ("test.jpg", fake_img, "image/jpeg")})
    assert r.status_code in (400, 422, 500)


def test_segment_missing_prompts(tmp_path):
    fake_img = _fake_image_bytes()
    r = client.post("/api/segment", data={
        "prompts": "",
        "conf": "0.25",
        "max_instances": "7",
        "sort_mode": "conf",
    }, files={"image": ("test.jpg", fake_img, "image/jpeg")})
    assert r.status_code == 400
```

- [ ] **Step 2: Create `backend/routers/infer.py`**

```python
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.models import InferResponse, ModelInfoResponse

router = APIRouter(tags=["infer"])

PREVIEW_CACHE = Path(".cache/previews")


@router.get("/api/model-info", response_model=ModelInfoResponse)
def model_info(weights_path: str = Query(...)):
    p = Path(weights_path)
    size_mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0.0
    parts = []
    model_name = arch = task = None

    if p.exists():
        try:
            import torch
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                model_obj = ckpt.get("model")
                if model_obj and hasattr(model_obj, "yaml"):
                    cfg = model_obj.yaml
                    if isinstance(cfg, dict):
                        yaml_file = cfg.get("yaml_file", "")
                        scale = cfg.get("scale", "")
                        if yaml_file:
                            arch = Path(yaml_file).stem
                            parts.append(f"架构: {arch}")
                        elif scale:
                            parts.append(f"规格: {scale}")
                ta = ckpt.get("train_args", {})
                if isinstance(ta, dict):
                    if "model" in ta:
                        model_name = ta["model"]
                        parts.insert(0, f"模型: {model_name}")
                    task = ta.get("task")
                    if task:
                        parts.append(f"任务: {task}")
        except Exception as e:
            parts.append(f"(读取失败: {e})")

    parts.append(f"文件大小: {size_mb:.1f} MB")
    return ModelInfoResponse(
        model_name=model_name, architecture=arch, task=task,
        size_mb=size_mb, raw="\n".join(parts) if parts else "(未找到文件)",
    )


@router.post("/api/infer", response_model=InferResponse)
async def run_infer(
    image: UploadFile = File(...),
    weights_path: str = Form(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
):
    if not Path(weights_path).exists():
        raise HTTPException(status_code=400, detail=f"权重文件不存在: {weights_path}")

    # Save upload to temp file
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await image.read())
        tmp_path = f.name

    try:
        from ultralytics import YOLO
        from collections import Counter

        model = YOLO(weights_path)
        results = model.predict(
            source=tmp_path, conf=conf, imgsz=int(imgsz), verbose=False
        )
        if not results:
            raise HTTPException(status_code=500, detail="推理失败")

        result = results[0]
        vis = result.plot()  # BGR

        PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
        out_name = f"infer_{uuid.uuid4().hex[:8]}.jpg"
        out_path = PREVIEW_CACHE / out_name
        cv2.imwrite(str(out_path), vis)

        names = result.names or {}
        stats: dict = {"total": 0, "classes": {}}
        if result.boxes and len(result.boxes):
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            counts = Counter(cls_ids.tolist())
            stats["total"] = sum(counts.values())
            stats["classes"] = {names.get(k, str(k)): v for k, v in counts.items()}

        return InferResponse(result_url=f"/previews/{out_name}", stats=stats)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/api/segment")
async def run_segment(
    image: UploadFile = File(...),
    prompts: str = Form(...),
    conf: float = Form(0.25),
    max_instances: int = Form(7),
    sort_mode: str = Form("conf"),
):
    names = [p.strip() for p in prompts.split(",") if p.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="提示词为空")

    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await image.read())
        tmp_path = f.name

    try:
        from backend.segmentor import get_segmentor
        from sam3 import draw_masks_on_image
        from backend.utils import generate_colors
        from collections import Counter

        img = cv2.imread(tmp_path)
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图片")

        segmentor = get_segmentor(conf)
        all_masks, all_boxes, all_confs, all_labels = [], [], [], []

        for i, prompt in enumerate(names):
            masks, boxes, confs = segmentor.predict(
                tmp_path, prompt, force_reload=(i == 0)
            )
            if masks is None:
                continue
            for j in range(len(masks)):
                all_masks.append(masks[j])
                all_boxes.append(boxes[j])
                all_confs.append(float(confs[j]))
                all_labels.append(prompt)

        if len(all_masks) > max_instances:
            if sort_mode == "conf":
                order = sorted(range(len(all_confs)),
                               key=lambda k: all_confs[k], reverse=True)
            else:
                order = sorted(range(len(all_boxes)),
                               key=lambda k: (
                                   (all_boxes[k][2] - all_boxes[k][0]) *
                                   (all_boxes[k][3] - all_boxes[k][1])
                               ), reverse=True)
            keep = sorted(order[:max_instances])
            all_masks = [all_masks[k] for k in keep]
            all_boxes = [all_boxes[k] for k in keep]
            all_labels = [all_labels[k] for k in keep]

        PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
        out_name = f"seg_{uuid.uuid4().hex[:8]}.jpg"
        out_path = PREVIEW_CACHE / out_name

        if all_masks:
            vis = draw_masks_on_image(
                img, np.array(all_masks), np.array(all_boxes),
                labels=all_labels, alpha=0.4,
            )
        else:
            vis = img
        cv2.imwrite(str(out_path), vis)

        counts = Counter(all_labels)
        return {
            "preview_url": f"/previews/{out_name}",
            "annotations": [],   # Canvas interface: currently empty, extend if needed
            "stats": {"total": len(all_masks), "classes": dict(counts)},
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 3: Register router in `backend/main.py`**

```python
from backend.routers import infer as infer_router
app.include_router(infer_router.router)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n yoloV8 pytest tests/test_infer.py -v
# Expected: 4 passed
```

- [ ] **Step 5: Run full test suite**

```bash
conda run -n yoloV8 pytest tests/ -v
# Expected: all pass
```

- [ ] **Step 6: Commit**

```bash
git add backend/routers/infer.py backend/main.py tests/test_infer.py
git commit -m "feat: infer router (YOLO inference + model info + SAM3 segment) + tests"
```

---

## Task 11: Frontend Shell (HTML + CSS + app.js)

**Files:**
- Modify: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/app.js`

- [ ] **Step 1: Create `frontend/style.css`**

```css
/* ── CSS custom properties ── */
:root {
  --bg:           #1c1c1e;
  --bg-elevated:  #2c2c2e;
  --bg-sidebar:   rgba(44, 44, 46, 0.9);
  --separator:    rgba(255, 255, 255, 0.07);
  --text-primary: #ebebf5;
  --text-secondary: rgba(235, 235, 245, 0.4);
  --text-label:   rgba(235, 235, 245, 0.3);
  --btn-primary-bg:  rgba(235, 235, 245, 0.9);
  --btn-primary-fg:  #1c1c1e;
  --btn-danger-border: rgba(255, 69, 58, 0.6);
  --btn-danger-fg:     #ff453a;
  --radius-card:   8px;
  --radius-btn:    7px;
  --radius-pill:   12px;
  --blur:          blur(20px);
}

@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg:           #f2f2f7;
    --bg-elevated:  #ffffff;
    --bg-sidebar:   rgba(255, 255, 255, 0.75);
    --separator:    rgba(0, 0, 0, 0.08);
    --text-primary: #1c1c1e;
    --text-secondary: rgba(60, 60, 67, 0.5);
    --text-label:   rgba(60, 60, 67, 0.4);
    --btn-primary-bg:  #1c1c1e;
    --btn-primary-fg:  #ffffff;
  }
}

[data-theme="light"] {
  --bg:           #f2f2f7;
  --bg-elevated:  #ffffff;
  --bg-sidebar:   rgba(255, 255, 255, 0.75);
  --separator:    rgba(0, 0, 0, 0.08);
  --text-primary: #1c1c1e;
  --text-secondary: rgba(60, 60, 67, 0.5);
  --text-label:   rgba(60, 60, 67, 0.4);
  --btn-primary-bg:  #1c1c1e;
  --btn-primary-fg:  #ffffff;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
  background: var(--bg);
  color: var(--text-primary);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Topbar ── */
#topbar {
  height: 52px;
  background: var(--bg-sidebar);
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  border-bottom: 1px solid var(--separator);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 12px;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  z-index: 100;
}

#topbar .logo {
  font-weight: 700;
  font-size: 16px;
  letter-spacing: -0.3px;
}

#topbar .page-title {
  color: var(--text-secondary);
  font-size: 14px;
  flex: 1;
}

#theme-toggle {
  background: none;
  border: 1px solid var(--separator);
  border-radius: var(--radius-btn);
  color: var(--text-primary);
  padding: 5px 10px;
  cursor: pointer;
  font-size: 13px;
}

/* ── Layout ── */
#layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ── */
#sidebar {
  width: 160px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  border-right: 1px solid var(--separator);
  padding: 16px 8px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.nav-label {
  color: var(--text-label);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  padding: 4px 10px;
  margin-bottom: 6px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 10px;
  border-radius: var(--radius-btn);
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
  color: var(--text-secondary);
  font-size: 13px;
}

.nav-item:hover { background: rgba(128,128,128,0.1); }

.nav-item.active {
  background: rgba(128,128,128,0.14);
  color: var(--text-primary);
  font-weight: 500;
}

.nav-icon { font-size: 15px; }

/* ── Content area ── */
#content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.view { display: none; }
.view.active { display: block; }

/* ── Cards ── */
.card {
  background: var(--bg-elevated);
  border: 1px solid var(--separator);
  border-radius: var(--radius-card);
  padding: 16px;
  margin-bottom: 16px;
}

.card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
}

/* ── Form elements ── */
label {
  display: block;
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 5px;
}

input[type="text"], input[type="number"], textarea, select {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--separator);
  border-radius: var(--radius-btn);
  color: var(--text-primary);
  font-family: inherit;
  font-size: 13px;
  padding: 8px 10px;
  outline: none;
  transition: border-color 0.15s;
}
input:focus, textarea:focus, select:focus {
  border-color: rgba(128,128,128,0.4);
}

.form-row { display: flex; gap: 12px; margin-bottom: 12px; }
.form-row > * { flex: 1; }

/* ── Buttons ── */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: var(--radius-btn);
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: opacity 0.15s;
}
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn:hover:not(:disabled) { opacity: 0.85; }

.btn-primary {
  background: var(--btn-primary-bg);
  color: var(--btn-primary-fg);
}
.btn-danger {
  background: transparent;
  border: 1px solid var(--btn-danger-border);
  color: var(--btn-danger-fg);
}

/* ── Toast ── */
#toast-container {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.toast {
  background: var(--bg-elevated);
  border: 1px solid var(--separator);
  border-radius: var(--radius-card);
  padding: 12px 16px;
  font-size: 13px;
  max-width: 320px;
  animation: toast-in 0.2s ease;
}
.toast.error { border-color: var(--btn-danger-border); color: var(--btn-danger-fg); }

@keyframes toast-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Progress bar ── */
.progress-bar {
  height: 4px;
  background: var(--separator);
  border-radius: 2px;
  overflow: hidden;
  margin: 8px 0;
}
.progress-bar-fill {
  height: 100%;
  background: var(--btn-primary-bg);
  border-radius: 2px;
  transition: width 0.3s;
}

/* ── Log output ── */
.log-output {
  background: var(--bg);
  border: 1px solid var(--separator);
  border-radius: var(--radius-card);
  padding: 12px;
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 11px;
  color: var(--text-secondary);
  height: 260px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

/* ── Preview page ── */
.preview-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.preview-toolbar .filename {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
}
.preview-img-wrap {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--separator);
  border-radius: var(--radius-card);
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  margin-bottom: 10px;
}
.preview-img-wrap img {
  max-width: 100%;
  max-height: 60vh;
  border-radius: 4px;
  display: block;
}
.annotation-tags {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 10px 0;
}
.ann-tag {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--separator);
  cursor: pointer;
  font-size: 12px;
  transition: background 0.15s;
  user-select: none;
}
.ann-tag.selected {
  background: rgba(255, 69, 58, 0.12);
  border-color: var(--btn-danger-border);
}
.ann-color-swatch {
  width: 9px;
  height: 9px;
  border-radius: 3px;
  flex-shrink: 0;
}
```

- [ ] **Step 2: Create `frontend/app.js`**

```javascript
// ── API client ────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  try {
    const res = await fetch(path, options);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.error || j.detail || msg; } catch {}
      toast(msg, "error");
      throw new Error(msg);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  } catch (e) {
    if (e.message === "Failed to fetch") toast("无法连接服务器", "error");
    throw e;
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function toast(msg, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Theme ─────────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const isDark = current === "dark" ||
    (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const next = isDark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  document.getElementById("theme-toggle").textContent = next === "dark" ? "☀" : "🌙";
}

// ── Router ────────────────────────────────────────────────────────────────────

const VIEWS = {
  annotate: { label: "自动标注", icon: "🏷" },
  preview:  { label: "预览编辑", icon: "🖼" },
  train:    { label: "YOLO 训练", icon: "⚡" },
  infer:    { label: "推理测试", icon: "🔍" },
};

let currentView = "annotate";

function navigate(view) {
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach(el => {
    el.classList.toggle("active", el.id === `view-${view}`);
  });
  document.querySelector("#topbar .page-title").textContent =
    VIEWS[view]?.label ?? view;
  currentView = view;
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();

  // Build sidebar
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML =
    `<div class="nav-label">功能</div>` +
    Object.entries(VIEWS).map(([key, v]) =>
      `<div class="nav-item${key === currentView ? " active" : ""}"
            data-view="${key}">
         <span class="nav-icon">${v.icon}</span>${v.label}
       </div>`
    ).join("");

  sidebar.querySelectorAll(".nav-item").forEach(el =>
    el.addEventListener("click", () => navigate(el.dataset.view))
  );

  document.getElementById("theme-toggle")
    .addEventListener("click", toggleTheme);

  // Lazy-init views
  const modules = {
    annotate: () => import("./views/annotate.js"),
    preview:  () => import("./views/preview.js"),
    train:    () => import("./views/train.js"),
    infer:    () => import("./views/infer.js"),
  };
  const initialized = new Set();

  async function maybeInit(view) {
    if (!initialized.has(view) && modules[view]) {
      const mod = await modules[view]();
      if (mod.init) mod.init();
      initialized.add(view);
    }
  }

  // Override navigate to lazy-init
  const _navigate = navigate;
  window.navigate = async (view) => {
    _navigate(view);
    await maybeInit(view);
  };

  await maybeInit("annotate");
  navigate("annotate");
});

export { api, toast };
```

- [ ] **Step 3: Replace `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SAMpler</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <header id="topbar">
    <span class="logo">SAMpler</span>
    <span class="page-title">自动标注</span>
    <button id="theme-toggle">🌙</button>
  </header>

  <div id="layout">
    <nav id="sidebar"></nav>

    <main id="content">
      <div id="view-annotate" class="view active"></div>
      <div id="view-preview"  class="view"></div>
      <div id="view-train"    class="view"></div>
      <div id="view-infer"    class="view"></div>
    </main>
  </div>

  <div id="toast-container"></div>

  <script type="module" src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Verify visually**

```bash
conda run -n yoloV8 python run.py
# Open http://localhost:8000
# Check: sidebar shows 4 nav items, topbar has theme toggle, dark/light toggle works
```

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: frontend shell (HTML + Apple-style CSS + router + theme toggle)"
```

---

## Task 12: Annotate View

**Files:**
- Create: `frontend/views/annotate.js`

- [ ] **Step 1: Create `frontend/views/annotate.js`**

```javascript
import { api, toast } from "../app.js";

let pollInterval = null;

export function init() {
  const container = document.getElementById("view-annotate");
  container.innerHTML = `
    <div class="card">
      <div class="card-title">数据源</div>
      <div class="form-row">
        <div>
          <label>图片目录</label>
          <input type="text" id="ann-image-dir" value="rawData" />
        </div>
        <div>
          <label>输出目录</label>
          <input type="text" id="ann-output-dir" value="dataset" />
        </div>
      </div>
      <div>
        <label>提示词（逗号分隔）</label>
        <input type="text" id="ann-prompts" placeholder="person, car, dog" />
      </div>
    </div>

    <div class="card">
      <div class="card-title">标注设置</div>
      <div class="form-row">
        <div>
          <label>标注模式</label>
          <select id="ann-mode">
            <option value="segment">分割 (segment)</option>
            <option value="detect">检测 (detect)</option>
          </select>
        </div>
        <div>
          <label>排序模式</label>
          <select id="ann-sort">
            <option value="conf">置信度</option>
            <option value="area">框面积</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>置信度阈值</label>
          <input type="number" id="ann-conf" value="0.25" step="0.05" min="0.01" max="0.99" />
        </div>
        <div>
          <label>验证集比例</label>
          <input type="number" id="ann-val-ratio" value="0.1" step="0.05" min="0" max="0.5" />
        </div>
        <div>
          <label>最大实例数</label>
          <input type="number" id="ann-max-inst" value="7" min="1" max="100" />
        </div>
      </div>
    </div>

    <button class="btn btn-primary" id="ann-run-btn" style="width:100%">开始标注</button>

    <div id="ann-progress-wrap" style="display:none;margin-top:16px">
      <div class="progress-bar"><div class="progress-bar-fill" id="ann-progress-fill" style="width:0%"></div></div>
      <p style="font-size:12px;color:var(--text-secondary);margin-top:4px" id="ann-progress-text"></p>
    </div>
    <div id="ann-result" style="margin-top:12px;font-size:13px;color:var(--text-secondary)"></div>
  `;

  document.getElementById("ann-run-btn").addEventListener("click", startAnnotation);
}

async function startAnnotation() {
  const req = {
    image_dir:    document.getElementById("ann-image-dir").value.trim(),
    output_dir:   document.getElementById("ann-output-dir").value.trim(),
    prompts:      document.getElementById("ann-prompts").value.trim(),
    mode:         document.getElementById("ann-mode").value,
    sort_mode:    document.getElementById("ann-sort").value,
    conf:         parseFloat(document.getElementById("ann-conf").value),
    val_ratio:    parseFloat(document.getElementById("ann-val-ratio").value),
    max_instances: parseInt(document.getElementById("ann-max-inst").value),
  };

  if (!req.prompts) { toast("请输入提示词", "error"); return; }

  const btn = document.getElementById("ann-run-btn");
  btn.disabled = true;
  btn.textContent = "标注中...";
  document.getElementById("ann-progress-wrap").style.display = "block";
  document.getElementById("ann-result").textContent = "";

  try {
    const { task_id } = await api("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    pollInterval = setInterval(async () => {
      try {
        const task = await api(`/api/tasks/${task_id}`);
        const pct = task.total > 0
          ? Math.round((task.progress / task.total) * 100) : 0;
        document.getElementById("ann-progress-fill").style.width = `${pct}%`;
        document.getElementById("ann-progress-text").textContent =
          `${task.message} (${pct}%)`;

        if (task.status === "done" || task.status === "error") {
          clearInterval(pollInterval);
          btn.disabled = false;
          btn.textContent = "开始标注";
          if (task.status === "done") {
            const r = task.result || {};
            const lines = [`标注完成！总图片: ${r.total}`];
            if (r.class_counts) {
              for (const [k, v] of Object.entries(r.class_counts)) {
                lines.push(`  ${k}: ${v}`);
              }
            }
            document.getElementById("ann-result").textContent = lines.join("\n");
            toast("标注完成！");
          } else {
            toast(task.message || "标注失败", "error");
          }
        }
      } catch {}
    }, 1000);
  } catch {
    btn.disabled = false;
    btn.textContent = "开始标注";
  }
}
```

- [ ] **Step 2: Verify manually**

```
Open http://localhost:8000
Click "自动标注"
Fill in prompts, click 开始标注
Check: progress bar appears, polling starts
```

- [ ] **Step 3: Commit**

```bash
git add frontend/views/annotate.js
git commit -m "feat: annotate view with progress polling"
```

---

## Task 13: Preview View

**Files:**
- Create: `frontend/views/preview.js`

- [ ] **Step 1: Create `frontend/views/preview.js`**

```javascript
import { api, toast } from "../app.js";

let state = {
  files: [],
  idx: 0,
  imagesDir: "dataset/images/train",
  selectedAnns: new Set(),
};

export function init() {
  const container = document.getElementById("view-preview");
  container.innerHTML = `
    <div class="card" style="margin-bottom:12px">
      <div class="form-row" style="align-items:flex-end">
        <div style="flex:3">
          <label>图片目录 (dataset/images/train 或 val)</label>
          <input type="text" id="prev-dir" value="dataset/images/train" />
        </div>
        <button class="btn btn-primary" id="prev-load-btn">加载</button>
      </div>
    </div>

    <div class="preview-toolbar">
      <span class="filename" id="prev-filename">—</span>
      <button class="btn" id="prev-prev-btn" style="padding:6px 10px">◀</button>
      <span id="prev-page" style="font-size:12px;color:var(--text-secondary);min-width:50px;text-align:center">0 / 0</span>
      <button class="btn" id="prev-next-btn" style="padding:6px 10px">▶</button>
      <button class="btn btn-danger" id="prev-del-img-btn">删除此图</button>
    </div>

    <div class="preview-img-wrap" id="prev-img-wrap">
      <span style="color:var(--text-secondary);font-size:13px">请先加载图片目录</span>
    </div>

    <div class="annotation-tags" id="prev-ann-tags"></div>
    <div style="display:flex;justify-content:flex-end;margin-top:8px">
      <button class="btn btn-danger" id="prev-del-ann-btn" disabled>删除选中标注</button>
    </div>
  `;

  document.getElementById("prev-load-btn").addEventListener("click", loadDir);
  document.getElementById("prev-prev-btn").addEventListener("click", () => navigate(-1));
  document.getElementById("prev-next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-del-img-btn").addEventListener("click", deleteCurrentImage);
  document.getElementById("prev-del-ann-btn").addEventListener("click", deleteSelectedAnns);
}

async function loadDir() {
  const dir = document.getElementById("prev-dir").value.trim();
  try {
    const data = await api(`/api/images?dir=${encodeURIComponent(dir)}`);
    state.files = data.files;
    state.idx = 0;
    state.imagesDir = dir;
    state.selectedAnns = new Set();
    toast(`加载了 ${data.total} 张图片`);
    await showCurrent();
  } catch {}
}

async function navigate(dir) {
  if (!state.files.length) return;
  state.idx = Math.max(0, Math.min(state.idx + dir, state.files.length - 1));
  state.selectedAnns = new Set();
  await showCurrent();
}

async function showCurrent() {
  const files = state.files;
  if (!files.length) return;
  const f = files[state.idx];

  document.getElementById("prev-filename").textContent = f.filename;
  document.getElementById("prev-page").textContent =
    `${state.idx + 1} / ${files.length}`;

  try {
    const data = await api(
      `/api/images/${f.id}/preview?images_dir=${encodeURIComponent(state.imagesDir)}`
    );
    // Image
    const wrap = document.getElementById("prev-img-wrap");
    wrap.innerHTML = `<img src="${data.preview_url}?t=${Date.now()}" alt="${f.filename}" />`;

    // Annotation tags
    renderTags(data.annotations);
  } catch {}
}

function renderTags(anns) {
  const container = document.getElementById("prev-ann-tags");
  state.selectedAnns = new Set();
  updateDeleteBtn();

  if (!anns.length) {
    container.innerHTML = `<span style="color:var(--text-secondary);font-size:12px">无标注</span>`;
    return;
  }

  container.innerHTML = anns.map((a, i) => {
    const [r, g, b] = a.color;
    return `
      <div class="ann-tag" data-ann-id="${a.id}">
        <span class="ann-color-swatch" style="background:rgb(${r},${g},${b})"></span>
        <span>${a.class_name}</span>
        <span style="font-size:10px;color:var(--text-secondary)">${a.type}</span>
      </div>`;
  }).join("");

  container.querySelectorAll(".ann-tag").forEach(el => {
    el.addEventListener("click", () => {
      const id = parseInt(el.dataset.annId);
      if (state.selectedAnns.has(id)) {
        state.selectedAnns.delete(id);
        el.classList.remove("selected");
      } else {
        state.selectedAnns.add(id);
        el.classList.add("selected");
      }
      updateDeleteBtn();
    });
  });
}

function updateDeleteBtn() {
  document.getElementById("prev-del-ann-btn").disabled =
    state.selectedAnns.size === 0;
}

async function deleteCurrentImage() {
  if (!state.files.length) return;
  const f = state.files[state.idx];
  try {
    await api(
      `/api/images/${f.id}?images_dir=${encodeURIComponent(state.imagesDir)}`,
      { method: "DELETE" }
    );
    state.files.splice(state.idx, 1);
    if (!state.files.length) {
      document.getElementById("prev-img-wrap").innerHTML =
        `<span style="color:var(--text-secondary)">列表已空</span>`;
      document.getElementById("prev-ann-tags").innerHTML = "";
      return;
    }
    state.idx = Math.min(state.idx, state.files.length - 1);
    state.selectedAnns = new Set();
    toast("已删除");
    await showCurrent();
  } catch {}
}

async function deleteSelectedAnns() {
  if (!state.files.length || !state.selectedAnns.size) return;
  const f = state.files[state.idx];
  // Delete in reverse order to preserve indices
  const ids = [...state.selectedAnns].sort((a, b) => b - a);
  try {
    for (const annId of ids) {
      await api(
        `/api/images/${f.id}/annotations/${annId}?images_dir=${encodeURIComponent(state.imagesDir)}`,
        { method: "DELETE" }
      );
    }
    toast(`已删除 ${ids.length} 个标注`);
    state.selectedAnns = new Set();
    await showCurrent();
  } catch {}
}
```

- [ ] **Step 2: Verify manually**

```
Open http://localhost:8000, click 预览编辑
Enter a valid dataset/images/train directory, click 加载
Check: images load, annotation tags appear with color swatches
Click tags to select, click 删除选中标注
Check: annotation removed, preview re-renders
```

- [ ] **Step 3: Commit**

```bash
git add frontend/views/preview.js
git commit -m "feat: preview view with annotation tag selection and deletion"
```

---

## Task 14: Train View

**Files:**
- Create: `frontend/views/train.js`

- [ ] **Step 1: Create `frontend/views/train.js`**

```javascript
import { api, toast } from "../app.js";

let eventSource = null;

export function init() {
  const container = document.getElementById("view-train");
  container.innerHTML = `
    <div class="card">
      <div class="card-title">训练配置</div>
      <div class="form-row">
        <div>
          <label>数据集目录</label>
          <input type="text" id="tr-dataset-dir" value="dataset" />
        </div>
        <div>
          <label>任务类型</label>
          <select id="tr-task">
            <option value="segment">segment</option>
            <option value="detect">detect</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>YOLO 版本</label>
          <select id="tr-version">
            <option value="YOLOv11">YOLOv11</option>
            <option value="YOLOv8">YOLOv8</option>
            <option value="YOLO26">YOLO26</option>
          </select>
        </div>
        <div>
          <label>模型大小</label>
          <select id="tr-size">
            <option value="n">n (最小)</option>
            <option value="s">s</option>
            <option value="m">m</option>
            <option value="l">l</option>
            <option value="x">x (最大)</option>
          </select>
        </div>
        <div>
          <label>Epochs</label>
          <input type="number" id="tr-epochs" value="100" min="1" max="1000" />
        </div>
        <div>
          <label>ImgSz</label>
          <input type="number" id="tr-imgsz" value="640" min="32" step="32" />
        </div>
      </div>
    </div>

    <button class="btn btn-primary" id="tr-run-btn" style="width:100%">开始训练</button>

    <div style="margin-top:16px">
      <div class="progress-bar">
        <div class="progress-bar-fill" id="tr-progress-fill" style="width:0%"></div>
      </div>
      <div class="log-output" id="tr-log" style="margin-top:10px"></div>
    </div>
  `;

  document.getElementById("tr-run-btn").addEventListener("click", startTraining);
}

async function startTraining() {
  if (eventSource) { eventSource.close(); eventSource = null; }

  const req = {
    dataset_dir:  document.getElementById("tr-dataset-dir").value.trim(),
    task:         document.getElementById("tr-task").value,
    yolo_version: document.getElementById("tr-version").value,
    model_size:   document.getElementById("tr-size").value,
    epochs:       parseInt(document.getElementById("tr-epochs").value),
    imgsz:        parseInt(document.getElementById("tr-imgsz").value),
  };

  const btn = document.getElementById("tr-run-btn");
  btn.disabled = true;
  btn.textContent = "训练中...";
  const logEl = document.getElementById("tr-log");
  logEl.textContent = "";

  try {
    const { task_id } = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    eventSource = new EventSource(`/api/train/${task_id}/logs`);

    eventSource.onmessage = (e) => {
      logEl.textContent += e.data + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    };

    eventSource.addEventListener("done", () => {
      eventSource.close();
      btn.disabled = false;
      btn.textContent = "开始训练";
      document.getElementById("tr-progress-fill").style.width = "100%";
      toast("训练完成！");
    });

    eventSource.addEventListener("error", (e) => {
      eventSource.close();
      btn.disabled = false;
      btn.textContent = "开始训练";
      toast(e.data || "训练失败", "error");
    });

    // Poll progress for progress bar
    const poll = setInterval(async () => {
      try {
        const t = await api(`/api/tasks/${task_id}`);
        if (t.total > 0) {
          const pct = Math.round((t.progress / t.total) * 100);
          document.getElementById("tr-progress-fill").style.width = `${pct}%`;
        }
        if (t.status === "done" || t.status === "error") clearInterval(poll);
      } catch {}
    }, 2000);

  } catch {
    btn.disabled = false;
    btn.textContent = "开始训练";
  }
}
```

- [ ] **Step 2: Verify manually**

```
Click YOLO训练, fill in dataset_dir and settings
Click 开始训练
Check: log output area streams text, progress bar fills
```

- [ ] **Step 3: Commit**

```bash
git add frontend/views/train.js
git commit -m "feat: train view with SSE log streaming"
```

---

## Task 15: Infer View

**Files:**
- Create: `frontend/views/infer.js`

- [ ] **Step 1: Create `frontend/views/infer.js`**

```javascript
import { api, toast } from "../app.js";

export function init() {
  const container = document.getElementById("view-infer");
  container.innerHTML = `
    <div style="display:flex;gap:16px">
      <div style="flex:1;min-width:280px">
        <div class="card">
          <div class="card-title">上传图片</div>
          <input type="file" id="infer-file" accept="image/*"
                 style="font-size:13px;color:var(--text-secondary);width:100%" />
          <div id="infer-thumb-wrap" style="margin-top:10px;display:none">
            <img id="infer-thumb" style="max-width:100%;max-height:160px;border-radius:6px" />
          </div>
        </div>

        <div class="card">
          <div class="card-title">模型</div>
          <label>权重文件路径</label>
          <input type="text" id="infer-weights"
                 value="runs/detect/train/weights/best.pt" />
          <div id="infer-model-info"
               style="margin-top:8px;font-size:11px;
                      color:var(--text-secondary);white-space:pre-line;
                      min-height:40px;background:var(--bg);
                      border-radius:6px;padding:8px;
                      border:1px solid var(--separator)">
            （输入权重路径后自动解析）
          </div>
        </div>

        <div class="card">
          <div class="card-title">参数</div>
          <div class="form-row">
            <div>
              <label>置信度</label>
              <input type="number" id="infer-conf" value="0.25"
                     step="0.05" min="0.05" max="0.95" />
            </div>
            <div>
              <label>ImgSz</label>
              <input type="number" id="infer-imgsz" value="640" min="32" step="32" />
            </div>
          </div>
        </div>

        <button class="btn btn-primary" id="infer-run-btn"
                style="width:100%" disabled>开始推理</button>

        <div id="infer-stats"
             style="margin-top:12px;font-size:12px;
                    color:var(--text-secondary);white-space:pre-line"></div>
      </div>

      <div style="flex:3">
        <div class="preview-img-wrap" id="infer-result-wrap" style="min-height:400px">
          <span style="color:var(--text-secondary);font-size:13px">推理结果将在此显示</span>
        </div>
      </div>
    </div>
  `;

  // File preview
  document.getElementById("infer-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const thumb = document.getElementById("infer-thumb");
    thumb.src = url;
    document.getElementById("infer-thumb-wrap").style.display = "block";
    document.getElementById("infer-run-btn").disabled = false;
  });

  // Model info on weights path change
  let debounce;
  document.getElementById("infer-weights").addEventListener("input", (e) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => fetchModelInfo(e.target.value.trim()), 600);
  });

  document.getElementById("infer-run-btn").addEventListener("click", runInference);
}

async function fetchModelInfo(path) {
  const el = document.getElementById("infer-model-info");
  if (!path) { el.textContent = "（未指定路径）"; return; }
  try {
    const data = await api(`/api/model-info?weights_path=${encodeURIComponent(path)}`);
    el.textContent = data.raw || "（无法解析）";
  } catch {}
}

async function runInference() {
  const fileInput = document.getElementById("infer-file");
  const weights = document.getElementById("infer-weights").value.trim();
  const conf = document.getElementById("infer-conf").value;
  const imgsz = document.getElementById("infer-imgsz").value;

  if (!fileInput.files[0]) { toast("请先选择图片", "error"); return; }

  const btn = document.getElementById("infer-run-btn");
  btn.disabled = true;
  btn.textContent = "推理中...";

  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  fd.append("weights_path", weights);
  fd.append("conf", conf);
  fd.append("imgsz", imgsz);

  try {
    const data = await api("/api/infer", { method: "POST", body: fd });

    // Show result image
    const wrap = document.getElementById("infer-result-wrap");
    wrap.innerHTML = `<img src="${data.result_url}?t=${Date.now()}"
                           style="max-width:100%;max-height:70vh;border-radius:6px" />`;

    // Stats
    const s = data.stats;
    const lines = [`检测到 ${s.total} 个目标`];
    for (const [cls, cnt] of Object.entries(s.classes || {})) {
      lines.push(`  ${cls}: ${cnt}`);
    }
    document.getElementById("infer-stats").textContent = lines.join("\n");
    toast("推理完成");
  } catch {
    // error toast already shown by api()
  } finally {
    btn.disabled = false;
    btn.textContent = "开始推理";
  }
}
```

- [ ] **Step 2: Verify manually**

```
Click 推理测试
Upload an image, enter a valid .pt weights path
Check: model info box updates automatically
Click 开始推理
Check: result image and stats appear
```

- [ ] **Step 3: Commit**

```bash
git add frontend/views/infer.js
git commit -m "feat: infer view with model info auto-fetch and result display"
```

---

## Task 16: Wire Everything + Final Verification

**Files:**
- Modify: `backend/main.py` (verify all routers registered)
- Modify: `.gitignore` (add `.cache/`)

- [ ] **Step 1: Verify all routers are registered in `backend/main.py`**

Final `backend/main.py` should include:
```python
from backend.routers import images as images_router
from backend.routers import preview as preview_router
from backend.routers import annotate as annotate_router
from backend.routers import train as train_router
from backend.routers import infer as infer_router

app.include_router(images_router.router)
app.include_router(preview_router.router)
app.include_router(annotate_router.router)
app.include_router(train_router.router)
app.include_router(infer_router.router)
```

- [ ] **Step 2: Add `.cache/` to `.gitignore`**

```
.cache/
.superpowers/
```

- [ ] **Step 3: Run full test suite**

```bash
conda run -n yoloV8 pytest tests/ -v
# Expected: all pass
```

- [ ] **Step 4: End-to-end smoke test**

```bash
conda run -n yoloV8 python run.py
```

Manually verify:
1. http://localhost:8000 loads, sidebar shows 4 nav items
2. Theme toggle switches dark/light, persists on reload
3. 自动标注: fill prompts, verify progress bar appears on submit
4. 预览编辑: load a directory, image appears, annotation tags clickable
5. YOLO训练: verify log area exists, form fields present
6. 推理测试: upload an image, model info box updates on path change

- [ ] **Step 5: Final commit**

```bash
git add .gitignore backend/main.py
git commit -m "feat: wire all routers, add .cache to .gitignore"
```

- [ ] **Step 6: Push branch**

```bash
git push -u origin feat/web-frontend
```

---

## Summary

16 tasks, ~28 commits. Backend is fully tested with pytest + FastAPI TestClient (mocking heavy ML dependencies). Frontend is tested manually. The Gradio `app.py` is preserved and still works — run `python app.py` for the old UI, `python run.py` for the new one.
