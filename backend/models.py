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
