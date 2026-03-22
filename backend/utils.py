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
