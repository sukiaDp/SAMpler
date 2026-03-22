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

PREVIEW_CACHE = Path(__file__).parent.parent.parent / ".cache" / "previews"


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
