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

PREVIEW_CACHE = Path(__file__).parent.parent.parent / ".cache" / "previews"


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
