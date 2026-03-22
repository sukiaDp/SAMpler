import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.models import InferResponse, ModelInfoResponse

router = APIRouter(tags=["infer"])

PREVIEW_CACHE = Path(__file__).parent.parent.parent / ".cache" / "previews"


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
