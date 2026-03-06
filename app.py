"""
SAM3 自动标注 + YOLO 训练 Gradio 应用

功能：
1. 自动标注：使用 SAM3 文本提示对图片目录进行自动标注，生成 YOLO 格式数据集
2. 标注预览：逐张翻页查看标注结果，支持删除整张图或单独删除某个标注实例
3. 一键训练：选择 YOLO 模型版本和参数，一键启动训练
4. 单图预览：拖入单张图片，绘制 SAM3 分割结果
"""

import random
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml

from sam3 import SAM3Segmentor, draw_masks_on_image


def browse_folder(current_path: str = "") -> str:
    """打开系统文件夹选择对话框，返回选中的路径"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    initial = current_path if current_path and Path(current_path).is_dir() else "."
    selected = filedialog.askdirectory(
        title="选择文件夹",
        initialdir=initial,
    )
    root.destroy()

    return selected if selected else current_path


# ── 全局状态 ──────────────────────────────────────────────────────────────────
preview_paths: list[str] = []  # 预览图路径列表
preview_annotations: list[dict] = []  # 与 preview_paths 平行，每张图的标注实例元数据
annotation_mode: str = "detect"  # 当前标注模式
dataset_dir: str = "dataset"  # 当前数据集目录

# ── SAM3 单例 ─────────────────────────────────────────────────────────────────
_segmentor: SAM3Segmentor | None = None
_segmentor_conf: float | None = None


def get_segmentor(conf: float = 0.25) -> SAM3Segmentor:
    """懒加载 SAM3Segmentor 单例，conf 变化时更新阈值而不重建模型"""
    global _segmentor, _segmentor_conf
    if _segmentor is None:
        _segmentor = SAM3Segmentor(model_path="sam3.pt", conf=conf, device="0", half=True)
        _segmentor_conf = conf
    elif conf != _segmentor_conf:
        _segmentor.predictor.args.conf = conf
        _segmentor_conf = conf
    return _segmentor


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def collect_images(image_dir: str) -> list[str]:
    """收集目录下所有图片文件"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    files = [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
    if not files:
        raise ValueError(f"目录中没有找到图片: {image_dir}")
    return sorted(files)


def xyxy_to_xywh_norm(box, img_w, img_h):
    """xyxy → 归一化 xywh 中心格式"""
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


def mask_to_polygon_norm(mask, img_w, img_h):
    """mask → 归一化多边形点列表，返回 None 如果无有效轮廓"""
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
    points = approx.reshape(-1, 2).astype(np.float64)
    points[:, 0] /= img_w
    points[:, 1] /= img_h
    return points


def build_model_name(version: str, size: str, task: str) -> str:
    """构建 YOLO 模型文件名"""
    seg_suffix = "-seg" if task == "segment" else ""
    prefix_map = {
        "YOLOv8": f"yolov8{size}{seg_suffix}.pt",
        "YOLOv11": f"yolo11{size}{seg_suffix}.pt",
        "YOLO26": f"yolo26{size}{seg_suffix}.pt",
    }
    return prefix_map[version]


def generate_colors(n: int) -> list[tuple]:
    """生成 n 个颜色 (BGR)，与 draw_masks_on_image 使用相同种子"""
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]


# ── 标注实例重绘 & UI 构建 ────────────────────────────────────────────────────

def redraw_from_annotations(img_bgr: np.ndarray, contours: list[dict], mode: str) -> np.ndarray:
    """从标注数据重绘预览图（不需要原始 mask，纯解析 label_line）"""
    output = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    alpha = 0.4

    for ann in contours:
        color = ann["color"]
        class_name = ann["class_name"]
        parts = ann["label_line"].split()
        values = list(map(float, parts[1:]))

        if mode == "detect":
            xc, yc, bw, bh = values
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            # 半透明填充
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        else:  # segment
            pts = np.array(values).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts_int = pts.astype(np.int32)
            # 半透明填充多边形
            mask_layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_layer, [pts_int], 255)
            colored = np.zeros_like(img_bgr)
            colored[mask_layer > 0] = color
            output = cv2.addWeighted(output, 1, colored, alpha, 0)
            cv2.polylines(output, [pts_int], True, color, 2)
            # bbox
            bx1, by1 = pts_int.min(axis=0)
            bx2, by2 = pts_int.max(axis=0)
            x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)

        # 标签文字
        (tw, th), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(output, class_name, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


def build_contour_html(contours: list[dict]) -> str:
    """构建标注实例列表 HTML，含颜色色块"""
    if not contours:
        return "<p style='color:#888'>当前图片无标注</p>"
    rows = []
    for i, c in enumerate(contours):
        b, g, r = c["color"]  # BGR → RGB
        css = f"rgb({r},{g},{b})"
        # 根据 label_line 字段数判断类型：5个数值=检测框，更多=分割多边形
        n_values = len(c["label_line"].split()) - 1  # 去掉 class_id
        kind = "框" if n_values == 4 else "多边形"
        rows.append(
            f"<tr>"
            f"<td style='text-align:center;padding:4px 8px'>{i}</td>"
            f"<td style='text-align:center;padding:4px 8px'>"
            f"<div style='width:18px;height:18px;background:{css};"
            f"border:1px solid #999;border-radius:3px;display:inline-block'></div></td>"
            f"<td style='padding:4px 8px'>{c['class_name']}</td>"
            f"<td style='text-align:center;padding:4px 8px'>{kind}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr style='border-bottom:2px solid #aaa'>"
        "<th style='padding:4px 8px'>序号</th>"
        "<th style='padding:4px 8px'>颜色</th>"
        "<th style='padding:4px 8px'>类别</th>"
        "<th style='padding:4px 8px'>类型</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def build_contour_choices(contours: list[dict]) -> list[str]:
    """构建 CheckboxGroup 的选项列表"""
    return [f"#{i} {c['class_name']}" for i, c in enumerate(contours)]


def _get_contour_outputs(idx: int):
    """获取指定索引图片的标注实例 HTML 和 CheckboxGroup 更新"""
    if not preview_annotations or idx < 0 or idx >= len(preview_annotations):
        return "<p style='color:#888'>当前图片无标注</p>", gr.update(choices=[], value=[])
    contours = preview_annotations[idx]["contours"]
    return build_contour_html(contours), gr.update(choices=build_contour_choices(contours), value=[])


# ── 阶段一：自动标注 ─────────────────────────────────────────────────────────

def run_annotation(
    image_dir: str,
    prompts_text: str,
    mode: str,
    conf: float,
    val_ratio: float,
    output_dir: str,
    max_instances: int,
    sort_mode: str,
    progress=gr.Progress(track_tqdm=False),
):
    """执行自动标注流程"""
    global preview_paths, preview_annotations, annotation_mode, dataset_dir
    max_instances = int(max_instances)

    annotation_mode = mode
    dataset_dir = output_dir

    # 解析提示词
    names = [p.strip() for p in prompts_text.split(",") if p.strip()]
    if not names:
        raise gr.Error("请输入至少一个提示词")

    # 收集图片
    image_files = collect_images(image_dir)
    random.seed(42)
    random.shuffle(image_files)

    # 划分 train/val
    n_val = max(1, int(len(image_files) * val_ratio))
    val_files = set(image_files[:n_val])
    train_files = image_files[n_val:]

    # 创建目录
    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_dir = out / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # 获取 SAM3（首次调用时加载，后续复用）
    progress(0, desc="加载 SAM3 模型...")
    segmentor = get_segmentor(conf)

    # 统计
    class_counts = {name: 0 for name in names}
    total = len(image_files)
    preview_paths_local = []
    preview_annotations_local = []

    for idx, img_path in enumerate(image_files):
        progress((idx + 1) / total, desc=f"标注中 {idx + 1}/{total}")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_name = Path(img_path).stem + ".jpg"
        split = "val" if img_path in val_files else "train"

        all_masks = []
        all_boxes = []
        all_labels = []
        all_confs = []
        label_lines = []

        for class_id, prompt in enumerate(names):
            force = (class_id == 0)
            masks, boxes, confs = segmentor.predict(img_path, prompt, force_reload=force)

            if masks is None:
                continue

            for i in range(len(masks)):
                mask = masks[i]
                box = boxes[i]
                c = confs[i]

                if mode == "detect":
                    xc, yc, bw, bh = xyxy_to_xywh_norm(box, w, h)
                    label_lines.append(
                        f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    )
                else:  # segment
                    poly = mask_to_polygon_norm(mask, w, h)
                    if poly is None:
                        continue
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                    label_lines.append(f"{class_id} {coords}")

                all_masks.append(mask)
                all_boxes.append(box)
                all_confs.append(float(c))
                all_labels.append(prompt)
                class_counts[prompt] += 1

        # 限制最大标注数
        if len(all_masks) > max_instances:
            if sort_mode == "置信度":
                order = sorted(range(len(all_confs)), key=lambda k: all_confs[k], reverse=True)
            else:  # 框面积
                order = sorted(range(len(all_boxes)), key=lambda k: (
                    (all_boxes[k][2] - all_boxes[k][0]) * (all_boxes[k][3] - all_boxes[k][1])
                ), reverse=True)
            keep = sorted(order[:max_instances])  # 保持原始顺序
            all_masks = [all_masks[k] for k in keep]
            all_boxes = [all_boxes[k] for k in keep]
            all_confs = [all_confs[k] for k in keep]
            all_labels = [all_labels[k] for k in keep]
            label_lines = [label_lines[k] for k in keep]

        # 写入标签文件
        label_path = out / "labels" / split / (Path(img_path).stem + ".txt")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

        # 复制图片
        dst_img = out / "images" / split / img_name
        if Path(img_path).resolve() != dst_img.resolve():
            shutil.copy2(img_path, dst_img)

        # 为每个标注实例分配颜色并存储元数据
        img_colors = generate_colors(len(all_masks))
        contours_data = []
        for i in range(len(all_masks)):
            contours_data.append({
                "label_line": label_lines[i],
                "class_name": all_labels[i],
                "color": img_colors[i],
            })
        preview_annotations_local.append({
            "img_path": str(dst_img),
            "split": split,
            "stem": Path(img_path).stem,
            "contours": contours_data,
        })

        # 生成预览图（使用已分配的颜色，保证预览和列表一致）
        preview_file = preview_dir / img_name
        if all_masks:
            masks_arr = np.array(all_masks)
            boxes_arr = np.array(all_boxes)
            vis = draw_masks_on_image(
                img, masks_arr, boxes_arr,
                labels=all_labels, alpha=0.4, colors=img_colors,
            )
            cv2.imwrite(str(preview_file), vis)
        else:
            cv2.imwrite(str(preview_file), img)
        preview_paths_local.append(str(preview_file))

    # 生成 data.yaml
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(names)},
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

    # 更新全局列表
    preview_paths = preview_paths_local
    preview_annotations = preview_annotations_local

    # 构建统计信息
    stats = (
        f"标注完成！\n"
        f"总图片数: {total}\n"
        f"训练集: {len(train_files)} 张\n"
        f"验证集: {len(val_files)} 张\n"
        f"标注模式: {mode}\n"
        f"数据集路径: {out.resolve()}\n\n"
        f"各类别检测数量:\n"
    )
    for name, cnt in class_counts.items():
        stats += f"  {name}: {cnt}\n"

    first_preview = preview_paths[0] if preview_paths else None
    page_text = f"1 / {len(preview_paths)}" if preview_paths else "0 / 0"
    contour_html, contour_cb = _get_contour_outputs(0)

    return stats, first_preview, page_text, 0, contour_html, contour_cb


# ── 阶段二：标注预览 ─────────────────────────────────────────────────────────

def navigate_preview(current_idx: int, direction: int):
    """翻页查看预览图"""
    if not preview_paths:
        html, cb = _get_contour_outputs(-1)
        return None, "0 / 0", current_idx, html, cb

    new_idx = current_idx + direction
    new_idx = max(0, min(new_idx, len(preview_paths) - 1))

    img = cv2.imread(preview_paths[new_idx])
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    page_text = f"{new_idx + 1} / {len(preview_paths)}"
    html, cb = _get_contour_outputs(new_idx)
    return img, page_text, new_idx, html, cb


def delete_current(current_idx: int):
    """删除当前预览图及其对应的数据集图片和标签文件，自动跳到下一张"""
    global preview_paths, preview_annotations, dataset_dir

    if not preview_paths or current_idx < 0 or current_idx >= len(preview_paths):
        html, cb = _get_contour_outputs(-1)
        return None, "0 / 0", current_idx, "", html, cb

    preview_file = Path(preview_paths[current_idx])
    stem = preview_file.stem
    out = Path(dataset_dir)

    deleted = []

    if preview_file.exists():
        preview_file.unlink()
        deleted.append(f"预览: {preview_file.name}")

    for split in ("train", "val"):
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img_file = out / "images" / split / (stem + ext)
            if img_file.exists():
                img_file.unlink()
                deleted.append(f"图片: images/{split}/{img_file.name}")
                break
        label_file = out / "labels" / split / (stem + ".txt")
        if label_file.exists():
            label_file.unlink()
            deleted.append(f"标签: labels/{split}/{label_file.name}")

    preview_paths.pop(current_idx)
    preview_annotations.pop(current_idx)

    if not preview_paths:
        html, cb = _get_contour_outputs(-1)
        return None, "0 / 0", 0, "已删除: " + ", ".join(deleted) + "\n列表已空", html, cb

    new_idx = min(current_idx, len(preview_paths) - 1)

    img = cv2.imread(preview_paths[new_idx])
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    page_text = f"{new_idx + 1} / {len(preview_paths)}"
    msg = "已删除: " + ", ".join(deleted)
    html, cb = _get_contour_outputs(new_idx)
    return img, page_text, new_idx, msg, html, cb


def delete_selected_contours(current_idx: int, selected: list[str]):
    """删除当前图片中用户选中的标注实例，重绘预览并更新标签文件"""
    global preview_annotations, annotation_mode, dataset_dir

    if not selected:
        html, cb = _get_contour_outputs(current_idx)
        return None, "", html, cb  # 图片不变

    if current_idx < 0 or current_idx >= len(preview_annotations):
        html, cb = _get_contour_outputs(-1)
        return None, "无有效图片", html, cb

    ann = preview_annotations[current_idx]
    contours = ann["contours"]

    # 解析选中的序号（"#0 tree" → 0）
    indices_to_delete = set()
    for s in selected:
        try:
            indices_to_delete.add(int(s.split()[0].lstrip("#")))
        except (ValueError, IndexError):
            continue

    deleted_names = [contours[i]["class_name"] for i in sorted(indices_to_delete) if i < len(contours)]

    # 过滤保留的标注实例
    new_contours = [c for i, c in enumerate(contours) if i not in indices_to_delete]
    ann["contours"] = new_contours

    # 重写标签文件
    out = Path(dataset_dir)
    label_file = out / "labels" / ann["split"] / (ann["stem"] + ".txt")
    lines = [c["label_line"] for c in new_contours]
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    # 重绘预览图
    img_bgr = cv2.imread(ann["img_path"])
    if img_bgr is None:
        html, cb = _get_contour_outputs(current_idx)
        return None, "无法读取原图", html, cb

    if new_contours:
        vis = redraw_from_annotations(img_bgr, new_contours, annotation_mode)
    else:
        vis = img_bgr

    # 覆盖预览缓存文件
    cv2.imwrite(preview_paths[current_idx], vis)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    msg = f"已删除 {len(indices_to_delete)} 个标注: {', '.join(deleted_names)}"
    html, cb = _get_contour_outputs(current_idx)
    return vis_rgb, msg, html, cb


# ── 阶段三：一键训练 ─────────────────────────────────────────────────────────

def run_training_async(
    yolo_version: str,
    model_size: str,
    epochs: int,
    imgsz: int,
):
    """异步启动训练，返回实时日志的生成器"""
    global annotation_mode, dataset_dir

    task = annotation_mode
    model_file = build_model_name(yolo_version, model_size, task)
    data_yaml = str(Path(dataset_dir).resolve() / "data.yaml")

    if not Path(data_yaml).exists():
        raise gr.Error(f"数据集配置不存在: {data_yaml}\n请先完成标注步骤")

    log_lines = []
    training_done = threading.Event()

    log_lines.append(f"模型: {model_file}\n")
    log_lines.append(f"任务: {task}\n")
    log_lines.append(f"数据集: {data_yaml}\n")
    log_lines.append(f"Epochs: {epochs}, ImgSz: {imgsz}\n")
    log_lines.append("=" * 60 + "\n")

    class LogCapture:
        def __init__(self, original, buffer):
            self.original = original
            self.buffer = buffer

        def write(self, text):
            self.original.write(text)
            if text.strip():
                self.buffer.append(text)

        def flush(self):
            self.original.flush()

    def train_thread():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = LogCapture(old_stdout, log_lines)
        sys.stderr = LogCapture(old_stderr, log_lines)
        try:
            from ultralytics import YOLO
            model = YOLO(model_file)
            results = model.train(
                data=data_yaml,
                epochs=int(epochs),
                imgsz=int(imgsz),
                task=task,
                verbose=True,
            )
            save_dir = str(results.save_dir) if hasattr(results, "save_dir") else "runs/"
            log_lines.append(f"\n{'=' * 60}\n训练完成！结果保存在: {save_dir}\n")
        except Exception as e:
            log_lines.append(f"\n训练出错: {e}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            training_done.set()

    t = threading.Thread(target=train_thread, daemon=True)
    t.start()

    last_len = 0
    while not training_done.is_set():
        current = "".join(log_lines)
        if len(current) != last_len:
            last_len = len(current)
            yield current
        time.sleep(1)

    yield "".join(log_lines)


# ── 单图预览 ─────────────────────────────────────────────────────────────────

def run_single_image_segmentation(
    image_path: str, prompts_text: str, conf: float,
    max_instances: int, sort_mode: str,
):
    """对单张图片执行 SAM3 分割并绘制结果"""
    if image_path is None:
        raise gr.Error("请先上传一张图片")

    names = [p.strip() for p in prompts_text.split(",") if p.strip()]
    if not names:
        raise gr.Error("请输入至少一个提示词")

    img = cv2.imread(image_path)
    if img is None:
        raise gr.Error(f"无法读取图片: {image_path}")

    max_instances = int(max_instances)
    segmentor = get_segmentor(conf)

    all_masks = []
    all_boxes = []
    all_confs = []
    all_labels = []

    for i, prompt in enumerate(names):
        force = (i == 0)
        masks, boxes, confs = segmentor.predict(image_path, prompt, force_reload=force)
        if masks is None:
            continue
        for j in range(len(masks)):
            all_masks.append(masks[j])
            all_boxes.append(boxes[j])
            all_confs.append(float(confs[j]))
            all_labels.append(prompt)

    if not all_masks:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "未检测到任何目标"

    # 限制最大标注数
    if len(all_masks) > max_instances:
        if sort_mode == "置信度":
            order = sorted(range(len(all_confs)), key=lambda k: all_confs[k], reverse=True)
        else:
            order = sorted(range(len(all_boxes)), key=lambda k: (
                (all_boxes[k][2] - all_boxes[k][0]) * (all_boxes[k][3] - all_boxes[k][1])
            ), reverse=True)
        keep = sorted(order[:max_instances])
        all_masks = [all_masks[k] for k in keep]
        all_boxes = [all_boxes[k] for k in keep]
        all_labels = [all_labels[k] for k in keep]

    masks_arr = np.array(all_masks)
    boxes_arr = np.array(all_boxes)
    vis = draw_masks_on_image(img, masks_arr, boxes_arr, labels=all_labels, alpha=0.4)

    info = f"检测到 {len(all_masks)} 个目标\n"
    counts = Counter(all_labels)
    for name, cnt in counts.items():
        info += f"  {name}: {cnt}\n"

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), info


# ── Gradio 界面 ──────────────────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="SAM3 自动标注 + YOLO 训练", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SAM3 自动标注 + YOLO 训练")

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════
            #  标签页一：自动标注 + 预览 + 训练
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("自动标注与训练"):

                current_idx = gr.State(0)

                # ── 阶段一 ──
                gr.Markdown("## 阶段一：自动标注")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            img_dir = gr.Textbox(label="图片目录路径", value="rawData", scale=4)
                            img_browse = gr.Button("浏览...", scale=1, min_width=70)
                        prompts = gr.Textbox(
                            label="文本提示词（逗号分隔）",
                            placeholder="tree,person,bush",
                        )
                        mode = gr.Radio(
                            choices=["detect", "segment"],
                            value="detect",
                            label="标注模式",
                        )
                        conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="置信度阈值",
                        )
                        val_slider = gr.Slider(
                            minimum=0.05, maximum=0.5, value=0.2, step=0.05,
                            label="验证集比例",
                        )
                        with gr.Row():
                            max_inst = gr.Number(label="每张图最大标注数", value=7, precision=0)
                            sort_mode = gr.Radio(
                                choices=["置信度", "框面积"],
                                value="置信度",
                                label="超出时保留依据",
                            )
                        with gr.Row():
                            out_dir = gr.Textbox(label="输出目录", value="dataset/", scale=4)
                            out_browse = gr.Button("浏览...", scale=1, min_width=70)
                        annotate_btn = gr.Button("开始标注", variant="primary")

                    with gr.Column(scale=1):
                        stats_box = gr.Textbox(
                            label="标注统计", lines=12, interactive=False
                        )

                # ── 阶段二 ──
                gr.Markdown("## 阶段二：标注预览")
                with gr.Row():
                    prev_btn = gr.Button("上一张")
                    page_label = gr.Textbox(
                        value="0 / 0", label="当前图片", interactive=False,
                        scale=1, min_width=120,
                    )
                    next_btn = gr.Button("下一张")
                    delete_btn = gr.Button("删除此张", variant="stop")

                with gr.Row():
                    with gr.Column(scale=3):
                        preview_image = gr.Image(label="标注预览", height=600)
                    with gr.Column(scale=1):
                        contour_html = gr.HTML(
                            value="<p style='color:#888'>当前图片无标注</p>",
                            label="标注实例列表",
                        )
                        contour_select = gr.CheckboxGroup(
                            choices=[], value=[], label="选择要删除的标注",
                        )
                        delete_contour_btn = gr.Button("删除选中标注", variant="stop")

                delete_msg = gr.Textbox(label="操作提示", interactive=False)

                # ── 阶段三 ──
                gr.Markdown("## 阶段三：一键训练")
                with gr.Row():
                    with gr.Column(scale=1):
                        yolo_version = gr.Dropdown(
                            choices=["YOLOv8", "YOLOv11", "YOLO26"],
                            value="YOLOv8",
                            label="YOLO 版本",
                        )
                        model_size = gr.Dropdown(
                            choices=["n", "s", "m", "l", "x"],
                            value="n",
                            label="模型规格",
                        )
                        task_display = gr.Textbox(
                            label="任务类型（跟随标注模式）",
                            value="detect",
                            interactive=False,
                        )
                        epochs = gr.Number(label="训练轮数 (epochs)", value=100)
                        imgsz = gr.Number(label="图片尺寸 (imgsz)", value=640)
                        train_btn = gr.Button("开始训练", variant="primary")

                    with gr.Column(scale=2):
                        train_log = gr.Textbox(
                            label="训练日志",
                            lines=25,
                            max_lines=50,
                            interactive=False,
                            autoscroll=True,
                        )

                # ── 事件绑定 ──

                # 浏览文件夹
                img_browse.click(fn=browse_folder, inputs=[img_dir], outputs=[img_dir])
                out_browse.click(fn=browse_folder, inputs=[out_dir], outputs=[out_dir])

                # 标注
                annotate_btn.click(
                    fn=run_annotation,
                    inputs=[img_dir, prompts, mode, conf_slider, val_slider,
                            out_dir, max_inst, sort_mode],
                    outputs=[stats_box, preview_image, page_label, current_idx,
                             contour_html, contour_select],
                )

                # 标注模式 → 任务类型
                mode.change(fn=lambda m: m, inputs=[mode], outputs=[task_display])

                # 上一张
                prev_btn.click(
                    fn=lambda idx: navigate_preview(idx, -1),
                    inputs=[current_idx],
                    outputs=[preview_image, page_label, current_idx,
                             contour_html, contour_select],
                )

                # 下一张
                next_btn.click(
                    fn=lambda idx: navigate_preview(idx, 1),
                    inputs=[current_idx],
                    outputs=[preview_image, page_label, current_idx,
                             contour_html, contour_select],
                )

                # 删除整张图
                delete_btn.click(
                    fn=delete_current,
                    inputs=[current_idx],
                    outputs=[preview_image, page_label, current_idx, delete_msg,
                             contour_html, contour_select],
                )

                # 删除选中轮廓
                delete_contour_btn.click(
                    fn=delete_selected_contours,
                    inputs=[current_idx, contour_select],
                    outputs=[preview_image, delete_msg,
                             contour_html, contour_select],
                )

                # 训练
                train_btn.click(
                    fn=run_training_async,
                    inputs=[yolo_version, model_size, epochs, imgsz],
                    outputs=[train_log],
                )

            # ══════════════════════════════════════════════════════════
            #  标签页二：单图分割预览
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("单图分割预览"):
                gr.Markdown("## 拖入图片，查看 SAM3 分割结果")

                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            label="上传图片", type="filepath", height=300,
                        )
                        single_prompts = gr.Textbox(
                            label="文本提示词（逗号分隔）",
                            placeholder="tree,person,bush",
                        )
                        single_conf = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="置信度阈值",
                        )
                        with gr.Row():
                            single_max_inst = gr.Number(label="最大标注数", value=7, precision=0)
                            single_sort = gr.Radio(
                                choices=["置信度", "框面积"],
                                value="置信度",
                                label="超出时保留依据",
                            )
                        single_btn = gr.Button("开始分割", variant="primary")
                        single_info = gr.Textbox(
                            label="检测结果", lines=5, interactive=False,
                        )

                    with gr.Column(scale=2):
                        single_result = gr.Image(label="分割结果", height=600)

                single_btn.click(
                    fn=run_single_image_segmentation,
                    inputs=[single_image, single_prompts, single_conf,
                            single_max_inst, single_sort],
                    outputs=[single_result, single_info],
                )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
