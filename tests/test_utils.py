import numpy as np
import pytest
from backend.utils import (
    collect_images, xyxy_to_xywh_norm, mask_to_polygon_norm,
    build_model_name, generate_colors, detect_mode_from_label,
)
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
