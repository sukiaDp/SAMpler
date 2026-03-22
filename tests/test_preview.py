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
