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
    assert r.status_code in (400, 422)
