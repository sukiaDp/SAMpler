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
