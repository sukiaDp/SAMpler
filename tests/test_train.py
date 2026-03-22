from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_train_returns_task_id(tmp_path):
    # Create a minimal data.yaml so the router doesn't error on validation
    (tmp_path / "data.yaml").write_text("names:\n  0: cat\n")
    r = client.post("/api/train", json={
        "dataset_dir": str(tmp_path),
        "task": "detect",
        "yolo_version": "YOLOv8",
        "model_size": "n",
        "epochs": 1,
        "imgsz": 32,
    })
    assert r.status_code == 202
    assert "task_id" in r.json()


def test_train_missing_data_yaml():
    r = client.post("/api/train", json={
        "dataset_dir": "/nonexistent/xyz",
        "task": "detect",
        "yolo_version": "YOLOv8",
        "model_size": "n",
        "epochs": 1,
        "imgsz": 32,
    })
    # Should accept and return task_id; error surfaces via task status
    assert r.status_code == 202


def test_train_logs_sse_invalid_task():
    # SSE endpoint for unknown task should close immediately or 404
    r = client.get("/api/train/nonexistent/logs")
    assert r.status_code == 404
