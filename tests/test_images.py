import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_list_images_missing_dir():
    r = client.get("/api/images", params={"dir": "/nonexistent/dir/xyz"})
    assert r.status_code == 404


def test_list_images_empty_dir(tmp_path):
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 404  # no image files


def test_list_images_returns_files(tmp_path):
    (tmp_path / "cat.jpg").touch()
    (tmp_path / "dog.png").touch()
    (tmp_path / "notes.txt").touch()
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    ids = [f["id"] for f in data["files"]]
    assert "cat" in ids and "dog" in ids


def test_list_images_has_label_false(tmp_path):
    (tmp_path / "img.jpg").touch()
    r = client.get("/api/images", params={"dir": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["files"][0]["has_label"] is False


def test_list_images_has_label_true(tmp_path):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "img.jpg").touch()
    (lbl_dir / "img.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    r = client.get("/api/images", params={"dir": str(img_dir)})
    assert r.json()["files"][0]["has_label"] is True


def test_delete_image(tmp_path):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "cat.jpg").write_bytes(b"fake")
    (lbl_dir / "cat.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    r = client.delete(
        "/api/images/cat",
        params={"images_dir": str(img_dir)},
    )
    assert r.status_code == 200
    assert not (img_dir / "cat.jpg").exists()
    assert not (lbl_dir / "cat.txt").exists()


def test_delete_image_not_found(tmp_path):
    r = client.delete(
        "/api/images/ghost",
        params={"images_dir": str(tmp_path)},
    )
    assert r.status_code == 404
