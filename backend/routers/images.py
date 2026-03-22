from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from backend.models import ImagesResponse, ImageItem
from backend.utils import IMAGE_EXTENSIONS

router = APIRouter(prefix="/api/images", tags=["images"])


def _find_label(stem: str, images_dir: Path) -> Path | None:
    """Given images/train or images/val, find the corresponding label file."""
    # images_dir = .../images/train → labels_dir = .../labels/train
    parts = images_dir.parts
    try:
        img_idx = next(i for i, p in enumerate(parts) if p == "images")
        labels_dir = Path(*parts[:img_idx]) / "labels" / parts[img_idx + 1]
        candidate = labels_dir / f"{stem}.txt"
        return candidate if candidate.exists() else None
    except (StopIteration, IndexError):
        return None


@router.get("", response_model=ImagesResponse)
def list_images(dir: str = Query(..., description="图片目录路径")):
    p = Path(dir)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir}")
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise HTTPException(status_code=404, detail=f"目录中无图片: {dir}")
    items = [
        ImageItem(
            id=f.stem,
            filename=f.name,
            has_label=_find_label(f.stem, p) is not None,
        )
        for f in files
    ]
    return ImagesResponse(files=items, total=len(items))


@router.delete("/{image_id}")
def delete_image(image_id: str, images_dir: str = Query(...)):
    p = Path(images_dir)
    # Find the image file
    img_file = next(
        (p / f"{image_id}{ext}" for ext in IMAGE_EXTENSIONS
         if (p / f"{image_id}{ext}").exists()),
        None,
    )
    if img_file is None:
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")

    deleted = []
    img_file.unlink()
    deleted.append(str(img_file))

    label = _find_label(image_id, p)
    if label and label.exists():
        label.unlink()
        deleted.append(str(label))

    return {"deleted": deleted}
