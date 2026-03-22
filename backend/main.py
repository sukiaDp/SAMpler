import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SAMpler")

from backend.routers import images as images_router, preview as preview_router, annotate as annotate_router
app.include_router(images_router.router)
app.include_router(preview_router.router)
app.include_router(annotate_router.router)

# Preview cache dir
PREVIEW_DIR = Path(__file__).parent.parent / ".cache" / "previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/previews", StaticFiles(directory=str(PREVIEW_DIR), check_dir=False), name="previews")

# Frontend static files (registered last so API routes take priority)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    """Serve frontend SPA — all non-API paths return index.html"""
    candidate = FRONTEND_DIR / full_path
    if candidate.is_file():
        return FileResponse(str(candidate))
    return FileResponse(str(FRONTEND_DIR / "index.html"))
