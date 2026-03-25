"""SAM3 lazy singleton — loaded once per process, reused across requests."""
import threading
import sys
import tempfile
import numpy as np
import cv2
from contextlib import contextmanager
from pathlib import Path

# Import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3 import SAM3Segmentor

_lock = threading.Lock()
_segmentor: SAM3Segmentor | None = None
_segmentor_conf: float | None = None

# State: "not_found" | "idle" | "loading" | "warming" | "ready" | "inferring"
_state: str = "not_found"

MODEL_PATH = Path(__file__).parent.parent / "sam3.pt"


def get_status() -> str:
    if not MODEL_PATH.exists():
        return "not_found"
    if _segmentor is None:
        return "idle"
    return _state


@contextmanager
def inferring():
    """Context manager: marks the singleton as busy during inference."""
    global _state
    prev = _state
    _state = "inferring"
    try:
        yield
    finally:
        _state = prev


def get_segmentor(conf: float = 0.25) -> SAM3Segmentor:
    global _segmentor, _segmentor_conf, _state
    with _lock:
        if _segmentor is None:
            _state = "loading"
            _segmentor = SAM3Segmentor(
                model_path=str(MODEL_PATH), conf=conf, device="0", half=True
            )
            _segmentor_conf = conf
            # Warmup: run a dummy predict on a black image so GPU/encoder is
            # fully initialized before the first real request arrives
            _state = "warming"
            try:
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    cv2.imwrite(f.name, dummy)
                    _segmentor.predict(f.name, "object", force_reload=True)
                    Path(f.name).unlink(missing_ok=True)
            except Exception:
                pass  # warmup failure is non-fatal
            _state = "ready"
        elif conf != _segmentor_conf:
            _segmentor.predictor.args.conf = conf
            _segmentor_conf = conf
        return _segmentor
