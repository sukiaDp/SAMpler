"""SAM3 lazy singleton — loaded once per process, reused across requests."""
import threading
import sys
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

MODEL_PATH   = Path(__file__).parent.parent / "sam3.pt"
PREHEAT_IMG  = Path(__file__).parent.parent / "preheat.webp"


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
            # Warmup: run a dummy predict so GPU/encoder is fully initialized
            # before the first real request arrives
            _state = "warming"
            try:
                _segmentor.predict(str(PREHEAT_IMG), "object", force_reload=True)
            except Exception:
                pass  # warmup failure is non-fatal
            _state = "ready"
        elif conf != _segmentor_conf:
            _segmentor.predictor.args.conf = conf
            _segmentor_conf = conf
        return _segmentor
