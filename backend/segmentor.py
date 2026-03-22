"""SAM3 lazy singleton — loaded once per process, reused across requests."""
import threading
import sys
from pathlib import Path

# Import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from sam3 import SAM3Segmentor

_lock = threading.Lock()
_segmentor: SAM3Segmentor | None = None
_segmentor_conf: float | None = None


def get_segmentor(conf: float = 0.25) -> SAM3Segmentor:
    global _segmentor, _segmentor_conf
    with _lock:
        if _segmentor is None:
            _segmentor = SAM3Segmentor(
                model_path="sam3.pt", conf=conf, device="0", half=True
            )
            _segmentor_conf = conf
        elif conf != _segmentor_conf:
            _segmentor.predictor.args.conf = conf
            _segmentor_conf = conf
        return _segmentor
