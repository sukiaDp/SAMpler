import re
import sys
import threading
import time
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models import TrainRequest
from backend.tasks import registry
from backend.utils import build_model_name

router = APIRouter(tags=["train"])

# Per-task log buffers: task_id → list[str]
_log_buffers: dict[str, list[str]] = {}
_log_lock = threading.Lock()


@router.post("/api/train", status_code=202)
def start_train(req: TrainRequest):
    task_id = registry.create()
    with _log_lock:
        _log_buffers[task_id] = []
    t = threading.Thread(
        target=_run_training, args=(task_id, req), daemon=True
    )
    t.start()
    return {"task_id": task_id}


@router.get("/api/train/{task_id}/logs")
def stream_logs(task_id: str):
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    def generate():
        last = 0
        idle_ticks = 0  # ticks since last data sent
        while True:
            with _log_lock:
                buf = _log_buffers.get(task_id, [])
                new_lines = buf[last:]
                last += len(new_lines)
            if new_lines:
                idle_ticks = 0
            for line in new_lines:
                yield f"data: {line}\n\n"

            t = registry.get(task_id)
            if t and t.status in ("done", "error"):
                # Drain remaining
                with _log_lock:
                    buf = _log_buffers.get(task_id, [])
                    for line in buf[last:]:
                        yield f"data: {line}\n\n"
                if t.status == "error":
                    yield f"event: train_error\ndata: {t.message}\n\n"
                else:
                    yield "event: done\ndata: 训练完成\n\n"
                break

            # Send a comment-line heartbeat every ~5 s to keep connection alive
            idle_ticks += 1
            if idle_ticks >= 10:
                yield ": heartbeat\n\n"
                idle_ticks = 0
            time.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _run_training(task_id: str, req: TrainRequest):
    registry.update(task_id, status="running")

    def log(text: str):
        with _log_lock:
            buf = _log_buffers.get(task_id)
            if buf is not None:
                buf.append(text.rstrip())

    try:
        data_yaml = Path(req.dataset_dir) / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

        model_file = build_model_name(req.yolo_version, req.model_size, req.task)
        log(f"模型: {model_file}")
        log(f"任务: {req.task}")
        log(f"数据集: {data_yaml}")
        log(f"Epochs: {req.epochs}, ImgSz: {req.imgsz}")
        log("=" * 60)

        save_dir = "runs/"  # default; overwritten on success

        class LogCapture:
            def __init__(self, orig):
                self.orig = orig
                self._line = ""
            def write(self, text):
                self.orig.write(text)
                for ch in text:
                    if ch == "\r":
                        self._line = ""        # carriage return: discard partial line
                    elif ch == "\n":
                        clean = _ANSI_RE.sub("", self._line).strip()
                        if clean:
                            log(clean)
                        self._line = ""        # newline: commit and reset
                    else:
                        self._line += ch
            def flush(self):
                self.orig.flush()

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = LogCapture(old_out)
        sys.stderr = LogCapture(old_err)
        try:
            from ultralytics import YOLO
            model = YOLO(model_file)
            results = model.train(
                data=str(data_yaml),
                epochs=req.epochs,
                imgsz=req.imgsz,
                task=req.task,
                verbose=True,
            )
            save_dir = str(results.save_dir) if hasattr(results, "save_dir") else "runs/"
            log(f"\n{'=' * 60}")
            log(f"训练完成！结果保存在: {save_dir}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        registry.update(task_id, status="done", message="训练完成",
                        result={"save_dir": save_dir})
    except Exception as e:
        log(f"训练出错: {e}")
        registry.update(task_id, status="error", message=str(e))
