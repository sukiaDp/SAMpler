import threading
import uuid
from typing import Optional
from backend.models import TaskStatus


class TaskRegistry:
    """Thread-safe in-memory task store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskStatus] = {}

    def create(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = TaskStatus(task_id=task_id, status="pending")
        return task_id

    def get(self, task_id: str) -> Optional[TaskStatus]:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy() if task is not None else None

    def update(self, task_id: str, **kwargs) -> None:
        valid_fields = TaskStatus.model_fields
        for k in kwargs:
            if k not in valid_fields:
                raise ValueError(f"Unknown TaskStatus field: {k!r}")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for k, v in kwargs.items():
                setattr(task, k, v)


# Module-level singleton used by all routers
registry = TaskRegistry()
