from backend.tasks import TaskRegistry


def test_create_returns_unique_ids():
    reg = TaskRegistry()
    ids = {reg.create() for _ in range(10)}
    assert len(ids) == 10


def test_get_missing_returns_none():
    reg = TaskRegistry()
    assert reg.get("nonexistent") is None


def test_update_task_status():
    reg = TaskRegistry()
    tid = reg.create()
    reg.update(tid, status="running", progress=5, total=10)
    t = reg.get(tid)
    assert t.status == "running"
    assert t.progress == 5


def test_update_nonexistent_does_not_raise():
    reg = TaskRegistry()
    reg.update("ghost", status="done")  # should not raise
