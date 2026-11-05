"""Background jobs: progress, results, errors, cancellation, one at a time."""
import threading
import time

import pytest

pytest.importorskip("starlette")
from vidlabel.ui.jobs import JobManager  # noqa: E402


def wait_for(job, timeout=10):
    t0 = time.time()
    while job.status in ("queued", "running"):
        assert time.time() - t0 < timeout
        time.sleep(0.01)
    return job


def test_result_progress_and_messages():
    jm = JobManager()

    def work(ctx):
        ctx.progress(0.5, "halfway")
        return {"x": 1}
    j = wait_for(jm.submit("k", "t", work))
    assert j.status == "done" and j.result == {"x": 1} and j.progress == 1.0 and "halfway" in j.messages


def test_error_is_reported_not_raised():
    jm = JobManager()
    j = wait_for(jm.submit("k", "t", lambda ctx: (_ for _ in ()).throw(ValueError("bad thing"))))
    assert j.status == "failed" and "bad thing" in j.error


def test_cancel_between_steps_and_one_at_a_time():
    jm = JobManager()
    started, release = threading.Event(), threading.Event()

    def slow(ctx):
        started.set()
        release.wait(5)
        ctx.check()                                   # raises when cancelled
        return "finished"
    a = jm.submit("k", "slow", slow)
    b = jm.submit("k", "second", lambda ctx: "second ran")
    assert started.wait(5) and jm.busy()
    assert b.status == "queued"                       # a single worker: the second job waits
    jm.cancel(a.id)
    release.set()
    assert wait_for(a).status == "cancelled"
    assert wait_for(b).status == "done" and b.result == "second ran"


def test_unknown_job_is_a_keyerror():
    with pytest.raises(KeyError):
        JobManager().get("nope")


def test_static_files_are_part_of_the_package():
    from vidlabel.ui import app
    for name in ("index.html", "app.js", "style.css"):
        assert (app.STATIC / name).is_file()
