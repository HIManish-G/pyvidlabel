"""Background jobs for slow operations (hashing, deep validation, indexing, caching, builds).

One worker thread runs jobs one after another, so two heavy jobs never fight over the disk. A job function receives a `JobContext`: it
reports progress with `ctx.progress(fraction, message)` (fraction may be None when the work cannot be measured) and calls `ctx.check()`
between steps, which raises `JobCancelled` when the user pressed cancel. Cancelling stops a job at its next step, not in the middle of
a single library call.
"""
from __future__ import annotations

import queue
import secrets
import threading
import time
import traceback
from typing import Any, Callable

KEEP = 100          # finished jobs remembered


class JobCancelled(Exception):
    """Raised inside a job when it was cancelled."""


class Job:
    def __init__(self, kind: str, title: str):
        self.id = secrets.token_hex(6)
        self.kind, self.title = kind, title
        self.status = "queued"          # queued, running, done, failed, cancelled
        self.progress: float | None = None
        self.messages: list[str] = []
        self.result: Any = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.cancel_requested = threading.Event()

    def as_dict(self, with_messages: bool = True) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "progress": self.progress,
                "messages": self.messages[-200:] if with_messages else [], "result": self.result, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished,
                "cancel_requested": self.cancel_requested.is_set()}


class JobContext:
    def __init__(self, job: Job):
        self._job = job

    def progress(self, fraction: float | None = None, message: str | None = None) -> None:
        self._job.progress = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        if message:
            self._job.messages.append(message)

    def check(self) -> None:
        if self._job.cancel_requested.is_set():
            raise JobCancelled()


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[tuple[Job, Callable[[JobContext], Any]]]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, kind: str, title: str, fn: Callable[[JobContext], Any]) -> Job:
        job = Job(kind, title)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            for old in [i for i in self._order if self._jobs[i].finished is not None][:-KEEP]:
                self._order.remove(old)
                del self._jobs[old]
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="vidlabel-jobs", daemon=True)
                self._worker.start()
        self._queue.put((job, fn))
        return job

    def _run(self) -> None:
        while True:
            try:
                job, fn = self._queue.get(timeout=30)
            except queue.Empty:
                return                                             # the thread ends when idle and is restarted on demand
            if job.cancel_requested.is_set():
                job.status, job.finished = "cancelled", time.time()
                continue
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(JobContext(job))
                job.status = "done"
                job.progress = 1.0
            except JobCancelled:
                job.status = "cancelled"
            except BaseException as e:                              # the job's error goes to the page, not to the server log only
                job.status = "failed"
                job.error = str(e) or type(e).__name__
                job.messages.append(traceback.format_exc(limit=3) if _debug() else job.error)
            job.finished = time.time()

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"no job '{job_id}'") from None

    def list(self) -> list[dict]:
        with self._lock:
            return [self._jobs[i].as_dict(with_messages=False) for i in reversed(self._order)]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.finished is None:
            job.cancel_requested.set()
        return job

    def busy(self) -> bool:
        return any(j.finished is None for j in self._jobs.values())


def _debug() -> bool:
    import os
    return os.environ.get("VIDLABEL_DEBUG") == "1"
