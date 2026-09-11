"""
api/jobs.py

Background job manager for benchmark runs.

1.x exposed `POST /api/full-benchmark` as a synchronous endpoint that blocked
for 30 to 60 seconds while mutating a module-level dict, partly outside the
lock it had just acquired. There was no job id, no progress, and a second
caller could race the status field.

2.0 runs the benchmark on a worker thread, returns a job id immediately, and
exposes progress. All shared state lives behind one lock, and only one
benchmark may run at a time because two concurrent benchmarks would each be
measuring the other.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("BenchMind.Jobs")


@dataclass
class Job:
    id: str
    status: str = "queued"       # queued|running|finished|failed|cancelled
    stage: str = "queued"
    progress: float = 0.0        # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_result:
            data.pop("result", None)
        if self.started_at:
            end = self.finished_at or time.time()
            data["elapsed_seconds"] = round(end - self.started_at, 2)
        return data


class JobManager:
    """
    Tracks benchmark jobs. One benchmark at a time, by design: two concurrent
    runs would contend for the same cores and both results would be garbage.
    """

    def __init__(self, max_history: int = 25):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._active_id: Optional[str] = None
        self._max_history = max_history

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active_id is not None

    def active_job(self) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(self._active_id) if self._active_id else None

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._jobs[j].to_dict(include_result=False)
                    for j in reversed(self._order) if j in self._jobs]

    def submit(self, fn: Callable[..., Dict[str, Any]],
               params: Optional[Dict[str, Any]] = None) -> Job:
        """
        Start a benchmark on a worker thread.

        `fn` receives a `progress` keyword: a callable taking (stage, fraction).
        Raises RuntimeError if a benchmark is already running.
        """
        with self._lock:
            if self._active_id is not None:
                raise RuntimeError(
                    "A benchmark is already running. Two concurrent benchmarks "
                    "would contend for the same cores and invalidate both results."
                )
            job = Job(id=uuid.uuid4().hex[:12], params=params or {})
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._active_id = job.id
            self._trim_locked()

        def _progress(stage: str, fraction: float) -> None:
            with self._lock:
                j = self._jobs.get(job.id)
                if j is not None:
                    j.stage = stage
                    j.progress = max(0.0, min(1.0, float(fraction)))

        def _runner() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = time.time()
                job.stage = "starting"
            try:
                result = fn(progress=_progress)
                with self._lock:
                    job.result = result
                    job.status = "finished"
                    job.stage = "finished"
                    job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                logger.error("Benchmark job %s failed: %s", job.id, e, exc_info=True)
                with self._lock:
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = f"{type(e).__name__}: {e}"
                    job.result = {"traceback": traceback.format_exc()}
            finally:
                with self._lock:
                    job.finished_at = time.time()
                    if self._active_id == job.id:
                        self._active_id = None

        threading.Thread(target=_runner, name=f"BenchMindJob-{job.id}",
                         daemon=True).start()
        return job

    def _trim_locked(self) -> None:
        while len(self._order) > self._max_history:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def latest_finished(self) -> Optional[Job]:
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs.get(job_id)
                if job and job.status == "finished":
                    return job
        return None


job_manager = JobManager()
