"""
api/main.py

BenchMind HTTP API.

Changes from 1.x:

* `@app.on_event` replaced with the lifespan context manager (the old decorator
  is deprecated in current FastAPI).
* `POST /api/benchmark` returns a job id immediately instead of blocking the
  request for 30 to 60 seconds. Progress is available by polling
  `/api/jobs/{id}` or streaming `/api/jobs/{id}/events`.
* Benchmark state lives in a single lock-protected JobManager instead of a
  module-level dict mutated partly outside its own lock.
* The WebSocket monitor now sends only genuinely new telemetry samples rather
  than re-sending the latest snapshot on a timer.
* The dashboard is served from here, so there is one cross-platform UI instead
  of a half-finished desktop app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ai.analysis import analyze_bottleneck, analyze_scaling, analyze_throttling
from ai.stability_engine import build_stability_report
from ai.summary_engine import generate_summary
from api.jobs import job_manager
from benchmarks.cpu.cpu_suite import run_full_cpu_suite
from benchmarks.cpu.multi_core import run_scaling_curve
from benchmarks.cpu_test import warmup_cpu
from benchmarks.gpu.gpu_suite import run_gpu_suite
from monitoring.environment import describe_fingerprint, get_environment_fingerprint
from monitoring.system_info import get_system_info
from monitoring.telemetry_service import TelemetryService
from monitoring.validity import check_run_validity
from storage.history import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("BenchMind.API")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BenchMind starting up.")
    TelemetryService.get_instance().start()
    yield
    logger.info("BenchMind shutting down.")
    TelemetryService.get_instance().stop()


app = FastAPI(
    title="BenchMind",
    version="2.0.0",
    description="Hardware benchmarking and diagnostics suite.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class BenchmarkRequest(BaseModel):
    mode: str = Field("standard", description="quick | standard | full")
    include_gpu: bool = True
    include_scaling: bool = Field(
        False, description="Sweep thread counts. Adds a minute or more.")
    skip_validity_gate: bool = Field(
        False, description="Run even if conditions are bad. Result is marked invalid.")
    save_to_history: bool = True
    llm_report: bool = Field(
        False, description="Ask a model for a narrative. Needs ANTHROPIC_API_KEY.")


# --------------------------------------------------------------------------
# Basic endpoints
# --------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "BenchMind API Running",
        "version": "2.0.0",
        "dashboard": "/dashboard",
        "docs": "/docs",
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard assets not found.")
    return FileResponse(str(index))


@app.get("/api/system-info")
def system_info() -> Dict[str, Any]:
    info = get_system_info()
    fingerprint = get_environment_fingerprint()
    info["environment"] = fingerprint
    info["environment_summary"] = describe_fingerprint(fingerprint)
    return info


@app.get("/api/environment")
def environment() -> Dict[str, Any]:
    fp = get_environment_fingerprint(refresh=True)
    return {"fingerprint": fp, "summary": describe_fingerprint(fp)}


@app.get("/api/validity")
def validity(sample_seconds: float = Query(2.0, ge=0.2, le=10.0)) -> Dict[str, Any]:
    """Check whether right now is a good moment to benchmark."""
    return check_run_validity(sample_seconds=sample_seconds).to_dict()


@app.get("/api/telemetry/current")
def telemetry_current() -> Dict[str, Any]:
    service = TelemetryService.get_instance()
    service.start()
    snap = service.get_current()
    return {
        "snapshot": snap.to_dict() if snap else None,
        "health": service.sampling_health(),
    }


# --------------------------------------------------------------------------
# Benchmark execution
# --------------------------------------------------------------------------
def _execute_benchmark(req: BenchmarkRequest, progress=None) -> Dict[str, Any]:
    """
    The whole benchmark pipeline. Runs on a worker thread via JobManager.

    Order matters: validity is checked before anything heats the machine up,
    and telemetry is started before the warmup so the window covers everything.
    """
    def emit(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    emit("checking_conditions", 0.02)
    validity_report = check_run_validity(sample_seconds=2.0)
    if validity_report.verdict == "invalid" and not req.skip_validity_gate:
        logger.warning("Validity gate: %s", validity_report.verdict)

    telemetry = TelemetryService.get_instance()
    telemetry.start()

    emit("warmup", 0.05)
    warmup_cpu(seconds=3.0)

    start_mono = time.monotonic()

    emit("cpu", 0.10)

    def cpu_progress(stage: str, frac: float) -> None:
        emit(f"cpu:{stage}", 0.10 + frac * 0.55)

    cpu_result = run_full_cpu_suite(mode=req.mode, progress=cpu_progress)

    gpu_result: Optional[Dict[str, Any]] = None
    if req.include_gpu:
        emit("gpu", 0.68)
        gpu_result = run_gpu_suite()

    scaling_curve = None
    if req.include_scaling:
        emit("scaling_curve", 0.78)
        scaling_curve = run_scaling_curve(workload_key="integer", reps=3)

    end_mono = time.monotonic()
    emit("analysis", 0.90)

    timeline = telemetry.get_logs_format(
        start_time=start_mono, end_time=end_mono, use_monotonic=True)

    throttle = analyze_throttling(timeline)
    bottleneck = analyze_bottleneck(cpu_result.get("subtests", []))
    scaling_analysis = analyze_scaling(scaling_curve) if scaling_curve else None

    store = get_store()
    fingerprint_hash = (cpu_result.get("environment") or {}).get("fingerprint_hash")
    history_scores = [
        r["cpu_index"] for r in store.recent_runs(
            limit=5, fingerprint_hash=fingerprint_hash, mode=req.mode)
        if r.get("cpu_index")
    ]
    history_scores.append(cpu_result.get("cpu_index", 0))

    stability = build_stability_report(
        run_scores=history_scores,
        cpu_log=timeline.get("cpu", []),
        throttle_report=throttle,
    )

    summary = generate_summary(
        cpu_result,
        gpu_result=gpu_result,
        stability=stability.to_dict(),
        throttle=throttle,
        bottleneck=bottleneck,
        scaling=scaling_analysis,
        validity=validity_report.to_dict(),
    )

    result: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at": time.time(),
        "system": get_system_info(),
        "environment": cpu_result.get("environment"),
        "validity": validity_report.to_dict(),
        "cpu_benchmark": cpu_result,
        "gpu_benchmark": gpu_result,
        "scaling_curve": scaling_curve,
        "scaling_analysis": scaling_analysis,
        "throttling": throttle,
        "bottleneck": bottleneck,
        "stability": stability.to_dict(),
        "summary": summary,
        "timeline": timeline,
        "telemetry_health": telemetry.sampling_health(),
    }

    if req.llm_report:
        emit("llm_report", 0.95)
        try:
            from ai.llm_report import generate_llm_report
            narrative = generate_llm_report(result)
            if narrative:
                result["llm_report"] = narrative
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM report unavailable: %s", e)

    if req.save_to_history:
        emit("saving", 0.98)
        try:
            run_id = store.save_run(result)
            result["history_run_id"] = run_id
            result["regression"] = store.detect_regression(result, exclude_run_id=run_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Could not save run to history: %s", e, exc_info=True)
            result["history_error"] = str(e)

    emit("finished", 1.0)
    return result


@app.post("/api/benchmark")
def start_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    """Start a benchmark. Returns immediately with a job id."""
    if req.mode not in ("quick", "standard", "full"):
        raise HTTPException(status_code=400, detail="mode must be quick, standard or full")
    try:
        job = job_manager.submit(lambda progress: _execute_benchmark(req, progress),
                                 params=req.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    return {
        "job_id": job.id,
        "status": job.status,
        "poll": f"/api/jobs/{job.id}",
        "stream": f"/api/jobs/{job.id}/events",
    }


@app.get("/api/jobs")
def list_jobs() -> Dict[str, Any]:
    return {"jobs": job_manager.list_jobs(), "busy": job_manager.busy}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, include_result: bool = True) -> Dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job.to_dict(include_result=include_result)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Server-sent events stream of job progress. Ends when the job finishes."""
    if job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    async def event_stream():
        last_payload = None
        while True:
            job = job_manager.get(job_id)
            if job is None:
                break
            payload = json.dumps({
                "status": job.status,
                "stage": job.stage,
                "progress": round(job.progress, 3),
                "error": job.error,
            })
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job.status in ("finished", "failed", "cancelled"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# Results, history, comparison
# --------------------------------------------------------------------------
@app.get("/api/dashboard")
def dashboard_data() -> Dict[str, Any]:
    """Latest finished benchmark, or system info when nothing has run yet."""
    job = job_manager.latest_finished()
    if job is None or not job.result:
        return {"status": "no_benchmark_run", "system": get_system_info(),
                "busy": job_manager.busy}

    r = job.result
    return {
        "status": "finished",
        "job_id": job.id,
        "system": r.get("system"),
        "environment": r.get("environment"),
        "validity": r.get("validity"),
        "cpu": r.get("cpu_benchmark"),
        "gpu": r.get("gpu_benchmark"),
        "throttling": r.get("throttling"),
        "bottleneck": r.get("bottleneck"),
        "scaling_curve": r.get("scaling_curve"),
        "scaling_analysis": r.get("scaling_analysis"),
        "stability": r.get("stability"),
        "summary": r.get("summary"),
        "regression": r.get("regression"),
        "llm_report": r.get("llm_report"),
        "timeline": r.get("timeline"),
    }


@app.get("/api/history")
def history(limit: int = Query(20, ge=1, le=200),
            mode: Optional[str] = None,
            same_environment: bool = True) -> Dict[str, Any]:
    store = get_store()
    fingerprint = (get_environment_fingerprint().get("fingerprint_hash")
                   if same_environment else None)
    return {
        "runs": store.recent_runs(limit=limit, fingerprint_hash=fingerprint, mode=mode),
        "timeline": store.score_timeline(fingerprint_hash=fingerprint,
                                         mode=mode or "standard", limit=limit),
        "filtered_by_environment": same_environment,
        "fingerprint_hash": fingerprint,
    }


@app.get("/api/history/{run_id}")
def history_run(run_id: int) -> Dict[str, Any]:
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id")
    return run


@app.delete("/api/history/{run_id}")
def delete_history_run(run_id: int) -> Dict[str, Any]:
    return {"deleted": get_store().delete_run(run_id)}


# --------------------------------------------------------------------------
# Live telemetry stream
# --------------------------------------------------------------------------
@app.websocket("/ws/live-monitor")
async def live_monitor_socket(websocket: WebSocket):
    """
    Stream telemetry samples.

    1.x polled `get_current()` every 0.5 s while the sampler ran at 0.2 s, so
    it dropped roughly 60% of samples and re-sent duplicates whenever no new
    sample had arrived. This version tracks the last monotonic timestamp sent
    and emits only genuinely new snapshots.
    """
    await websocket.accept()
    service = TelemetryService.get_instance()
    service.start()
    logger.info("Live monitor connected.")

    last_sent = 0.0
    try:
        while True:
            snapshot = service.get_current()
            if snapshot is not None and snapshot.monotonic_time > last_sent:
                last_sent = snapshot.monotonic_time
                job = job_manager.active_job()
                await websocket.send_json({
                    "cpu": snapshot.cpu_utilization,
                    "ram": snapshot.ram_utilization,
                    "cpu_temp": snapshot.cpu_temp,
                    "gpu_temp": snapshot.gpu_temp,
                    "cpu_freq": snapshot.cpu_freq,
                    "per_core": snapshot.per_core_utilization,
                    "temp_source": snapshot.temp_source,
                    "timestamp": snapshot.timestamp,
                    "job": ({"id": job.id, "stage": job.stage,
                             "progress": round(job.progress, 3)} if job else None),
                })
            await asyncio.sleep(service.interval / 2)
    except WebSocketDisconnect:
        logger.info("Live monitor disconnected.")
    except Exception as e:  # noqa: BLE001
        logger.info("Live monitor closed: %s", repr(e))
