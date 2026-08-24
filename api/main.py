import asyncio
import logging
import time
import threading
from fastapi import FastAPI, WebSocket

from benchmarks.cpu_test import run_cpu_test, warmup_cpu
from benchmarks.gpu_test import run_gpu_test
from monitoring.telemetry_service import TelemetryService
from monitoring.system_info import get_system_info
from ai.stability_engine import calculate_stability
from ai.summary_engine import generate_summary

logger = logging.getLogger("BenchMind.API")

app = FastAPI()

# ===== GLOBAL BENCHMARK STATE =====
benchmark_state = {
    "status": "idle",
    "result": None
}

benchmark_lock = threading.Lock()


@app.on_event("startup")
def startup_event():
    logger.info("Starting up BenchMind API and TelemetryService...")
    TelemetryService.get_instance().start()


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down BenchMind API and TelemetryService...")
    TelemetryService.get_instance().stop()


# ===== ROOT =====
@app.get("/")
def root():
    return {"message": "BenchMind API Running 🚀"}


# ===== SYSTEM INFO =====
@app.get("/api/system-info")
def system_info():
    info = get_system_info()
    gpu_names = []

    try:
        import pyopencl as cl
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                name = device.name.strip()
                if "RaptorLake" in name:
                    name = "Intel UHD Graphics"
                gpu_names.append(name)
    except Exception as e:
        logger.debug("Failed to query PyOpenCL devices: %s", e)
        gpu_names = []

    info["gpu_count"] = len(gpu_names)
    info["gpus"] = gpu_names
    return info


# ===== BENCHMARK STATUS =====
@app.get("/api/benchmark-status")
def benchmark_status():
    return benchmark_state


# ===== FULL BENCHMARK =====
@app.post("/api/full-benchmark")
def full_benchmark():
    with benchmark_lock:
        if benchmark_state["status"] in ["warming_up", "running"]:
            return {"message": "Benchmark already running"}

        benchmark_state["status"] = "warming_up"
        benchmark_state["result"] = None

    telemetry_service = TelemetryService.get_instance()
    telemetry_service.start()

    warmup_cpu(seconds=2)

    benchmark_state["status"] = "running"
    start_time = time.monotonic()

    cpu_result = run_cpu_test()
    gpu_result = run_gpu_test()

    end_time = time.monotonic()

    benchmark_state["status"] = "analyzing"

    logs = telemetry_service.get_logs_format(
        start_time=start_time,
        end_time=end_time,
        use_monotonic=True
    )
    stability = calculate_stability(logs["cpu"])

    summary = generate_summary(
        cpu_result,
        gpu_result,
        stability
    )

    final_result = {
        "system": system_info(),
        "cpu_benchmark": cpu_result,
        "gpu_benchmark": gpu_result,
        "stability": stability,
        "summary": summary,
        "timeline": logs,
    }

    benchmark_state["status"] = "finished"
    benchmark_state["result"] = final_result

    return final_result


# ===== DASHBOARD =====
@app.get("/api/dashboard")
def dashboard():
    if benchmark_state["result"] is None:
        return {
            "status": "no_benchmark_run",
            "system": system_info()
        }

    result = benchmark_state["result"]
    return {
        "status": benchmark_state["status"],
        "system": result["system"],
        "cpu": result["cpu_benchmark"],
        "gpu": result["gpu_benchmark"],
        "stability": result["stability"],
        "summary": result["summary"]
    }


# ===== REALTIME MONITOR STREAM =====
@app.websocket("/ws/live-monitor")
async def live_monitor_socket(websocket: WebSocket):
    await websocket.accept()

    telemetry_service = TelemetryService.get_instance()
    telemetry_service.start()

    logger.info("WebSocket monitor connected.")

    try:
        while True:
            snapshot = telemetry_service.get_current()
            if snapshot is None:
                await asyncio.sleep(0.1)
                continue

            data = {
                "cpu": snapshot.cpu_utilization,
                "ram": snapshot.ram_utilization,
                "cpu_temp": snapshot.cpu_temp,
                "gpu_temp": snapshot.gpu_temp,
                "timestamp": snapshot.timestamp,
            }

            await websocket.send_json(data)
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.info("WebSocket monitor closed: %s", repr(e))
    finally:
        logger.info("WebSocket monitor disconnected.")