from fastapi import FastAPI, WebSocket
import threading
import asyncio

from benchmarks.cpu_test import run_cpu_test, warmup_cpu
from benchmarks.gpu_test import run_gpu_test
from monitoring.live_monitor import TimelineCollector
from monitoring.system_info import get_system_info
from ai.stability_engine import calculate_stability
from ai.summary_engine import generate_summary


app = FastAPI()


# ===== GLOBAL BENCHMARK STATE =====
benchmark_state = {
    "status": "idle",
    "result": None
}

benchmark_lock = threading.Lock()


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

    except:
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

    collector = TimelineCollector(interval=0.1)

    warmup_cpu(seconds=2)

    benchmark_state["status"] = "running"

    collector.start()

    cpu_result = run_cpu_test()
    gpu_result = run_gpu_test()

    collector.stop()

    benchmark_state["status"] = "analyzing"

    logs = collector.get_logs()
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

    collector = TimelineCollector(interval=0.5)
    collector.start()

    print("WebSocket monitor started.")

    try:
        while True:

            logs = collector.get_logs()

            data = {
                "cpu": logs["cpu"][-1] if logs["cpu"] else None,
                "ram": logs["ram"][-1] if logs["ram"] else None,
                "cpu_temp": logs["cpu_temp"][-1] if logs["cpu_temp"] else None,
                "gpu_temp": logs["gpu_temp"][-1] if logs["gpu_temp"] else None,
            }

            print("WS DATA:", data)

            await websocket.send_json(data)

            await asyncio.sleep(0.5)

    except Exception as e:
        print("WebSocket error:", repr(e))

    finally:
        collector.stop()
        print("WebSocket monitor stopped.")