from db.client import supabase

def save_system_profile(system_info: dict) -> str:
    gpus = system_info.get("gpus", [])
    primary_gpu = gpus[0] if len(gpus) > 0 else "N/A"
    secondary_gpu = gpus[1] if len(gpus) > 1 else None

    response = supabase.table("systems").insert({
        "cpu_name": system_info.get("cpu"),
        "physical_cores": system_info.get("physical_cores"),
        "logical_cores": system_info.get("logical_cores"),
        "ram_gb": system_info.get("ram"),
        "primary_gpu": primary_gpu,
        "secondary_gpu": secondary_gpu,
        "os_info": system_info.get("os")
    }).execute()

    return response.data[0]["id"]

def save_benchmark_result(system_id: str, results: dict) -> str:
    response = supabase.table("benchmark_runs").insert({
        "system_id": system_id,
        "cpu_single_score": results.get("cpu_single_score"),
        "cpu_multi_score": results.get("cpu_multi_score"),
        "gpu_score": results.get("gpu_score"),
        "stability_score": results.get("stability"),
        "summary_text": results.get("summary"),
        "status": "finished"
    }).execute()

    return response.data[0]["id"]

def save_timeline_logs(benchmark_id: str, timeline_records: list):
    records = []
    for record in timeline_records:
        records.append({
            "benchmark_id": benchmark_id,
            "time_offset": record.get("time"),
            "cpu_usage": record.get("cpu"),
            "ram_usage": record.get("ram"),
            "cpu_temp": record.get("cpu_temp"),
            "gpu_temp": record.get("gpu_temp")
        })

    if records:
        supabase.table("benchmark_logs").insert(records).execute()

def get_recent_benchmarks(limit: int = 10):
    response = supabase.table("benchmark_runs")\
        .select("id, cpu_single_score, cpu_multi_score, gpu_score, stability_score, created_at, systems(cpu_name, primary_gpu)")\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
    return response.data
