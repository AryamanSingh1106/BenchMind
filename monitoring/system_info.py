import platform
import psutil

def get_system_info():
    return {
        "cpu": platform.processor(),
        "cores": psutil.cpu_count(),
        "ram": round(psutil.virtual_memory().total / (1024**3), 2)
    }

print(get_system_info())