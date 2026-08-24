import psutil
import subprocess


def get_cpu_name():
    try:
        # Use PowerShell instead of WMIC (more reliable on Win11)
        command = 'powershell -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"'
        output = subprocess.check_output(command, shell=True)

        cpu_name = output.decode(errors="ignore").strip()

        if cpu_name:
            return cpu_name

    except Exception:
        pass

    return "Unknown CPU"


def get_system_info():
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    return {
        "cpu": get_cpu_name(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram": round(psutil.virtual_memory().total / (1024**3), 2),
    }