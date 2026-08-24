import psutil
import time
import threading

from monitoring.temp_reader import get_temperatures


class TimelineCollector:

    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False
        self.thread = None

        # timeline logs
        self.cpu_log = []
        self.ram_log = []
        self.cpu_temp_log = []
        self.gpu_temp_log = []
        self.timestamp_log = []

    # ===== MAIN LOOP =====
    def collect_data(self):

        while self.running:

            # CPU + RAM
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            # temps from LibreHardwareMonitor
            temps = get_temperatures()

            cpu_temp = temps.get("cpu_temp")
            gpu_temp = temps.get("gpu_temp")

            # store logs
            self.cpu_log.append(cpu)
            self.ram_log.append(ram)
            self.cpu_temp_log.append(cpu_temp)
            self.gpu_temp_log.append(gpu_temp)
            self.timestamp_log.append(time.time())

            time.sleep(self.interval)

    # ===== START =====
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self.collect_data,
                daemon=True
            )
            self.thread.start()

    # ===== STOP =====
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    # ===== GET LOGS =====
    def get_logs(self):
        return {
            "cpu": self.cpu_log,
            "ram": self.ram_log,
            "cpu_temp": self.cpu_temp_log,
            "gpu_temp": self.gpu_temp_log,
            "time": self.timestamp_log,
        }