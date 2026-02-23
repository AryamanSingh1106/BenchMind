import psutil
import time

while True:
    print("CPU:", psutil.cpu_percent())
    time.sleep(1)