from monitoring.live_monitor import TimelineCollector
import time

collector = TimelineCollector(interval=1)

print("Starting collector...")
collector.start()

time.sleep(5)

collector.stop()

print("Logs:")
print(collector.get_logs())