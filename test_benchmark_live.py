import sys
import threading
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import psutil

from monitoring.live_monitor import TimelineCollector
from benchmarks.cpu_test import run_cpu_test
from ai.stability_engine import calculate_stability


app = QApplication(sys.argv)

# ===== WINDOW =====
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("BenchMind Live Benchmark")

plot = win.addPlot(title="CPU Usage During Benchmark")
curve = plot.plot()

graph_data = []

collector = TimelineCollector(interval=0.2)


def update_graph():
    cpu = psutil.cpu_percent()
    graph_data.append(cpu)

    if len(graph_data) > 100:
        graph_data.pop(0)

    curve.setData(graph_data)


timer = QTimer()
timer.timeout.connect(update_graph)
timer.start(200)


def benchmark_worker():
    print("Starting timeline collector...")
    collector.start()

    result = run_cpu_test()

    collector.stop()

    logs = collector.get_logs()
    stability = calculate_stability(logs["cpu"])

    print("\n===== RESULTS =====")
    print(result)
    print("Stability Score:", stability)


def start_benchmark():
    thread = threading.Thread(target=benchmark_worker)
    thread.start()


# start benchmark after UI loads
QTimer.singleShot(1000, start_benchmark)

sys.exit(app.exec_())