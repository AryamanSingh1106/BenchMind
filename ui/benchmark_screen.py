import sys
import threading
import psutil
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
)

from PyQt5.QtCore import QTimer

from monitoring.live_monitor import TimelineCollector
from benchmarks.cpu_test import run_cpu_test
from ai.stability_engine import calculate_stability


class BenchmarkScreen(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("BenchMind - Benchmark Screen")
        self.resize(800, 600)

        layout = QVBoxLayout()

        # ===== RUN BUTTON =====
        self.run_button = QPushButton("Run CPU Benchmark")
        self.run_button.clicked.connect(self.start_benchmark)
        layout.addWidget(self.run_button)

        # ===== RESULT LABELS =====
        self.result_label = QLabel("CPU Score: --")
        self.stability_label = QLabel("Stability Score: --")
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        layout.addWidget(self.result_label)
        layout.addWidget(self.stability_label)

        # ===== GRAPH =====
        self.graph_widget = pg.PlotWidget(title="CPU Usage Live")
        self.curve = self.graph_widget.plot()
        layout.addWidget(self.graph_widget)

        self.setLayout(layout)

        # ===== DATA =====
        self.graph_data = []

        self.collector = TimelineCollector(interval=0.2)

        # timer for live graph
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(200)

    def update_graph(self):
        cpu = psutil.cpu_percent()

        self.graph_data.append(cpu)

        if len(self.graph_data) > 100:
            self.graph_data.pop(0)

        self.curve.setData(self.graph_data)

    def benchmark_worker(self):

        self.collector.start()

        result = run_cpu_test()

        self.collector.stop()

        self.status_label.setText("Status: Analyzing Stability...")

        logs = self.collector.get_logs()
        stability = calculate_stability(logs["cpu"])

        self.result_label.setText(
            f"CPU Score: {result['cpu_score']}"
         )

        self.stability_label.setText(
            f"Stability Score: {stability}%"
         )

        self.status_label.setText("Status: Finished ✔")

        self.run_button.setText("Run CPU Benchmark")
        self.run_button.setEnabled(True)

        self.run_button.setEnabled(True)

    def start_benchmark(self):

        # reset UI
        self.graph_data.clear()
        self.curve.setData([])

        self.result_label.setText("CPU Score: --")
        self.stability_label.setText("Stability Score: --")
        self.status_label.setText("Status: Running Benchmark...")

        self.run_button.setText("Running...")
        self.run_button.setEnabled(False)

        thread = threading.Thread(target=self.benchmark_worker)
        thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BenchmarkScreen()
    window.show()
    sys.exit(app.exec_())