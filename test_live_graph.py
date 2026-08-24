import sys
import psutil
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from PyQt5.QtCore import QTimer


app = QApplication(sys.argv)

# window
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("BenchMind Live CPU Graph")

plot = win.addPlot(title="CPU Usage (%)")
curve = plot.plot()

data = []


def update():
    cpu = psutil.cpu_percent()
    data.append(cpu)

    if len(data) > 50:
        data.pop(0)

    curve.setData(data)


timer = QTimer()
timer.timeout.connect(update)
timer.start(200)  # update every 200ms

sys.exit(app.exec_())