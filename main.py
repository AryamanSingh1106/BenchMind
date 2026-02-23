from PyQt5.QtWidgets import QApplication, QLabel
import sys

app = QApplication(sys.argv)

label = QLabel("BenchMind Running 🚀")
label.show()

sys.exit(app.exec_())