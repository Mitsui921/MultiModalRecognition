
import os
import warnings

from SignRecognition import Sign_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')
    app = QApplication(argv)

    window = QMainWindow()
    ui = Sign_MainWindow(window)

    window.show()
    exit(app.exec_())
