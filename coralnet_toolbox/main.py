# TODO
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - Short or Long code for ML?
#   - if import labels are too long, add "..."
#   - don't crash when importing the same image twice
#   - Clean up QImage, Rasterio Image, PixMap
#   - Update labels found with model deploy

from coralnet_toolbox.QtMain import MainWindow
from PyQt5.QtWidgets import QApplication


def run():
    app = QApplication([])
    app.setStyle('WindowsXP')
    main_window = MainWindow()
    main_window.show()
    app.exec_()


if __name__ == '__main__':
    run()