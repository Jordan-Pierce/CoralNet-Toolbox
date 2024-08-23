# TODO
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - progress bar for exporting annotations JSON
#   - Deploy model after training?
#   - Short or Long code for ML?
#   - if import labels are too long, add "..."
#   - filter images that contain annotations
#   - show a distribution of data in the create dataset
#   - allow only training dataset

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