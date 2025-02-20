import traceback

from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.QtMainWindow import MainWindow
from coralnet_toolbox.utilities import console_user

from coralnet_toolbox import __version__

# ----------------------------------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------------------------------


def run():
    try:
        app = QApplication([])
        app.setStyle('Fusion')
        main_window = MainWindow(__version__)
        main_window.show()
        app.exec_()

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    run()
