import traceback

from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.QtMainWindow import MainWindow
from coralnet_toolbox.utilities import console_user
from coralnet_toolbox import get_version


# ----------------------------------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------------------------------


def run():
    try:
        app = QApplication([])
        app.setStyle('WindowsXP')
        main_window = MainWindow(version=get_version())
        main_window.show()
        app.exec_()

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    run()
