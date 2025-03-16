import sys
import traceback

import qdarktheme
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton

from coralnet_toolbox.QtMainWindow import MainWindow
from coralnet_toolbox.utilities import console_user

from coralnet_toolbox import __version__


# ----------------------------------------------------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------------------------------------------------


def execept_hook(cls, exception, traceback_obj):
    """Handle uncaught exceptions including Qt errors"""
    error_msg = f"{cls.__name__}: {exception}\n\n"
    error_msg += ''.join(traceback.format_tb(traceback_obj))
    
    # Log the error
    print(error_msg)
    
    # If Qt is initialized, show error in GUI
    if QApplication.instance() is not None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("An unexpected error occurred")
        msg_box.setDetailedText(error_msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    sys.__excepthook__(cls, exception, traceback_obj)
    sys.exit(1)    
    

def run():
    main_window = None
    app = None
    
    try:
        # Install the exception hook
        sys.excepthook = execept_hook
        app = QApplication(sys.argv)
        qdarktheme.setup_theme("light")
        main_window = MainWindow(__version__)
        main_window.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        # Log the full traceback
        error_message = f"{e}\n{traceback.format_exc()}"
        console_user(error_message)
        
        # Show a detailed error message box
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Unexpected Error")
        error_dialog.setText("An unexpected error has occurred.")
        error_dialog.setDetailedText(error_message)
        
        # Only attempt to use main_window if it exists and is a valid object
        if main_window is not None:
            save_button = QPushButton("Save Project")
            error_dialog.addButton(save_button, QMessageBox.AcceptRole)
            
            # Show the error dialog
            result = error_dialog.exec_()
            
            # Check if save button was clicked and main_window method exists
            if (error_dialog.clickedButton() == save_button and hasattr(main_window, 'open_save_project_dialog')):
                try:
                    main_window.open_save_project_dialog()
                except Exception as save_error:
                    QMessageBox.warning(None,
                                        "Save Error", 
                                        f"Could not save project: {save_error}")
        else:
            # Fallback error dialog if main_window is not available
            QMessageBox.critical(None, "Critical Error", error_message)
        
        # Ensure application exits
        if app is not None:
            app.quit()
            
        # Exit with error code
        sys.exit(1)


if __name__ == '__main__':
    run()
