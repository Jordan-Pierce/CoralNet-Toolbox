import sys
import traceback

import qdarktheme
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton, QTextEdit

from coralnet_toolbox.QtMainWindow import MainWindow

from coralnet_toolbox.utilities import console_user
from coralnet_toolbox.utilities import except_hook

from coralnet_toolbox import __version__


# ----------------------------------------------------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------------------------------------------------
    

def run():
    main_window = None
    app = None
    
    try:
        # Install the exception hook (initial setup without main_window)
        sys.excepthook = except_hook
        app = QApplication(sys.argv)
        qdarktheme.setup_theme("light")
        main_window = MainWindow(__version__)
        
        # Update excepthook with main_window reference using a lambda
        sys.excepthook = lambda cls, exception, traceback_obj: except_hook(
            cls, exception, traceback_obj, main_window)
        
        main_window.show()
        sys.exit(app.exec_())
        
    # Rest of the function remains unchanged
    except Exception as e:
        # Log the full traceback
        error_message = f"{e}\n{traceback.format_exc()}"
        
        # Print to console first
        console_user(error_message)
        
        # Ensure application exits
        if app is not None:
            app.quit()
            
        # Exit with error code
        sys.exit(1)


if __name__ == '__main__':
    run()
