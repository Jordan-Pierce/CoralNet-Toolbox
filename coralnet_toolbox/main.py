import os
import sys
import traceback
import warnings

from PyQt5.QtCore import Qt, QCoreApplication, QTimer
from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.QtMainWindow import MainWindow
from coralnet_toolbox.QtStartupNotices import show_startup_notices
from coralnet_toolbox.theme import apply_theme

from coralnet_toolbox.utilities import configure_gdal
from coralnet_toolbox.utilities import configure_file_limit
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
        # Third-party warnings are not actionable for someone annotating images,
        # so the application runs quiet. This is set here, for the application
        # only, and not at import time: modules are also imported by tests and
        # scripts, and a filter installed on import overrides -W. Run with
        # `python -W default` or PYTHONWARNINGS=default to see everything.
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            # Spawned workers (DataLoader on Windows and macOS) import this
            # module but never call run(); they read the environment instead
            os.environ["PYTHONWARNINGS"] = "ignore"

        # Install the exception hook (initial setup without main_window)
        sys.excepthook = except_hook

        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Before any raster is opened: GDAL reads this from the environment.
        configure_gdal()

        # Also before any raster is opened: each one holds a file descriptor for
        # its lifetime, and the POSIX default of 1024 is well under the number of
        # images a project routinely contains.
        configure_file_limit()

        app = QApplication(sys.argv)

        apply_theme(app)
        
        main_window = MainWindow(__version__)
        
        # Update excepthook with main_window reference using a lambda
        sys.excepthook = lambda cls, exception, traceback_obj: except_hook(
            cls, exception, traceback_obj, main_window)
        
        main_window.show()

        # Once the event loop is running, so the window is up behind any dialog
        QTimer.singleShot(0, lambda: show_startup_notices(main_window))

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
