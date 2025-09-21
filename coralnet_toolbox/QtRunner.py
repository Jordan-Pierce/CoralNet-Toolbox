import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5.QtWidgets import QProgressDialog, QApplication, QVBoxLayout, QLabel, QDialog, QPushButton


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TaskSignals(QObject):
    """Signals available from a running task."""
    started = pyqtSignal()
    finished = pyqtSignal(object)  # object: return value
    error = pyqtSignal(tuple)      # tuple: (exctype, value, traceback.format_exc())
    progress = pyqtSignal(int)     # int: progress percentage (0-100)
    message = pyqtSignal(str)      # str: status message


class TaskRunner(QRunnable):
    """
    A generic worker that runs any function with arguments in a thread.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        
        # Add progress callback if the function accepts it
        if hasattr(self.fn, '__code__') and 'progress_callback' in self.fn.__code__.co_varnames:
            self.kwargs['progress_callback'] = self.signals.progress.emit
        if hasattr(self.fn, '__code__') and 'message_callback' in self.fn.__code__.co_varnames:
            self.kwargs['message_callback'] = self.signals.message.emit

    @pyqtSlot()
    def run(self):
        """Execute the function in a thread."""
        self.signals.started.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            exctype, value, tb = sys.exc_info()
            self.signals.error.emit((exctype, value, traceback.format_exc()))


class GlobalTaskRunner:
    """
    A simple class to run any function in a thread with a basic progress dialog.
    """
    @staticmethod
    def run_task(parent, task_func, *args, title="Processing...", **kwargs):
        """
        Run a function in a thread with a progress dialog.
        
        Args:
            parent: The parent widget
            task_func: The function to execute
            *args: Arguments to pass to the function
            title: Title for the progress dialog
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function, or None if canceled or failed
        """
        # Create a simple progress dialog
        dialog = QProgressDialog(title, "Cancel", 0, 100, parent)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(True)
        dialog.setAutoReset(False)
        dialog.setAutoClose(False)
        dialog.show()
        
        # Create and configure the worker
        worker = TaskRunner(task_func, *args, **kwargs)
        result = [None]  # Use a list to store result from signal handler
        
        # Connect signals
        worker.signals.progress.connect(dialog.setValue)
        worker.signals.message.connect(dialog.setLabelText)
        
        def on_finished(task_result):
            result[0] = task_result
            dialog.setValue(100)
            dialog.accept()
            
        def on_error(exc_info):
            exctype, value, tb_str = exc_info
            print(f"Task error: {exctype.__name__}: {value}")
            dialog.reject()
            
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        
        # Start the task
        QThreadPool.globalInstance().start(worker)
        
        # Execute the dialog modally
        dialog.exec_()
        
        return result[0]