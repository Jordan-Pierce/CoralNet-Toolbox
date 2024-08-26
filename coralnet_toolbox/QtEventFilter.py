from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QObject, QEvent

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class GlobalEventFilter(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.deploy_model_dialog
        self.image_window = main_window.image_window

    def eventFilter(self, obj, event):
        # Check if the event is a wheel event
        if event.type() == QEvent.Wheel:
            # Handle Zoom wheel for setting annotation size
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.annotation_window.set_annotation_size(delta=16)  # Zoom in
                else:
                    self.annotation_window.set_annotation_size(delta=-16)  # Zoom out
                return True

        elif event.type() == QEvent.KeyPress:
            if event.modifiers() & Qt.ControlModifier:

                # Handle WASD keys for selecting Label
                if event.key() in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]:
                    self.label_window.handle_wasd_key(event.key())
                    return True

                # Handle hotkey for prediction
                if event.key() == Qt.Key_Z:
                    self.deploy_model_dialog.predict()
                    return True

                # Handle annotation cycling hotkeys
                if event.key() == Qt.Key_Left:
                    self.annotation_window.cycle_annotations(-1)
                    return True
                if event.key() == Qt.Key_Right:
                    self.annotation_window.cycle_annotations(1)
                    return True

                # Handle thumbnail cycling hotkeys
                if event.key() == Qt.Key_Up:
                    self.image_window.cycle_previous_image()
                    return True
                if event.key() == Qt.Key_Down:
                    self.image_window.cycle_next_image()
                    return True

                # Handle Delete key for Deleting selected annotations or images
                if event.key() == Qt.Key_Delete:

                    if self.annotation_window.selected_annotation:
                        self.annotation_window.delete_selected_annotation()
                        return True

                    if self.image_window.selected_image_row:
                        self.image_window.delete_selected_image()
                        return True

            # Handle Escape key for exiting program
            if event.key() == Qt.Key_Escape:
                self.show_exit_confirmation_dialog()
                return True

        # Return False for other events to allow them to be processed by the target object
        return False

    def show_exit_confirmation_dialog(self):
        # noinspection PyTypeChecker
        reply = QMessageBox.question(None,
                                     'Confirm Exit',
                                     'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()