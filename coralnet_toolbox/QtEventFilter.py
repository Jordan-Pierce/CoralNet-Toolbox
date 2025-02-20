import warnings

from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class GlobalEventFilter(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.classify_deploy_model_dialog = main_window.classify_deploy_model_dialog
        self.detect_deploy_model_dialog = main_window.detect_deploy_model_dialog
        self.segment_deploy_model_dialog = main_window.segment_deploy_model_dialog
        self.sam_deploy_generator_dialog = main_window.sam_deploy_generator_dialog
        self.auto_distill_deploy_model_dialog = main_window.auto_distill_deploy_model_dialog
        
        self.tile_inference_dialog = main_window.tile_inference_dialog

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.modifiers() & Qt.ControlModifier and not (event.modifiers() & Qt.ShiftModifier):

                # Handle WASD keys for selecting Label
                if event.key() in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]:
                    self.label_window.handle_wasd_key(event.key())
                    return True
                
                # Handle G key for displaying existing tile inference parameters
                if event.key() == Qt.Key_G:
                    if self.main_window.tile_inference_tool_action.isChecked():
                        self.tile_inference_dialog.update_tile_graphics()
                    return True
                
                # Handle hotkey for image classification prediction
                if event.key() == Qt.Key_1:
                    self.classify_deploy_model_dialog.predict()
                    return True

                # Handle hotkey for object detection prediction
                if event.key() == Qt.Key_2:
                    self.detect_deploy_model_dialog.predict()
                    return True

                # Handle hotkey for instance segmentation prediction
                if event.key() == Qt.Key_3:
                    self.segment_deploy_model_dialog.predict()
                    return True

                # Handle hotkey for segment everything prediction
                if event.key() == Qt.Key_4:
                    self.sam_deploy_generator_dialog.predict()
                    return True

                # Handle hotkey for auto distill prediction
                if event.key() == Qt.Key_5:
                    self.auto_distill_deploy_model_dialog.predict()
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

            # Handle Ctrl + Shift + S for saving project
            if event.key() == Qt.Key_S and event.modifiers() == (Qt.ShiftModifier | Qt.ControlModifier):
                self.main_window.save_project_as()
                return True

            # Select all annotations on < key press with Shift+Ctrl
            if event.key() == Qt.Key_Less and event.modifiers() == (Qt.ShiftModifier | Qt.ControlModifier):
                if self.main_window.select_tool_action.isChecked():
                    self.annotation_window.select_annotations()
                    return True

            # Unselect all annotations on > key press with Shift+Ctrl
            if event.key() == Qt.Key_Greater and event.modifiers() == (Qt.ShiftModifier | Qt.ControlModifier):
                if self.main_window.select_tool_action.isChecked():
                    self.annotation_window.select_annotations()
                    return True

            # Handle Escape key for exiting program
            if event.key() == Qt.Key_Escape:
                self.show_exit_confirmation_dialog()
                return True
            
        # Handle key release events 
        elif event.type() == QEvent.KeyRelease:
            
            # Handle G key for removing existing tile inference parameter
            if event.key() == Qt.Key_G:
                self.tile_inference_dialog.clear_tile_graphics()
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
