import warnings

from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox, QLineEdit

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class GlobalEventFilter(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.classify_deploy_model_dialog = main_window.classify_deploy_model_dialog
        self.detect_deploy_model_dialog = main_window.detect_deploy_model_dialog
        self.segment_deploy_model_dialog = main_window.segment_deploy_model_dialog
        self.semantic_deploy_model_dialog = main_window.semantic_deploy_model_dialog
        self.sam_deploy_generator_dialog = main_window.sam_deploy_generator_dialog
        self.see_anything_deploy_generator_dialog = main_window.see_anything_deploy_generator_dialog
        self.transformers_deploy_model_dialog = main_window.transformers_deploy_model_dialog
        
    def eventFilter(self, obj, event):
        # Check for explorer window first - this applies to all event types
        if hasattr(self.main_window, 'explorer_window') and self.main_window.explorer_window:
            # Special exception for WASD keys which should always work
            if event.type() == QEvent.KeyPress and event.key() in [Qt.Key_Up, Qt.Key_Down] and \
                event.modifiers() & Qt.ControlModifier:
                # Handle Ctrl+Up and Ctrl+Down for cycling labels
                if event.key() == Qt.Key_Up:
                    self.label_window.cycle_labels(-1)  # Cycle up/previous
                    return True
                if event.key() == Qt.Key_Down:
                    self.label_window.cycle_labels(1)  # Cycle down/next
                    return True
            
            # For all other events when explorer is visible, pass them through
            return False
            
        # Now handle keyboard events
        if event.type() == QEvent.KeyPress:
            if event.modifiers() & Qt.ControlModifier and not (event.modifiers() & Qt.ShiftModifier):
                
                # Handle Ctrl+Up and Ctrl+Down for cycling labels
                if event.key() == Qt.Key_Up:
                    self.label_window.cycle_labels(-1)  # Cycle up/previous
                    return True
                if event.key() == Qt.Key_Down:
                    self.label_window.cycle_labels(1)  # Cycle down/next
                    return True

                # Handle Alt key for switching between Select and Annotation tools
                if event.key() == Qt.Key_Alt:
                    self.main_window.switch_back_to_tool()
                    return True
                
                # Handle hotkey for image classification prediction
                if event.key() == Qt.Key_1:
                    if self.classify_deploy_model_dialog.loaded_model is not None:
                        self.classify_deploy_model_dialog.predict()
                    else:
                        self.main_window.open_classify_deploy_model_dialog()
                    return True

                # Handle hotkey for object detection prediction
                if event.key() == Qt.Key_2:
                    if self.detect_deploy_model_dialog.loaded_model is not None:
                        self.detect_deploy_model_dialog.predict()
                    else:
                        self.main_window.open_detect_deploy_model_dialog()
                    return True

                # Handle hotkey for instance segmentation prediction
                if event.key() == Qt.Key_3:
                    if self.segment_deploy_model_dialog.loaded_model is not None:
                        self.segment_deploy_model_dialog.predict()
                    else:
                        self.main_window.open_segment_deploy_model_dialog()
                    return True
                
                # Handle hotkey for semantic segmentation prediction
                if event.key() == Qt.Key_4:
                    if self.semantic_deploy_model_dialog.loaded_model is not None:
                        self.semantic_deploy_model_dialog.predict()
                    else:
                        self.main_window.open_semantic_deploy_model_dialog()
                    return True

                # Handle hotkey for segment everything prediction
                if event.key() == Qt.Key_5:
                    if self.sam_deploy_generator_dialog.loaded_model is not None:
                        self.sam_deploy_generator_dialog.predict()
                    else:
                        self.main_window.open_sam_deploy_generator_dialog()
                    return True
                
                # Handle hotkey for see anything (YOLOE) generator
                if event.key() == Qt.Key_6:
                    if self.see_anything_deploy_generator_dialog.loaded_model is not None:
                        self.see_anything_deploy_generator_dialog.predict()
                    else:
                        self.main_window.open_see_anything_deploy_generator_dialog()
                    return True

                # Handle hotkey for transformers prediction
                if event.key() == Qt.Key_7:
                    if self.transformers_deploy_model_dialog.loaded_model is not None:
                        self.transformers_deploy_model_dialog.predict()
                    else:
                        self.main_window.open_transformers_deploy_model_dialog()
                    return True

                # Handle annotation cycling hotkeys
                if event.key() == Qt.Key_Left:
                    self.annotation_window.cycle_annotations(-1)
                    return True
                if event.key() == Qt.Key_Right:
                    self.annotation_window.cycle_annotations(1)
                    return True

                # Handle undo/redo hotkeys globally to avoid focus issues
                if event.key() == Qt.Key_Z:
                    if self.annotation_window.selected_tool:
                        self.annotation_window.action_stack.undo()
                        return True
                
            # Handle redo hotkey (Ctrl+Shift+Z)
            if event.key() == Qt.Key_Z and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
                if self.annotation_window.selected_tool:
                    self.annotation_window.action_stack.redo()
                    return True
            
            # Delete (backspace or delete key) selected annotations when select tool is active
            if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
                # Check if a text input field has focus first
                if isinstance(QApplication.focusWidget(), QLineEdit):
                    return False  # Pass the event on to the QLineEdit

                # First check if the select tool is active
                if self.main_window.select_tool_action.isChecked():
                    selected_tool = self.annotation_window.selected_tool
                    select_tool = self.annotation_window.tools[selected_tool]
                    # Get the active subtool if it exists, pass to its keyPressEvent
                    if hasattr(select_tool, 'active_subtool') and select_tool.active_subtool:
                        select_tool.active_subtool.keyPressEvent(event)
                        return True
                            
                    # Otherwise, proceed with deletion if there are selected annotations
                    if self.annotation_window.selected_annotations:
                        self.annotation_window.delete_selected_annotations()
                        return True
                    
                    # Consume the event so it doesn't do anything else
                    return True
            
            # Handle image cycling hotkeys
            if event.key() == Qt.Key_Up and event.modifiers() == (Qt.AltModifier):
                self.image_window.cycle_previous_image()
                return True
            if event.key() == Qt.Key_Down and event.modifiers() == (Qt.AltModifier):
                self.image_window.cycle_next_image()
                return True

            # Handle Ctrl + S for saving project
            if event.key() == Qt.Key_S and event.modifiers() == (Qt.ControlModifier):
                self.main_window.save_project_as()
                return True

            # Handle Escape key for exiting program
            if event.key() == Qt.Key_Escape:
                self.show_exit_confirmation_dialog()
                return True

        # Return False for other events to allow them to be processed by the target object
        return False
    
    def show_exit_confirmation_dialog(self):
        """Show a confirmation dialog when the user attempts to exit the application."""
        # Create the message box
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Confirm Exit')
        msg_box.setText('Are you sure you want to exit?')
        msg_box.setIcon(QMessageBox.Question)
        
        # Set the custom icon
        msg_box.setWindowIcon(get_icon('coralnet.png'))
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        reply = msg_box.exec_()
        if reply == QMessageBox.Yes:
            QApplication.quit()
