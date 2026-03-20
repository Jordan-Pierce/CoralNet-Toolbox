"""
QtLayoutManager - Persistent window layout management using PyQtAds.

Automatically saves dock configuration on application exit and restores
on startup using PyQtAds' serialization (CDockManager.saveState/restoreState).

Configuration files stored in .cache/layout/ directory.
"""

import os
import json
import base64
from pathlib import Path

from PyQt5.QtCore import QByteArray
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SaveLayoutDialog(QDialog):
    """Simple dialog for saving a layout configuration with a custom name."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Layout")
        self.setModal(True)
        self.layout_name = None
        self.setup_ui()
    
    def setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("Enter a name for this layout:")
        layout.addWidget(label)
        
        # Input field
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., 'annotation-focus' or 'analysis'")
        self.name_input.returnPressed.connect(self.accept)
        layout.addWidget(self.name_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.on_save)
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(250, 80)
    
    def on_save(self):
        """Validate and save layout name."""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a layout name.")
            return
        
        # Sanitize the name (alphanumeric, hyphens, underscores only)
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        if not sanitized:
            QMessageBox.warning(self, "Invalid Name", "Layout name must contain at least one alphanumeric character.")
            return
        
        self.layout_name = sanitized
        self.accept()
    
    def get_layout_name(self):
        """Return the user-entered layout name."""
        return self.layout_name


class QtLayoutManager:
    """Manages saving and restoring dock layouts using PyQtAds serialization."""
    
    # Base cache directory
    CACHE_BASE = ".cache"
    LAYOUTS_SUBDIR = "layout"
    DEFAULT_LAYOUT_NAME = "default"
    
    @classmethod
    def get_cache_dir(cls) -> Path:
        """
        Get the cache directory path, creating it if necessary.
        Uses current working directory as base.
        
        Returns:
            Path: Path to cache directory
        """
        cache_dir = Path(cls.CACHE_BASE) / cls.LAYOUTS_SUBDIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @classmethod
    def get_layout_path(cls, layout_name: str = DEFAULT_LAYOUT_NAME) -> Path:
        """
        Get the path to a specific layout configuration file.
        
        Args:
            layout_name: Name of the layout (without extension)
            
        Returns:
            Path: Full path to the layout JSON file
        """
        cache_dir = cls.get_cache_dir()
        return cache_dir / f"{layout_name}.json"
    
    @classmethod
    def list_available_layouts(cls) -> list:
        """
        Discover all available layout configurations in the cache folder.
        
        Returns:
            list: Sorted list of layout names (without .json extension)
        """
        try:
            cache_dir = cls.get_cache_dir()
            layouts = []
            
            for layout_file in cache_dir.glob("*.json"):
                layout_name = layout_file.stem  # filename without extension
                layouts.append(layout_name)
            
            return sorted(layouts)
        except Exception as e:
            print(f"⚠️ Error listing layouts: {e}")
            return []
    
    @classmethod
    def save_layout(cls, dock_manager, layout_name: str = DEFAULT_LAYOUT_NAME) -> bool:
        """
        Save the current dock layout to a JSON file.
        
        PyQtAds' saveState() returns a QByteArray with serialized dock configuration.
        We encode it to base64 and store in JSON for human readability (optional).
        
        Args:
            dock_manager: The PyQtAds CDockManager instance
            layout_name: Name of the layout to save (default: "default")
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            layout_path = cls.get_layout_path(layout_name)
            
            # Serialize dock state using PyQtAds
            state = dock_manager.saveState()
            
            # Convert QByteArray to base64 string for JSON storage
            state_bytes = bytes(state)
            state_b64 = base64.b64encode(state_bytes).decode('utf-8')
            
            # Create config dict
            config = {
                'layout_name': layout_name,
                'dock_state': state_b64
            }
            
            # Write to JSON file
            with open(layout_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Layout saved to {layout_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save layout '{layout_name}': {e}")
            return False
    
    @classmethod
    def load_layout(cls, dock_manager, layout_name: str = DEFAULT_LAYOUT_NAME) -> bool:
        """
        Restore a dock layout from a saved configuration file.
        
        Args:
            dock_manager: The PyQtAds CDockManager instance
            layout_name: Name of the layout to restore (default: "default")
            
        Returns:
            bool: True if restored successfully, False otherwise
        """
        try:
            layout_path = cls.get_layout_path(layout_name)
            
            # Check if layout file exists
            if not layout_path.exists():
                print(f"ℹ️ Layout file not found: {layout_path}")
                return False
            
            # Read JSON and decode state
            with open(layout_path, 'r') as f:
                config = json.load(f)
            
            state_b64 = config.get('dock_state')
            if not state_b64:
                print(f"⚠️ Invalid layout file (missing dock_state): {layout_path}")
                return False
            
            # Decode base64 back to QByteArray
            state_bytes = base64.b64decode(state_b64.encode('utf-8'))
            state = QByteArray(state_bytes)
            
            # Restore using PyQtAds
            success = dock_manager.restoreState(state)
            
            if success:
                print(f"✅ Layout restored from {layout_path}")
            else:
                print(f"⚠️ Failed to restore layout (PyQtAds rejected state): {layout_path}")
            
            return success
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse layout JSON '{layout_name}': {e}")
            return False
        except Exception as e:
            print(f"⚠️ Failed to load layout '{layout_name}': {e}")
            return False
    
    @classmethod
    def save_and_close(cls, dock_manager, layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
        """
        Convenience method: save layout and gracefully handle errors.
        Intended for closeEvent() - won't raise exceptions.
        
        Args:
            dock_manager: The PyQtAds CDockManager instance
            layout_name: Name of the layout to save
        """
        try:
            cls.save_layout(dock_manager, layout_name)
        except Exception as e:
            print(f"⚠️ Error during automatic layout save: {e}")
    
    @classmethod
    def show_save_dialog_and_save(cls, dock_manager, parent=None) -> bool:
        """
        Show a save dialog and save the layout with the user-entered name.
        
        Args:
            dock_manager: The PyQtAds CDockManager instance
            parent: Parent widget for the dialog
            
        Returns:
            bool: True if saved successfully, False if cancelled or error
        """
        dialog = SaveLayoutDialog(parent)
        if dialog.exec_() == QDialog.Accepted:
            layout_name = dialog.get_layout_name()
            success = cls.save_layout(dock_manager, layout_name)
            if success:
                QMessageBox.information(
                    parent,
                    "Layout Saved",
                    f"Layout '{layout_name}' has been saved successfully."
                )
            return success
        return False
    
    @classmethod
    def restore_or_default(cls, dock_manager, layout_name: str = DEFAULT_LAYOUT_NAME,
                           fallback_fn=None) -> None:
        """
        Restore layout, falling back to default if it fails.
        
        Args:
            dock_manager: The PyQtAds CDockManager instance
            layout_name: Name of the layout to restore
            fallback_fn: Optional callable to invoke if restore fails
                        (e.g., reset_layout_to_default)
        """
        try:
            restored = cls.load_layout(dock_manager, layout_name)
            if not restored and fallback_fn:
                print("🔄 Falling back to default layout")
                fallback_fn()
        except Exception as e:
            print(f"⚠️ Error during automatic layout restore: {e}")
            if fallback_fn:
                fallback_fn()
