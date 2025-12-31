import warnings
import gc
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout, 
                             QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from coralnet_toolbox.Common.QtCollapsibleSection import CollapsibleSection
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(CollapsibleSection):
    """
    Z-Inference deployment dialog for loading and running depth estimation models.
    
    Inherits from CollapsibleSection to provide a collapsible UI in the status bar.
    """
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the Z-Inference deployment dialog.
        
        Args:
            main_window: Reference to the main window
            parent: Parent widget
        """
        super().__init__("Z-Inference", "z.png", parent)
        
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        
        # Initialize instance variables
        self.loaded_model = None
        self.imgsz = 512
        self.model_path = None
        
        # Setup UI components
        self.setup_model_layout()
        self.setup_buttons_layout()
        self.setup_status_layout()
        self.setup_deploy_layout()
        
    def setup_model_layout(self):
        """Setup the model selection and parameters layout."""
        layout = QFormLayout()
        
        # Model dropdown
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        
        # Define available models
        self.models = {
            "DA3-Large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        }
        
        # Add models to combo box
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)
            
        self.model_combo.setCurrentIndex(0)
        layout.addRow("Model:", self.model_combo)
        
        # Image size spinbox
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(256, 4096)
        self.imgsz_spinbox.setSingleStep(128)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size:", self.imgsz_spinbox)
        
        # Create widget and add to popup
        widget = QWidget()
        widget.setLayout(layout)
        self.add_widget(widget, "Model Selection")
        
    def setup_buttons_layout(self):
        """Setup the Load/Deactivate buttons layout."""
        layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_button)
        
        # Deactivate button
        self.deactivate_button = QPushButton("Deactivate Model")
        self.deactivate_button.clicked.connect(self.deactivate_model)
        self.deactivate_button.setEnabled(False)
        layout.addWidget(self.deactivate_button)
        
        # Create widget and add to popup
        widget = QWidget()
        widget.setLayout(layout)
        self.add_widget(widget, "Model Management")
        
    def setup_status_layout(self):
        """Setup the status label layout."""
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("No model loaded")
        layout.addWidget(self.status_label)
        
        # Create widget and add to popup
        widget = QWidget()
        widget.setLayout(layout)
        self.add_widget(widget, "Status")
        
    def setup_deploy_layout(self):
        """Setup the Deploy Model button layout."""
        layout = QVBoxLayout()
        
        # Deploy button
        self.deploy_button = QPushButton("Deploy Model")
        self.deploy_button.clicked.connect(self.deploy_model)
        self.deploy_button.setEnabled(False)
        layout.addWidget(self.deploy_button)
        
        # Create widget and add to popup
        widget = QWidget()
        widget.setLayout(layout)
        self.add_widget(widget, "Deployment")
        
    def load_model(self):
        """Load the selected depth estimation model."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        
        try:
            # Get selected model and image size
            model_name = self.model_combo.currentText()
            self.model_path = self.models.get(model_name, model_name)
            self.imgsz = self.imgsz_spinbox.value()
            
            # Import and instantiate the model based on selection
            if "DA3" in model_name:
                from coralnet_toolbox.Z.Models.DA3 import DA3
                self.loaded_model = DA3(self.main_window.device)
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Load the model
            self.loaded_model.load_model(self.model_path, self.imgsz)
            
            # Update UI state
            self.status_label.setText(f"Model loaded: {model_name}")
            self.load_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            # Only enable deploy button if an image is currently loaded
            self.deploy_button.setEnabled(self.main_window.image_window.current_raster is not None)
            
            progress_bar.finish_progress()
            QMessageBox.information(self.annotation_window, 
                                    "Model Loaded", 
                                    "Model loaded successfully")
            
        except Exception as e:
            self.status_label.setText("Model loading failed")
            QMessageBox.critical(self.annotation_window, 
                                 "Error Loading Model", 
                                 f"Error loading model: {e}")
            
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            
    def deactivate_model(self):
        """Deactivate and clear the loaded model from memory."""
        if self.loaded_model is not None:
            try:
                self.loaded_model.deactivate()
                self.loaded_model = None
                
                # Force garbage collection
                gc.collect()
                
                # Update UI state
                self.status_label.setText("Model deactivated")
                self.load_button.setEnabled(True)
                self.deactivate_button.setEnabled(False)
                self.deploy_button.setEnabled(False)
                
                QMessageBox.information(self.annotation_window, 
                                        "Model Deactivated", 
                                        "Model deactivated successfully")
                
            except Exception as e:
                QMessageBox.critical(self.annotation_window, 
                                     "Error", 
                                     f"Error deactivating model: {e}")
                
    def update_deploy_button_state(self):
        """Update deploy button enabled state based on model and image availability."""
        # Enable only if model is loaded AND image is loaded
        has_model = self.loaded_model is not None
        has_image = self.main_window.image_window.current_raster is not None
        self.deploy_button.setEnabled(has_model and has_image)
                
    def deploy_model(self):
        """Deploy the model on the current image to generate Z-channel."""
        if self.loaded_model is None:
            QMessageBox.warning(self.annotation_window, 
                                "No Model", 
                                "Please load a model first")
            return
            
        # Get current raster
        current_raster = self.main_window.image_window.current_raster
        if current_raster is None:
            QMessageBox.warning(self.annotation_window, 
                                "No Image", 
                                "Please load an image first")
            return
            
        # Check if raster already has z-channel
        if current_raster.z_channel is not None:
            reply = self._show_overwrite_dialog()
            
            if reply == "cancel":
                return
            # Pass the mode directly to predict
            overwrite_mode = reply
        else:
            overwrite_mode = "overwrite"
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Deploying Model")
        progress_bar.show()
        
        try:
            # Use the predict method with the selected overwrite mode
            self.predict([current_raster.image_path], progress_bar, overwrite_mode=overwrite_mode)
            progress_bar.finish_progress()

        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Error", 
                                 f"Error during deployment: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            
    def predict(self, image_paths, progress_bar, overwrite_mode="prompt"):
        """
        Run Z-Inference prediction on multiple images.
        
        Args:
            image_paths: List of image paths to process
            progress_bar: ProgressBar instance for tracking progress
            overwrite_mode: How to handle existing z-channels:
                           "prompt" - ask user once (for batch)
                           "overwrite" - overwrite without asking
                           "skip" - skip images with existing z-channels
                           "smart_fill" - fill NaN values with scaled predictions
        """
        if self.loaded_model is None:
            raise ValueError("No model loaded")
            
        # For batch processing with prompt mode, check if any images have z-channels
        if overwrite_mode == "prompt" and len(image_paths) > 1:
            has_z_channel = False
            for image_path in image_paths:
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if raster and raster.z_channel is not None:
                    has_z_channel = True
                    break
                    
            if has_z_channel:
                reply = self._show_overwrite_dialog()
                if reply == "cancel":
                    return
                # Update overwrite mode based on user choice
                overwrite_mode = reply
        
        # Process each image
        num_images = len(image_paths)
        for idx, image_path in enumerate(image_paths):
            try:
                progress_bar.set_title(f"Z-Inference: {idx + 1}/{num_images} - {os.path.basename(image_path)}")
                
                # Get raster for this image
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    print(f"Failed to get raster for {image_path}")
                    continue
                
                # Check if raster already has z-channel
                if raster.z_channel is not None:
                    if overwrite_mode == "skip":
                        continue
                    elif overwrite_mode == "smart_fill":
                        # Run prediction first
                        image_array = raster.get_numpy()
                        if image_array is None:
                            print(f"Failed to get image array from raster for {image_path}")
                            continue
                        
                        z_predicted = self.loaded_model.predict(image_array)
                        
                        # Apply Smart Fill
                        self._handle_smart_fill(raster, z_predicted)
                        
                        # Refresh visualization if this is the current raster
                        if raster == self.main_window.image_window.current_raster:
                            self.annotation_window.refresh_z_channel_visualization()
                            # Update z_type_button based on current type
                            if raster.z_data_type == 'elevation':
                                self.main_window.z_type_button.setIcon(self.main_window.elevation_icon)
                                self.main_window.z_type_button.setToolTip("Current: Elevation\nClick to convert to Depth")
                            else:
                                self.main_window.z_type_button.setIcon(self.main_window.depth_icon)
                                self.main_window.z_type_button.setToolTip("Current: Depth\nClick to convert to Elevation")
                        continue
                    # If overwrite_mode == "overwrite", continue with prediction below
                
                # Get numpy array from raster
                image_array = raster.get_numpy()
                
                if image_array is None:
                    print(f"Failed to get image array from raster for {image_path}")
                    continue
                    
                # Run prediction
                z_channel = self.loaded_model.predict(image_array)
                
                # Add z-channel to raster with 'depth' type (DA3 produces depth maps)
                raster.add_z_channel(z_channel, z_unit='meters', z_data_type='depth')
                
                # If this is the current raster, refresh visualization and enable controls
                if raster == self.main_window.image_window.current_raster:
                    # Refresh visualization
                    self.annotation_window.refresh_z_channel_visualization()
                    
                    # Enable Z-controls in status bar
                    self.main_window.z_unit_dropdown.setEnabled(True)
                    self.main_window.z_label.setEnabled(True)
                    self.main_window.z_colormap_dropdown.setEnabled(True)
                    self.main_window.z_dynamic_button.setEnabled(True)
                    self.main_window.z_type_button.setEnabled(True)  # Enable depth/elevation conversion
                    
                    # Set button to show depth mode
                    self.main_window.z_type_button.setIcon(self.main_window.depth_icon)
                    self.main_window.z_type_button.setToolTip("Current: Depth\nClick to convert to Elevation")
                    
                    # Set colormap to Turbo
                    turbo_index = self.main_window.z_colormap_dropdown.findText("Turbo")
                    if turbo_index >= 0:
                        self.main_window.z_colormap_dropdown.setCurrentIndex(turbo_index)
                        
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
    def _show_overwrite_dialog(self):
        """
        Show dialog when Z-channel already exists.
        
        Returns:
            str: "cancel", "overwrite", or "smart_fill"
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowIcon(get_icon("z.png"))
        msg.setWindowTitle("Z-Channel Exists")
        msg.setText("This image already has a Z-channel.")
        msg.setInformativeText("What would you like to do?")
        
        # Set minimum width for the dialog
        msg.setStyleSheet("QMessageBox { min-width: 500px; }")
        
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        overwrite_btn = msg.addButton("Overwrite", QMessageBox.AcceptRole)
        smart_fill_btn = msg.addButton("Smart Fill", QMessageBox.ActionRole)
        
        # Add tooltips to each button
        cancel_btn.setToolTip("Do nothing and return to the main window")
        overwrite_btn.setToolTip("Replace the entire existing Z-channel with new predictions")
        smart_fill_btn.setToolTip(
            "Preserve valid depth/elevation values and fill only NaN/missing areas "
            "with scaled predictions matched to existing data. \nAutomatically handles "
            "depth/elevation conversion and unit conversion to meters."
        )
        
        msg.exec_()
        
        if msg.clickedButton() == cancel_btn:
            return "cancel"
        elif msg.clickedButton() == overwrite_btn:
            return "overwrite"
        elif msg.clickedButton() == smart_fill_btn:
            return "smart_fill"
        else:
            return "cancel"
            
    def _handle_smart_fill(self, raster, z_predicted):
        """
        Handle Smart Fill - fill NaN values in existing z-channel with scaled predictions.
        
        This method:
        1. Converts existing Z-channel to meters if needed
        2. Converts predictions from depth to elevation if existing data is elevation type
        3. Computes scale and offset to map predicted range to existing range
        4. Fills only the NaN values with scaled predictions
        
        This preserves known-good depth measurements while filling gaps with model predictions.
        
        Args:
            raster: The raster object with existing z_channel
            z_predicted: New z-channel predictions from the model (numpy array, depth in meters)
        """
        import numpy as np
        from coralnet_toolbox.utilities import convert_scale_units
        
        if raster.z_channel is None:
            print("No existing z-channel to fill")
            return
            
        if raster.z_channel.dtype != np.float32:
            print("Smart Fill only works with float32 z-channels (not uint8)")
            QMessageBox.warning(self.annotation_window, 
                                "Incompatible Data Type", 
                                "Smart Fill only works with float32 depth data, not uint8.")
            return
        
        # Step 1: Check for NaN values first (before modifying anything)
        z_existing = raster.z_channel.copy()
        valid_mask = ~np.isnan(z_existing)
        nan_mask = np.isnan(z_existing)
        
        num_valid = np.sum(valid_mask)
        num_nans = np.sum(nan_mask)
        total_pixels = z_existing.size
        
        # Early return if no NaN values to fill
        if num_nans == 0:
            QMessageBox.information(self.annotation_window,
                                    "No Gaps to Fill",
                                    "The existing Z-channel has no missing (NaN) values to fill.")
            return
        
        # Step 2: Convert existing z-channel to meters if needed
        original_unit = raster.z_unit or 'meters'
        
        if original_unit != 'meters':
            print(f"Converting existing Z-channel from {original_unit} to meters")
            # Convert each value from original_unit to meters
            for i in range(z_existing.shape[0]):
                for j in range(z_existing.shape[1]):
                    if not np.isnan(z_existing[i, j]):
                        z_existing[i, j] = convert_scale_units(z_existing[i, j], original_unit, 'meters')
            print(f"Converted {original_unit} -> meters")
        
        # Step 3: Convert predictions to elevation if existing data is elevation
        z_predicted_adjusted = z_predicted.copy()
        if raster.z_data_type == 'elevation':
            if raster.z_inversion_reference is None:
                print("Warning: Elevation data without inversion reference, cannot convert predictions")
                QMessageBox.warning(self.annotation_window,
                                    "Missing Reference",
                                    "Elevation data is missing inversion reference. Cannot perform Smart Fill.")
                return
            
            # Convert depth predictions to elevation using existing's inversion reference
            # elevation = reference - depth
            print(f"Converting predictions from depth to elevation using reference: {raster.z_inversion_reference:.3f}")
            z_predicted_adjusted = raster.z_inversion_reference - z_predicted
        
        # Step 4: Check if we have enough valid pixels to compute scaling
        valid_percentage = (num_valid / total_pixels) * 100
        
        if num_valid < 100 or valid_percentage < 1.0:
            QMessageBox.warning(self.annotation_window, 
                                "Insufficient Data", 
                                f"Not enough valid depth data to compute scaling.\n"
                                f"Found {num_valid} valid pixels ({valid_percentage:.1f}% coverage).\n"
                                f"Need at least 100 valid pixels and 1% coverage.")
            return
        
        print(f"Smart Fill: Found {num_valid} valid pixels ({valid_percentage:.1f}%), {num_nans} NaN pixels to fill")
        
        # Step 5: Extract valid values from both datasets for scaling computation
        existing_valid = z_existing[valid_mask]
        predicted_valid = z_predicted_adjusted[valid_mask]
        
        # Compute statistics for scaling using percentiles (more robust than min/max)
        # Use 5th and 95th percentiles to avoid outliers
        existing_p05 = np.percentile(existing_valid, 5)
        existing_p95 = np.percentile(existing_valid, 95)
        predicted_p05 = np.percentile(predicted_valid, 5)
        predicted_p95 = np.percentile(predicted_valid, 95)
        
        existing_range = existing_p95 - existing_p05
        predicted_range = predicted_p95 - predicted_p05
        
        # Step 5: Compute linear scaling: z_scaled = scale * z_predicted + offset
        # Map predicted range to existing range
        if predicted_range > 1e-6:  # Avoid division by zero
            scale = existing_range / predicted_range
            offset = existing_p05 - (scale * predicted_p05)
        else:
            print("Warning: Predicted depth range is too small, using 1:1 scaling")
            scale = 1.0
            offset = np.mean(existing_valid) - np.mean(predicted_valid)
        
        # Apply scaling to predicted data
        z_scaled = scale * z_predicted_adjusted + offset
        
        print(f"Scaling parameters: scale={scale:.4f}, offset={offset:.4f}")
        print(f"Existing range: [{existing_p05:.3f}, {existing_p95:.3f}] meters")
        print(f"Predicted range: [{predicted_p05:.3f}, {predicted_p95:.3f}] -> "
              f"[{scale * predicted_p05 + offset:.3f}, {scale * predicted_p95 + offset:.3f}] meters")
        
        # Step 6: Fill NaN values with scaled predictions
        print(f"Filling {num_nans} NaN pixels ({(num_nans/total_pixels)*100:.1f}% of image)")
        
        # Create result by copying existing and filling NaNs
        z_result = z_existing.copy()
        z_result[nan_mask] = z_scaled[nan_mask]
        
        # Step 7: Update the z-channel with the filled result
        # Preserve existing metadata (z_data_type, z_inversion_reference)
        # Update unit to meters
        raster.z_channel = z_result
        raster.z_unit = 'meters'
        # z_data_type and z_inversion_reference are preserved
        
        print(f"Smart Fill complete: preserved {num_valid} pixels, filled {num_nans} pixels")
        print(f"Result type: {raster.z_data_type}, unit: {raster.z_unit}")
