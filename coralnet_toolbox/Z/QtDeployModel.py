import warnings

import os
import gc
import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout, 
                             QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from coralnet_toolbox.Common.QtCollapsibleSection import CollapsibleSection
from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import smart_fill_z_channel

from coralnet_toolbox.Icons import get_icon

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(CollapsibleSection):
    """
    Z-Inference deployment dialog for loading and running depth estimation models.
    
    Inherits from CollapsibleSection to provide a collapsible UI in the status bar.
    """
    
    def __init__(self, main_window, parent=None, highlighted_images=None):
        """
        Initialize the Z-Inference deployment dialog.
        
        Args:
            main_window: Reference to the main window
            parent: Parent widget
            highlighted_images: Optional list of image paths to process in batch
        """
        super().__init__("Z-Inference", "z.png", parent)
        
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        
        # Initialize instance variables
        self.loaded_model = None
        self.imgsz = 504
        self.model_path = None
        
        self.highlighted_images = highlighted_images if highlighted_images else []
        self.last_overwrite_mode = None  # Cache the overwrite choice for batch processing
        self.cuda_warning_shown = False  # Track if CUDA warning has been shown
        
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
        self.imgsz_spinbox.setRange(256, 504)
        self.imgsz_spinbox.setSingleStep(64)
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
        
        # Deploy button (works on current/highlighted images)
        self.deploy_button = QPushButton("Deploy Model")
        self.deploy_button.clicked.connect(self.deploy_model)
        self.deploy_button.setEnabled(False)
        self.deploy_button.setToolTip("Deploy model on the current/highlighted images")
        layout.addWidget(self.deploy_button)
        
        # Create widget and add to popup
        widget = QWidget()
        widget.setLayout(layout)
        self.add_widget(widget, "Deployment")
        
    def load_model(self):
        """Load the selected depth estimation model."""
        try:
            # Check if depth-anything-3 is installed
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            QMessageBox.warning(self, 
                                "Missing Package", 
                                "The 'awesome-depth-anything-3' package is required for Z-Inference.\n\n"
                                "Please install it via pip:\npip install awesome-depth-anything-3")
            return
        
        # Check if HF_TOKEN environment variable is set
        # (if not, user can definitely not access the model)
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token or not hf_token.strip():
            QMessageBox.warning(self, 
                                "HuggingFace Access Required", 
                                "Access to Depth-Anything-3 model weights requires HuggingFace approval.\n\n"
                                "Please:\n"
                                "1. Request access to 'depth-anything/DA3NESTED-GIANT-LARGE-1.1' on HuggingFace\n"
                                "2. Set your HF_TOKEN environment variable: export HF_TOKEN=your_token_here\n"
                                "3. Restart the application")
            return

        # Finally, check for CUDA availability before proceeding
        if not self.cuda_warning_shown and not torch.cuda.is_available():
            reply = QMessageBox.question(
                self.annotation_window,
                "No CUDA Detected",
                "Depth estimation models will run very slowly without CUDA support. "
                "Consider using a GPU-enabled device.\n\nDo you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            self.cuda_warning_shown = True
        
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
            # Update button state and text
            self.update_deploy_button_state()
            
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
                # Update button state and text
                self.update_deploy_button_state()
                
                QMessageBox.information(self.annotation_window, 
                                        "Model Deactivated", 
                                        "Model deactivated successfully")
                
            except Exception as e:
                QMessageBox.critical(self.annotation_window, 
                                     "Error", 
                                     f"Error deactivating model: {e}")
                
    def update_deploy_button_text(self):
        """
        Update deploy button text to show the count of images that will be processed.
        Format: "Deploy Model (N Highlighted Images)" or "Deploy Model (1 Current Image)" or "Deploy Model"
        """
        if len(self.highlighted_images) > 0:
            # Multiple or single highlighted images
            count_text = f"Deploy ({len(self.highlighted_images)} Highlighted)"
        elif self.main_window.image_window.current_raster is not None:
            # Fall back to current image count
            count_text = "Deploy (1 Highlighted)"
        else:
            # No images
            count_text = "Deploy"
        
        self.deploy_button.setText(count_text)
    
    def update_deploy_button_state(self):
        """Update deploy button enabled state and text based on model and image availability."""
        # Enable only if model is loaded AND (current image loaded OR images are highlighted)
        has_model = self.loaded_model is not None
        has_image = self.main_window.image_window.current_raster is not None
        has_highlighted = len(self.highlighted_images) > 0
        self.deploy_button.setEnabled(has_model and (has_image or has_highlighted))
        # Update button text
        self.update_deploy_button_text()
    
    def _validate_prediction_result(self, result, image_path=None):
        """
        Validate prediction result contains depth_maps.
        
        Args:
            result: Prediction result dictionary
            image_path: Optional path for error messages
            
        Returns:
            list or None: depth_maps array if valid, None otherwise
        """
        if not result or 'depth_maps' not in result or result['depth_maps'] is None:
            print(f"No depth maps returned{' for ' + image_path if image_path else ''}")
            return None
        
        depth_maps = result['depth_maps']
        if len(depth_maps) == 0:
            print(f"Empty depth maps returned{' for ' + image_path if image_path else ''}")
            return None
        
        return depth_maps
    
    def _extract_depth_map(self, depth_map, image_path=None):
        """
        Extract and validate a single depth map, ensuring it's 2D.
        
        Args:
            depth_map: Depth map array (possibly 3D)
            image_path: Optional path for error messages
            
        Returns:
            numpy.ndarray or None: 2D depth map if valid, None otherwise
        """
        # Extract depth map and ensure it's 2D
        if isinstance(depth_map, np.ndarray) and depth_map.ndim == 3:
            # Squeeze batch dimension if present
            if depth_map.shape[0] == 1:
                depth_map = depth_map[0]
            else:
                print(f"Unexpected 3D depth map shape{' for ' + image_path if image_path else ''}: {depth_map.shape}")
                return None
        
        # Validate it's 2D
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            shape_info = depth_map.shape if isinstance(depth_map, np.ndarray) else type(depth_map)
            print(f"Expected 2D depth map{' for ' + image_path if image_path else ''}, got shape {shape_info}")
            return None
        
        return depth_map
    
    def _resize_depth_map(self, depth_map, raster):
        """
        Resize depth map to match original raster dimensions if needed.
        
        Args:
            depth_map: 2D depth map array
            raster: Raster object with target dimensions
            
        Returns:
            numpy.ndarray: Resized depth map
        """
        if depth_map.shape != (raster.height, raster.width):
            # Use INTER_NEAREST to preserve depth values without interpolation artifacts
            depth_map = cv2.resize(depth_map, (raster.width, raster.height), interpolation=cv2.INTER_NEAREST)
        return depth_map
    
    def _update_camera_parameters(self, raster, result, index=0):
        """
        Update raster with camera intrinsics and extrinsics from prediction result.
        
        Args:
            raster: Raster object to update
            result: Prediction result dictionary
            index: Index in result arrays (default 0 for single image)
        """
        intrinsics = result.get('intrinsics')
        if intrinsics is not None and len(intrinsics) > index and intrinsics[index] is not None:
            raster.add_intrinsics(intrinsics[index])
        
        extrinsics = result.get('extrinsics')
        if extrinsics is not None and len(extrinsics) > index and extrinsics[index] is not None:
            raster.add_extrinsics(extrinsics[index])
    
    def deploy_model(self):
        """Deploy the model on highlighted images, or current image if none are highlighted."""
        if self.loaded_model is None:
            QMessageBox.warning(self.annotation_window, 
                                "No Model", 
                                "Please load a model first")
            return
        
        # Determine which images to process
        # If highlighted images exist, use them; otherwise use current image
        if self.highlighted_images:
            image_paths = self.highlighted_images
            # Reset the cached overwrite mode for this batch
            self.last_overwrite_mode = None
            show_dialog = True
        else:
            # Use current raster
            current_raster = self.main_window.image_window.current_raster
            if current_raster is None:
                QMessageBox.warning(self.annotation_window, 
                                    "No Image", 
                                    "Please load an image first")
                return
            
            image_paths = [current_raster.image_path]
            show_dialog = False
            
            # Check if raster already has z-channel (single image case)
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
            # Use the predict method
            if self.highlighted_images:
                # Batch mode - use prompt to show dialog if needed
                self.predict(image_paths, progress_bar, overwrite_mode="prompt", show_dialog=show_dialog)
            else:
                # Single image mode - use the selected overwrite mode
                self.predict(image_paths, progress_bar, overwrite_mode=overwrite_mode, show_dialog=show_dialog)
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
    
    def update_highlighted_images(self, highlighted_paths):
        """
        Update the list of highlighted images and button state.
        
        Args:
            highlighted_paths: List of image paths that are currently highlighted
        """
        self.highlighted_images = highlighted_paths if highlighted_paths else []
        # Update button state and text
        self.update_deploy_button_state()
            
    def predict(self, image_paths, progress_bar, overwrite_mode="prompt", show_dialog=True):
        """
        Run Z-Inference prediction on multiple images.
        
        Args:
            image_paths: List of image paths to process
            progress_bar: ProgressBar instance for tracking progress
            overwrite_mode: How to handle existing z-channels:
                           "overwrite" - overwrite without asking
                           "skip" - skip images with existing z-channels
                           "smart_fill" - fill NaN values with scaled predictions
            show_dialog: Whether to show the overwrite dialog (False for single image from deploy_model)
        """
        if self.loaded_model is None:
            raise ValueError("No model loaded")
            
        # Check if any images already have z-channels
        any_have_z = any(
            (raster := self.main_window.image_window.raster_manager.get_raster(path)) and (
                raster.z_channel is not None or raster.z_channel_path)
            for path in image_paths
        )
        
        # For batch processing, show dialog only if any images have existing z-channels
        if len(image_paths) > 1 and show_dialog and any_have_z:
            reply = self._show_overwrite_dialog(is_batch=True)
            if reply is None:
                return
            overwrite_mode = reply
        elif len(image_paths) > 1 and show_dialog and not any_have_z:
            overwrite_mode = "overwrite"
        
        # Process each image
        progress_bar.start_progress(len(image_paths))
        num_images = len(image_paths)
        
        # Collect paths that need prediction (overwrite or no existing z-channel)
        collected_overwrite = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                progress_bar.set_title(f"Z-Inference: {idx + 1}/{num_images} - {os.path.basename(image_path)}")
                
                # Get raster for this image
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    print(f"Failed to get raster for {image_path}")
                    progress_bar.update_progress()
                    continue
                
                # Load z_channel if it exists but not loaded
                if raster.z_channel is None and raster.z_channel_path:
                    raster.load_z_channel_from_file(raster.z_channel_path)
                
                # Check if raster already has z-channel
                if raster.z_channel is not None:
                    if overwrite_mode == "skip":
                        progress_bar.update_progress()
                        continue
                    
                    elif overwrite_mode == "smart_fill":
                        # Run prediction first
                        result = self.loaded_model.predict([image_path])
                        
                        # Validate and extract depth maps
                        depth_maps = self._validate_prediction_result(result, image_path)
                        if depth_maps is None:
                            progress_bar.update_progress()
                            continue
                        
                        # Extract 2D depth map
                        z_predicted = self._extract_depth_map(depth_maps[0], image_path)
                        if z_predicted is None:
                            progress_bar.update_progress()
                            continue
                        
                        # Resize to match original raster dimensions
                        z_predicted = self._resize_depth_map(z_predicted, raster)
                        
                        # Update camera parameters
                        self._update_camera_parameters(raster, result, index=0)
                        
                        # Apply Smart Fill
                        self._handle_smart_fill(raster, z_predicted)
                        
                        # Emit rasterUpdated signal to refresh UI (including status bar Z-value)
                        self.main_window.image_window.raster_manager.rasterUpdated.emit(image_path)
                        
                        # Refresh visualization if this is the current raster
                        if raster == self.main_window.image_window.current_raster:
                            self.annotation_window.refresh_z_channel_visualization()
                            
                        progress_bar.update_progress()
                        continue
                    # If overwrite_mode == "overwrite", fall through to collect for batch prediction
                else:
                    if overwrite_mode == "skip":
                        progress_bar.update_progress()
                        continue
                
                # Collect for batch prediction
                collected_overwrite.append(image_path)
                
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # For collected, progress will be updated after batch
                if image_path not in collected_overwrite:
                    progress_bar.update_progress()
        
        # Batch predict for collected images
        if collected_overwrite:
            try:
                result = self.loaded_model.predict(collected_overwrite)
                
                # Validate and extract depth maps
                depth_maps = self._validate_prediction_result(result)
                if depth_maps is None:
                    for _ in collected_overwrite:
                        progress_bar.update_progress()
                    return
                
                # Validate we have enough depth maps
                if len(depth_maps) != len(collected_overwrite):
                    print(f"Warning: Expected {len(collected_overwrite)} depth maps, got {len(depth_maps)}")
                
                for i, image_path in enumerate(collected_overwrite):
                    raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                    if raster is None:
                        progress_bar.update_progress()
                        continue
                    
                    # Check if we have a depth map for this image
                    if i >= len(depth_maps):
                        print(f"No depth map for image {i}: {image_path}")
                        progress_bar.update_progress()
                        continue
                    
                    # Extract 2D depth map
                    z_channel = self._extract_depth_map(depth_maps[i], image_path)
                    if z_channel is None:
                        progress_bar.update_progress()
                        continue
                    
                    # Resize to match original raster dimensions
                    z_channel = self._resize_depth_map(z_channel, raster)
                    
                    # Update camera parameters
                    self._update_camera_parameters(raster, result, index=i)
                    
                    # Add z-channel to raster with 'depth' type (DA3 produces depth maps)
                    # Set direction=1 explicitly (high values = far, which is depth semantics)
                    # Note: 0 values are automatically treated as nodata for depth maps
                    raster.add_z_channel(z_channel, z_unit='meters', z_data_type='depth', z_direction=1)
                    
                    # Note: add_z_channel now automatically emits zChannelUpdated signal
                    
                    # If this is the current raster, refresh visualization and enable controls
                    if raster == self.main_window.image_window.current_raster:
                        # Refresh visualization
                        self.annotation_window.refresh_z_channel_visualization()
                        
                        # Enable Z-controls in status bar now that z-channel exists
                        self.main_window.enable_z_visualization_controls(True)
                        
                        # Set colormap to Turbo
                        turbo_index = self.main_window.z_colormap_dropdown.findText("Turbo")
                        if turbo_index >= 0:
                            self.main_window.z_colormap_dropdown.setCurrentIndex(turbo_index)
                            
                    progress_bar.update_progress()
                    
            except Exception as e:
                print(f"Failed batch prediction: {e}")
                import traceback
                traceback.print_exc()
                # Update progress for remaining
                for _ in collected_overwrite:
                    progress_bar.update_progress()
            
    def _show_overwrite_dialog(self, is_batch=False):
        """
        Show dialog for Z-channel handling.
        
        Args:
            is_batch (bool): Whether this is for batch processing multiple images.
        
        Returns:
            str or None: "overwrite", "skip", "smart_fill", or None for cancel
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowIcon(get_icon("z.png"))
        
        # Set minimum width for the dialog
        msg.setStyleSheet("QMessageBox { min-width: 500px; }")
        
        if is_batch:
            msg.setWindowTitle("Z-Inference Options")
            msg.setText("How would you like to proceed with Z-Inference on these images?")
            msg.setInformativeText("")
            
            overwrite_btn = msg.addButton("Overwrite All", QMessageBox.AcceptRole)
            skip_btn = msg.addButton("Skip All", QMessageBox.RejectRole)
            smart_fill_btn = msg.addButton("Smart Fill All", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
            
            # Add tooltips
            overwrite_btn.setToolTip("Replace existing Z-channels or add to images without Z-channels")
            skip_btn.setToolTip("Skip all images without processing")
            smart_fill_btn.setToolTip(
                "Preserve valid depth/elevation values and fill only NaN/missing areas "
                "with scaled predictions matched to existing data. \nAutomatically handles "
                "depth/elevation conversion and unit conversion to meters. Skips images without existing Z-channels."
            )
            cancel_btn.setToolTip("Cancel the operation")
            
            msg.exec_()
            
            if msg.clickedButton() == cancel_btn:
                return None
            elif msg.clickedButton() == overwrite_btn:
                return "overwrite"
            elif msg.clickedButton() == skip_btn:
                return "skip"
            elif msg.clickedButton() == smart_fill_btn:
                return "smart_fill"
            else:
                return None
        else:
            msg.setWindowTitle("Z-Channel Exists")
            msg.setText("This image already has a Z-channel.")
            msg.setInformativeText("What would you like to do?")
            
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
        
        This method uses the smart_fill_z_channel utility function to preserve known-good
        depth measurements while filling gaps with model predictions.
        
        Args:
            raster: The raster object with existing z_channel
            z_predicted: New z-channel predictions from the model (numpy array, depth in meters)
        """        
        if raster.z_channel is None:
            print("No existing z-channel to fill")
            return
        
        try:
            # Perform smart fill using utility function
            z_result, stats = smart_fill_z_channel(
                existing_z=raster.z_channel,
                predicted_z=z_predicted,
                existing_unit=raster.z_unit or 'meters',
                existing_type=raster.z_data_type or 'depth',
                inversion_reference=raster.z_inversion_reference
            )
            
            # Update the raster with filled result
            raster.z_channel = z_result
            raster.z_unit = 'meters'
            # z_data_type and z_inversion_reference are preserved

        except ValueError as e:
            # Handle errors from smart_fill_z_channel
            error_msg = str(e)
            print(f"Smart Fill error: {error_msg}")
            
            if "only works with float32" in error_msg:
                QMessageBox.warning(self.annotation_window, 
                                    "Incompatible Data Type", 
                                    "Smart Fill only works with float32 depth data, not uint8.")
            elif "No valid" in error_msg or "Insufficient" in error_msg:
                QMessageBox.information(self.annotation_window,
                                        "No Gaps to Fill" if "No valid" in error_msg else "Insufficient Data",
                                        error_msg)
            elif "requires inversion reference" in error_msg:
                QMessageBox.warning(self.annotation_window,
                                    "Missing Reference",
                                    "Elevation data is missing inversion reference. Cannot perform Smart Fill.")
            else:
                QMessageBox.warning(self.annotation_window, 
                                    "Smart Fill Error", 
                                    error_msg)
            return
        except Exception as e:
            print(f"Unexpected error in Smart Fill: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.annotation_window,
                                 "Smart Fill Error",
                                 f"An unexpected error occurred: {e}")
            return
