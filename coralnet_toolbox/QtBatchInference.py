import warnings
import os
from itertools import groupby
from operator import attrgetter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox,
                             QFormLayout, QComboBox, QHBoxLayout, QCheckBox, QButtonGroup)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.Common import ThresholdsWidget
from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """
    Consolidated batch inference dialog for all models.
    Supports: Classify, Detect, Segment, Semantic, SAM, SeeAnything, Transformers.
    
    This dialog provides:
    - Model selection dropdown for all loaded models
    - ThresholdWidget for configurable inference thresholds
    - Task-specific options through subclassing
    - Images are selected through ImageWindow context menu (right-click)

    :param main_window: MainWindow object
    :param parent: Parent widget
    :param highlighted_images: List of image paths to process (required)
    """
    def __init__(self, main_window, parent=None, highlighted_images=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Batch Inference")
        self.resize(500, 400)

        # Initialize references to various deployment dialogs
        self.classify_dialog = getattr(main_window, 'classify_deploy_model_dialog', None)
        self.detect_dialog = getattr(main_window, 'detect_deploy_model_dialog', None)
        self.segment_dialog = getattr(main_window, 'segment_deploy_model_dialog', None)
        self.semantic_dialog = getattr(main_window, 'semantic_deploy_model_dialog', None)
        self.sam_dialog = getattr(main_window, 'sam_deploy_generator_dialog', None)
        self.see_anything_dialog = getattr(main_window, 'see_anything_deploy_generator_dialog', None)
        self.transformers_dialog = getattr(main_window, 'transformers_deploy_model_dialog', None)

        # Dictionary to store available model dialogs
        self.model_dialogs = {}
        self.model_keys = []
        self.loaded_model = None
        self.current_selected_model = None  # Track the current selected model

        # Storage for image paths and annotations
        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []
        
        # Store highlighted images if provided
        self.highlighted_images = highlighted_images if highlighted_images else []

        self.layout = QVBoxLayout(self)

        # Setup layouts in order
        self.setup_info_layout()
        self.setup_models_layout()
        self.setup_inference_layout()
        self.setup_thresholds_layout()
        self.setup_task_specific_layout()
        self.setup_buttons_layout()

    def showEvent(self, event):
        """
        Update model availability when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.update_status_label()
        self.update_model_availability()
        
        # Check if any models are available
        if not self.model_dialogs:
            QMessageBox.warning(self, 
                                "No Models Available", 
                                "Please load a model before opening batch inference.")
            self.reject()
            return
        
        # Untoggle all tools in the annotation window
        self.annotation_window.toolChanged.emit(None)
        
        if hasattr(self, 'thresholds_widget'):
            self.thresholds_widget.initialize_thresholds()

    def setup_info_layout(self):
        """
        Set up the info layout with explanatory text.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Perform batch inferencing on the selected images. "
            "It is highly recommended to save the current project before proceeding."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Set up the model selection dropdown.
        """
        group_box = QGroupBox("Select Model")
        form_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        form_layout.addRow("Model:", self.model_combo)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def setup_inference_layout(self):
        """
        Set up the inference type selection dropdown.
        """
        group_box = QGroupBox("Inference Type")
        form_layout = QFormLayout()

        self.inference_type_combo = QComboBox()
        self.inference_type_combo.addItem("Standard")
        self.inference_type_combo.addItem("Tiled")
        self.inference_type_combo.currentTextChanged.connect(self.on_inference_type_changed)
        form_layout.addRow("Type:", self.inference_type_combo)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def setup_thresholds_layout(self):
        """
        Set up the ThresholdWidget for configurable thresholds.
        All thresholds are shown, but specific ones are disabled based on model type.
        Can be overridden by subclasses to configure which thresholds to show.
        """
        # Show all thresholds - they will be enabled/disabled based on model selection
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True,
            title="Thresholds"
        )
        self.layout.addWidget(self.thresholds_widget)

    def setup_task_specific_layout(self):
        """
        Override in subclasses to add task-specific options.
        For Classify model, adds annotation selection options (review vs all).
        Visibility is controlled by on_model_changed().
        """
        # Create a group box for Classify-specific annotation options
        group_box = QGroupBox("Annotation Options")
        layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        annotation_options_group = QButtonGroup(self)

        # Add the checkboxes to the button group
        self.review_checkbox = QCheckBox("Predict Review Annotation")
        self.all_checkbox = QCheckBox("Predict All Annotations")
        annotation_options_group.addButton(self.review_checkbox)
        annotation_options_group.addButton(self.all_checkbox)

        # Ensure only one checkbox can be checked at a time
        annotation_options_group.setExclusive(True)
        # Set the default checkbox
        self.review_checkbox.setChecked(True)

        # Build the annotation layout
        layout.addWidget(self.review_checkbox)
        layout.addWidget(self.all_checkbox)

        group_box.setLayout(layout)
        self.task_specific_group = group_box
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the dialog buttons with status label.
        """
        # Create a horizontal layout for buttons and status label
        button_layout = QHBoxLayout()
        
        # Status label on the left
        self.status_label = QLabel()
        self.update_status_label()
        button_layout.addWidget(self.status_label)
        
        # Stretch to push buttons to the right
        button_layout.addStretch()
        
        # Buttons on the right
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        
        self.layout.addLayout(button_layout)

    def update_status_label(self):
        """
        Update the status label to show the number of images for batch inference.
        """
        num_images = len(self.highlighted_images)
        if num_images == 0:
            self.status_label.setText("No images selected")
        elif num_images == 1:
            self.status_label.setText("1 image selected")
        else:
            self.status_label.setText(f"{num_images} images selected")

    def update_status_label_for_tiled(self):
        """
        Update the status label to show the number of images and work areas for Tiled mode.
        Counts work areas across all selected images.
        """
        num_images = len(self.highlighted_images)
        total_work_areas = 0

        # Count total work areas across all selected images
        if self.image_window and hasattr(self.image_window, 'raster_manager'):
            for image_path in self.highlighted_images:
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster and hasattr(raster, 'work_areas'):
                    total_work_areas += len(raster.work_areas)

        # Format the status text
        if num_images == 0:
            self.status_label.setText("No images selected")
        elif num_images == 1:
            if total_work_areas == 0:
                self.status_label.setText("1 image, no work areas")
            elif total_work_areas == 1:
                self.status_label.setText("1 image, 1 work area")
            else:
                self.status_label.setText(f"1 image, {total_work_areas} work areas")
        else:
            if total_work_areas == 0:
                self.status_label.setText(f"{num_images} images, no work areas")
            elif total_work_areas == 1:
                self.status_label.setText(f"{num_images} images, 1 work area")
            else:
                self.status_label.setText(f"{num_images} images, {total_work_areas} work areas")

    def update_model_availability(self):
        """
        Check which models are loaded and populate the model dialog dictionary.
        """
        self.model_dialogs = {}

        if self.classify_dialog and getattr(self.classify_dialog, "loaded_model", None):
            self.model_dialogs["Classify"] = self.classify_dialog
        if self.detect_dialog and getattr(self.detect_dialog, "loaded_model", None):
            self.model_dialogs["Detect"] = self.detect_dialog
        if self.segment_dialog and getattr(self.segment_dialog, "loaded_model", None):
            self.model_dialogs["Segment"] = self.segment_dialog
        if self.semantic_dialog and getattr(self.semantic_dialog, "loaded_model", None):
            self.model_dialogs["Semantic"] = self.semantic_dialog
        if self.sam_dialog and getattr(self.sam_dialog, "loaded_model", None):
            self.model_dialogs["SAM"] = self.sam_dialog
        if self.see_anything_dialog and getattr(self.see_anything_dialog, "loaded_model", None):
            self.model_dialogs["See Anything"] = self.see_anything_dialog
        if self.transformers_dialog and getattr(self.transformers_dialog, "loaded_model", None):
            self.model_dialogs["Transformers"] = self.transformers_dialog

        self.update_model_combo()

    def update_model_combo(self):
        """
        Update the model dropdown with available models.
        Preserves the last selected model if it's still available.
        """
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_keys = []

        for key in sorted(self.model_dialogs.keys()):
            self.model_combo.addItem(key)
            self.model_keys.append(key)

        # Try to restore the current selected model, otherwise default to index 0
        selected_index = 0
        if self.current_selected_model and self.current_selected_model in self.model_keys:
            selected_index = self.model_keys.index(self.current_selected_model)
        
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(selected_index)

        self.model_combo.blockSignals(False)
        self.update_loaded_model()
        
        # Manually trigger the UI update for the selected model
        if self.model_combo.count() > 0:
            self.on_model_changed(selected_index)

    def update_loaded_model(self):
        """
        Update the loaded_model based on the current selection.
        """
        idx = self.model_combo.currentIndex()
        if 0 <= idx < len(self.model_keys):
            key = self.model_keys[idx]
            self.loaded_model = self.model_dialogs.get(key, None)
        else:
            self.loaded_model = None

    def on_model_changed(self, index):
        """
        Handle model selection change and update thresholds accordingly.
        
        :param index: Index of the selected model
        """
        # Update the loaded model
        self.update_loaded_model()
        
        # Update threshold visibility based on selected model
        if 0 <= index < len(self.model_keys):
            selected_model = self.model_keys[index]
            # Save the selected model for later
            self.current_selected_model = selected_model
            self.update_thresholds_for_model(selected_model)
            
            # Enable annotation options only for Classify model
            if selected_model == "Classify":
                self.task_specific_group.setEnabled(True)
                # Disable inference type dropdown for Classify model
                self.inference_type_combo.setEnabled(False)
                self.inference_type_combo.setCurrentText("Standard")
            else:
                self.task_specific_group.setEnabled(False)
                # Enable inference type dropdown for other models
                self.inference_type_combo.setEnabled(True)
    
    def on_inference_type_changed(self, inference_type):
        """
        Handle inference type change and update the annotation window tool accordingly.
        Also updates the status label to show work area information for Tiled mode.
        
        :param inference_type: The selected inference type ("Standard" or "Tiled")
        """
        if inference_type == "Tiled":
            # Set the annotation window tool to work_area
            self.annotation_window.toolChanged.emit("work_area")
            # Update status label to show images and work areas
            self.update_status_label_for_tiled()
        else:
            # Set the annotation window tool to None (untoggle all tools)
            self.annotation_window.toolChanged.emit(None)
            # Reset status label to show only images
            self.update_status_label()
    
    def update_thresholds_for_model(self, model_name):
        """
        Update the enabled state of thresholds based on model type.
        All thresholds remain visible, but specific ones are disabled.
        
        :param model_name: Name of the selected model
        """
        # Classify and Semantic only use uncertainty threshold
        if model_name in ("Classify", "Semantic"):
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=True,
                enable_iou=False,
                enable_area=False
            )
        # Detect and Segment use max_detections, uncertainty, iou, and area
        elif model_name in ("Detect", "Segment"):
            self.configure_thresholds(
                enable_max_detections=True,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # SAM uses uncertainty only
        elif model_name == "SAM":
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # See Anything uses uncertainty only
        elif model_name == "See Anything":
            self.configure_thresholds(
                enable_max_detections=False,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
        # Transformers uses uncertainty, iou, and area
        elif model_name == "Transformers":
            self.configure_thresholds(
                enable_max_detections=True,
                enable_uncertainty=True,
                enable_iou=True,
                enable_area=True
            )
    
    def configure_thresholds(self, enable_max_detections, enable_uncertainty,
                             enable_iou, enable_area):
        """
        Configure which thresholds are enabled/disabled.
        All thresholds remain visible, but specific ones are disabled and greyed out.
        
        :param enable_max_detections: Whether to enable max detections
        :param enable_uncertainty: Whether to enable uncertainty threshold
        :param enable_iou: Whether to enable IoU threshold
        :param enable_area: Whether to enable area threshold
        """
        # Helper function to get all QLabel widgets from the form layout
        def get_form_labels_for_widget(widget):
            """Find and return all QLabel widgets associated with a control in the form layout."""
            layout = self.thresholds_widget.layout()
            labels = []
            if layout and isinstance(layout, QFormLayout):
                for row in range(layout.rowCount()):
                    label_item = layout.itemAt(row, QFormLayout.LabelRole)
                    widget_item = layout.itemAt(row, QFormLayout.FieldRole)
                    
                    if widget_item and widget_item.widget() == widget:
                        if label_item and label_item.widget():
                            labels.append(label_item.widget())
            return labels
        
        # Enable/disable max detections
        if hasattr(self.thresholds_widget, 'max_detections_spinbox'):
            spinbox = self.thresholds_widget.max_detections_spinbox
            spinbox.setEnabled(enable_max_detections)
            # Find and disable associated labels
            labels = get_form_labels_for_widget(spinbox)
            for label in labels:
                label.setEnabled(enable_max_detections)
        
        # Enable/disable uncertainty threshold
        if hasattr(self.thresholds_widget, 'uncertainty_threshold_slider'):
            slider = self.thresholds_widget.uncertainty_threshold_slider
            slider.setEnabled(enable_uncertainty)
            value_label = getattr(self.thresholds_widget, 'uncertainty_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_uncertainty)
            # Find and disable associated title labels
            labels = get_form_labels_for_widget(slider)
            for label in labels:
                label.setEnabled(enable_uncertainty)
        
        # Enable/disable IoU threshold
        if hasattr(self.thresholds_widget, 'iou_threshold_slider'):
            slider = self.thresholds_widget.iou_threshold_slider
            slider.setEnabled(enable_iou)
            value_label = getattr(self.thresholds_widget, 'iou_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_iou)
            # Find and disable associated title labels
            labels = get_form_labels_for_widget(slider)
            for label in labels:
                label.setEnabled(enable_iou)
        
        # Enable/disable area threshold
        if hasattr(self.thresholds_widget, 'area_threshold_min_slider'):
            min_slider = self.thresholds_widget.area_threshold_min_slider
            max_slider = self.thresholds_widget.area_threshold_max_slider
            min_slider.setEnabled(enable_area)
            max_slider.setEnabled(enable_area)
            value_label = getattr(self.thresholds_widget, 'area_threshold_label', None)
            if value_label:
                value_label.setEnabled(enable_area)
            # Find and disable associated title labels
            min_labels = get_form_labels_for_widget(min_slider)
            max_labels = get_form_labels_for_widget(max_slider)
            for label in min_labels + max_labels:
                label.setEnabled(enable_area)

    def check_model_availability(self):
        """
        Check if a model is loaded and available for batch inference.

        :return: True if a model is loaded, False otherwise
        """
        self.update_model_availability()
        return self.loaded_model is not None

    def preprocess_annotations(self):
        """
        Get annotations based on user selection and preprocess them (Classify only).
        Groups annotations by image and crops them using concurrent processing.
        """
        # Get the annotations based on user selection
        if self.review_checkbox.isChecked():
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Check if annotations need to be cropped
        annotations_to_crop = []
        for annotation in self.annotations:
            if hasattr(annotation, 'cropped_image') and annotation.cropped_image:
                # Annotation already has cropped image, add to prepared patches
                self.prepared_patches.append(annotation)
            else:
                # Annotation needs to be cropped
                annotations_to_crop.append(annotation)

        # Only crop annotations that need cropping
        if annotations_to_crop:
            self.bulk_preprocess_patch_annotations(annotations_to_crop)

    def bulk_preprocess_patch_annotations(self, annotations_to_crop=None):
        """
        Bulk preprocess patch annotations by cropping the images concurrently.
        Uses ThreadPoolExecutor for parallel processing.

        Args:
            annotations_to_crop: List of annotations that need to be cropped.
                                If None, uses self.annotations.
        """
        if annotations_to_crop is None:
            annotations_to_crop = self.annotations

        if not annotations_to_crop:
            return

        # Get unique image paths for annotations that need cropping
        crop_image_paths = list(set(a.image_path for a in annotations_to_crop))

        # Create progress bar for cropping
        progress_bar = ProgressBar(self.annotation_window, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(crop_image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations_to_crop, key=attrgetter('image_path')),
                                      key=attrgetter('image_path'))

        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                # Dictionary to track futures and their corresponding image paths
                futures = {}

                # Process each group of annotations by image path
                for image_path, group in grouped_annotations:
                    # Convert group iterator to list for reuse
                    image_annotations = list(group)

                    # Submit cropping task asynchronously for each image
                    # Returns a Future object representing pending execution
                    future = executor.submit(self.annotation_window.crop_annotations,
                                             image_path,
                                             image_annotations,
                                             verbose=False)

                    # Store image path for each future for error reporting
                    futures[future] = image_path

                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        # Get cropped patches from completed task
                        cropped = future.result()
                        # Add cropped patches to prepared patches list
                        self.prepared_patches.extend(cropped)
                    except Exception as exc:
                        print(f"{futures[future]} generated an exception: {exc}")
                    finally:
                        # Update progress bar after each image is processed
                        progress_bar.update_progress()

        except Exception as e:
            print(f"Error in bulk preprocessing: {e}")

        finally:
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def get_selected_image_paths(self):
        """
        Get the selected image paths.
        Images are provided at initialization through the highlighted_images parameter.

        :return: List of selected image paths
        """
        # Return the highlighted images provided at initialization
        return self.highlighted_images

    def apply(self):
        """
        Apply the selected batch inference options.
        For Classify, runs preprocessing first to crop annotations.
        """
        if not self.check_model_availability():
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            self.image_paths = self.get_selected_image_paths()
            if not self.image_paths:
                QMessageBox.warning(self, "No Images", "No images selected for inference.")
                return

            # Get the selected model type
            idx = self.model_combo.currentIndex()
            if 0 <= idx < len(self.model_keys):
                selected_model = self.model_keys[idx]
                
                # Verify the model is available
                if selected_model not in self.model_dialogs:
                    QMessageBox.warning(self, 
                                        "Model Not Available",
                                        f"{selected_model} model is not loaded.")
                    return

                # For Classify, preprocess annotations first
                if selected_model == "Classify":
                    self.preprocess_annotations()

            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete batch inference: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            self.cleanup()

    def batch_inference(self):
        """
        Perform batch inference on selected images based on the selected model type.
        Routes to appropriate predict method with correct parameters for each model.
        """
        # Determine the selected model type
        idx = self.model_combo.currentIndex()
        if idx < 0 or idx >= len(self.model_keys):
            raise ValueError("No model selected")

        selected_model = self.model_keys[idx]
        # Get the correct model dialog from the dictionary based on selected model
        model_dialog = self.model_dialogs.get(selected_model, None)

        if model_dialog is None:
            raise ValueError(f"No model loaded for {selected_model}")

        progress_bar = ProgressBar(self.annotation_window, title="Batch Inference")
        progress_bar.show()

        try:
            # Classify: predict on grouped annotation patches
            if selected_model == "Classify":
                if not self.prepared_patches:
                    # No annotations to process, silently return
                    progress_bar.finish_progress()
                    progress_bar.stop_progress()
                    progress_bar.close()
                    return

                # Group annotations by image path
                groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')),
                                 key=attrgetter('image_path'))

                # Count number of unique image paths
                num_paths = len(set(a.image_path for a in self.prepared_patches))

                # Make predictions on each image's annotations
                for idx_path, (path, patches) in enumerate(groups):
                    try:
                        progress_bar.set_title(f"Predicting: {idx_path + 1}/{num_paths} - {os.path.basename(path)}")
                        model_dialog.predict(inputs=list(patches), progress_bar=progress_bar)
                    except Exception as e:
                        print(f"Failed to make predictions on {path}: {e}")
                        continue

            # Detect, Segment, Semantic: predict on image paths
            elif selected_model in ("Detect", "Segment", "Semantic"):
                model_dialog.predict(self.image_paths, progress_bar)

            # SAM, See Anything: predict on image paths (using deploy_generator_dialog)
            elif selected_model in ("SAM", "See Anything"):
                model_dialog.predict(self.image_paths, progress_bar)

            # Transformers: predict on image paths
            elif selected_model == "Transformers":
                model_dialog.predict(self.image_paths, progress_bar)

            else:
                raise ValueError(f"Unknown model type: {selected_model}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete batch inference: {str(e)}")
        finally:
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def cleanup(self):
        """
        Clean up resources after batch inference.
        """
        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []
        
        # Reset inference type to Standard
        self.inference_type_combo.blockSignals(True)
        self.inference_type_combo.setCurrentText("Standard")
        self.inference_type_combo.blockSignals(False)
        
        # Untoggle all tools in the annotation window
        self.annotation_window.toolChanged.emit(None)
