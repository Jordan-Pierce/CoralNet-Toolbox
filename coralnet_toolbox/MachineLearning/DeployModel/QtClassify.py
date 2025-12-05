import warnings

import gc
import os
from copy import deepcopy

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from torch.cuda import empty_cache
from ultralytics import YOLO

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.Results import ResultsProcessor

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Classification Model (Ctrl + 1)")

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        # Initialize thresholds in the widget
        self.thresholds_widget.initialize_thresholds()

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        # Currently no parameters other than thresholds for classification
        pass
    
    def setup_sam_layout(self):
        pass

    def setup_thresholds_layout(self):
        """
        Setup threshold control section using the reusable ThresholdsWidget.
        """
        # Create the thresholds widget with only uncertainty threshold enabled
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=False,
            show_uncertainty=True,
            show_iou=False,
            show_area=False
        )
        self.layout.addWidget(self.thresholds_widget)

    def load_model(self):
        """
        Load the classification model.
        """
        self.task = 'classify'
        
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # TODO: Improve batch size handling for different model types
            # Set BATCH_SIZE based on model type.
            # .engine models require a fixed batch size (usually 1)
            if self.model_path.endswith('.engine'):
                self.BATCH_SIZE = 1
            else:
                self.BATCH_SIZE = 0

            # Load the model (8.3.141) YOLO handles RTDETR too
            self.loaded_model = YOLO(self.model_path, task=self.task)

            try:
                imgsz = self.loaded_model.__dict__['overrides']['imgsz']
            except:
                imgsz = 256

            self.loaded_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            self.class_names = list(self.loaded_model.names.values())

            # Check for unmapped classes
            mapped_classes, unmapped_classes, unused_mapping_keys = self._find_unmapped_classes()

            # Handle class mapping (complete or partial)
            if not self.class_mapping:
                # No mapping file at all
                self.handle_missing_class_mapping()
            elif unmapped_classes:
                # Partial mapping - some classes are missing
                self.add_labels_to_label_window()
                self.handle_missing_class_mapping(unmapped_classes)
            else:
                # Complete mapping - all classes are mapped
                self.add_labels_to_label_window()

            # Display the class names
            self.check_and_display_class_names()

            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except RuntimeError:
            # Model load was cancelled by user
            self.loaded_model = None
            self.class_names = []
            self.auto_created_labels = set()
            QApplication.restoreOverrideCursor()
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def predict(self, inputs=None, progress_bar=None):
        """
        Predict the classification results for the given inputs.
        
        Args:
            inputs: List of annotations to predict on. If None, uses selected or all review annotations.
            progress_bar: Optional progress bar instance to use. If None, no progress bar is shown.
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the selected annotation
            inputs = self.annotation_window.selected_annotations.copy()
            # Unselect the annotations (regardless)
            self.annotation_window.unselect_annotations()

        if not inputs:
            # If no annotations are selected, predict all annotations in the image
            inputs = self.annotation_window.get_image_review_annotations()

        if not inputs:
            # If no annotations are available, return
            QApplication.restoreOverrideCursor()
            return

        # Create lists to store valid images and their corresponding annotations
        images_np = []
        valid_inputs = []

        # Crop annotations on-demand if needed and convert to numpy arrays
        for annotation in inputs:
            # Crop on-demand if not already cropped
            if not annotation.cropped_image:
                try:
                    # Get the rasterio source for this annotation's image
                    raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
                    if raster and raster.rasterio_src:
                        annotation.create_cropped_image(raster.rasterio_src)
                except Exception as e:
                    print(f"Error cropping annotation {annotation.id}: {str(e)}")
                    continue
            
            # Convert cropped image to numpy array
            if annotation.cropped_image:
                try:
                    img = pixmap_to_numpy(annotation.cropped_image)
                    images_np.append(img)
                    valid_inputs.append(annotation)
                except Exception as e:
                    print(f"Error converting pixmap to numpy: {str(e)}")
                    continue

        # Only proceed if we have valid images to process
        if images_np:            
            if not self.BATCH_SIZE:
                # Predict the classification results
                results = self.loaded_model(images_np,
                                            conf=self.thresholds_widget.get_uncertainty_thresh(),
                                            device=self.main_window.device,
                                            half=True,
                                            stream=True)
                
            else:  # process one by one
                results = []
                for _ in range(len(images_np)):
                    result = self.loaded_model(images_np[_],
                                               conf=self.thresholds_widget.get_uncertainty_thresh(),
                                               device=self.main_window.device,
                                               half=True)
                    if result:  # Ensure the result list is not empty
                        results.append(deepcopy(result[0]))  # Append the Results object itself

            # Create a result processor
            results_processor = ResultsProcessor(self.main_window,
                                                 self.class_mapping)

            # Process the classification results using the valid inputs
            # Pass the progress_bar parameter to avoid creating nested progress bars
            results_processor.process_classification_results(results, valid_inputs, progress_bar=progress_bar)

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()
