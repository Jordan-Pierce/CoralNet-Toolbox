import os
import warnings
import collections
import xml.etree.ElementTree as ET

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog, QGroupBox, 
                             QFormLayout, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Metashape Data Structures
# ----------------------------------------------------------------------------------------------------------------------

Sensor = collections.namedtuple("Sensor", ["id", "label", "width", "height", "calibration"])
Camera = collections.namedtuple("Camera", ["id", "sensor_id", "label", "transform"])
Calibration = collections.namedtuple("Calibration", ["f", "cx", "cy", "fx", "fy"])


# ----------------------------------------------------------------------------------------------------------------------
# Metashape I/O Functions
# ----------------------------------------------------------------------------------------------------------------------


def read_metashape_xml(xml_path):
    """Read Metashape cameras and sensors from XML file.
    
    Args:
        xml_path: Path to cameras.xml file
    
    Returns:
        Tuple of (sensors_dict, cameras_dict):
        - sensors_dict: Dictionary of Sensor objects keyed by sensor_id
        - cameras_dict: Dictionary of Camera objects keyed by camera_id
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    sensors = {}
    cameras = {}
    
    # Parse sensors
    sensors_element = root.find('.//sensors')
    if sensors_element is not None:
        for sensor_elem in sensors_element.findall('sensor'):
            sensor_id = int(sensor_elem.get('id'))
            label = sensor_elem.get('label', 'unknown')
            
            # Get resolution
            resolution = sensor_elem.find('resolution')
            if resolution is not None:
                width = int(resolution.get('width'))
                height = int(resolution.get('height'))
            else:
                width = height = 0
            
            # Get calibration parameters
            calibration_elem = sensor_elem.find('calibration')
            calibration = None
            if calibration_elem is not None:
                f_elem = calibration_elem.find('f')
                cx_elem = calibration_elem.find('cx')
                cy_elem = calibration_elem.find('cy')
                fx_elem = calibration_elem.find('fx')
                fy_elem = calibration_elem.find('fy')
                
                f = float(f_elem.text) if f_elem is not None else None
                cx = float(cx_elem.text) if cx_elem is not None else None
                cy = float(cy_elem.text) if cy_elem is not None else None
                fx = float(fx_elem.text) if fx_elem is not None else None
                fy = float(fy_elem.text) if fy_elem is not None else None
                
                calibration = Calibration(f=f, cx=cx, cy=cy, fx=fx, fy=fy)
            
            sensors[sensor_id] = Sensor(
                id=sensor_id,
                label=label,
                width=width,
                height=height,
                calibration=calibration
            )
    
    # Parse cameras
    cameras_element = root.find('.//cameras')
    if cameras_element is not None:
        for camera_elem in cameras_element.findall('camera'):
            camera_id = int(camera_elem.get('id'))
            sensor_id = int(camera_elem.get('sensor_id'))
            label = camera_elem.get('label', '')
            
            # Get transform (4x4 matrix in row-major format as space-separated string)
            transform_elem = camera_elem.find('transform')
            transform = None
            if transform_elem is not None and transform_elem.text:
                try:
                    values = [float(v) for v in transform_elem.text.split()]
                    if len(values) == 16:
                        # Reshape to 4x4 matrix (row-major)
                        transform = np.array(values).reshape(4, 4)
                except (ValueError, AttributeError):
                    pass
            
            cameras[camera_id] = Camera(
                id=camera_id,
                sensor_id=sensor_id,
                label=label,
                transform=transform
            )
    
    return sensors, cameras


def extract_intrinsics_extrinsics_from_metashape(sensors, cameras):
    """Extract intrinsics and extrinsics from Metashape sensors and cameras dicts.
    
    Handles sensor calibration extraction and camera-to-world transform inversion.
    Corrects coordinate system from Metashape to COLMAP/OpenCV (flips Y and Z).
    
    Args:
        sensors: Dictionary of Sensor objects from Metashape
        cameras: Dictionary of Camera objects from Metashape
    
    Returns:
        Tuple of (intrinsics, extrinsics, camera_labels) as numpy arrays and list:
        - intrinsics: (N, 3, 3) camera calibration matrices
        - extrinsics: (N, 4, 4) world-to-camera transformation matrices
        - camera_labels: List of camera labels in same order
    """
    intrinsics_list = []
    extrinsics_list = []
    camera_labels = []
    
    # Transformation matrix to flip Y and Z axes (Metashape -> OpenCV/COLMAP)
    # This corresponds to a 180-degree rotation around the X-axis
    # Multiplied on the right of the c2w matrix to transform the camera basis
    T_metashape_to_opencv = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    
    for cam_id, cam in cameras.items():
        # Skip cameras without transforms
        if cam.transform is None:
            continue
        
        # Get sensor calibration
        sensor = sensors.get(cam.sensor_id)
        if sensor is None or sensor.calibration is None:
            continue
        
        calib = sensor.calibration
        
        # Extract focal lengths - prefer fx/fy, fallback to f
        if calib.fx is not None and calib.fy is not None:
            fx = calib.fx
            fy = calib.fy
        elif calib.f is not None:
            fx = fy = calib.f
        else:
            continue  # No focal length available
        
        # Extract principal point
        image_center_x = sensor.width / 2.0
        image_center_y = sensor.height / 2.0
        
        if calib.cx is not None and calib.cy is not None:
            # Metashape: cx, cy are offsets from center, convert to absolute coordinates
            cx = image_center_x + calib.cx
            cy = image_center_y + calib.cy
        else:
            cx = image_center_x
            cy = image_center_y
        
        # Build intrinsics matrix K
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics_list.append(K)
        
        # Calculate Extrinsics
        try:
            # 1. Get Camera-to-World (c2w) from Metashape
            c2w_metashape = cam.transform
            
            # 2. Apply coordinate conversion: c2w_opencv = c2w_metashape @ T_flip
            # We multiply on the right to transform the camera's local axes 
            # relative to the world, preserving the camera center position.
            c2w_opencv = c2w_metashape @ T_metashape_to_opencv
            
            # 3. Invert to get World-to-Camera (w2c)
            w2c = np.linalg.inv(c2w_opencv)
            extrinsics_list.append(w2c)
            
        except np.linalg.LinAlgError:
            intrinsics_list.pop()  # Remove the intrinsics we just added
            continue
        
        camera_labels.append(cam.label)
    
    if len(intrinsics_list) == 0:
        return np.array([]), np.array([]), []
    
    return np.array(intrinsics_list), np.array(extrinsics_list), camera_labels


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportMetashapeCameras(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        
        self.setWindowTitle("Import Metashape Camera Parameters")
        self.resize(700, 120)
        
        # Setup the file selection layout
        self.setup_file_selection_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
    
    def setup_file_selection_layout(self):
        """Setup the layout for selecting Metashape XML file."""
        group_box = QGroupBox("Metashape File")
        layout = QFormLayout()
        
        # XML file
        self.xml_file_edit = QLineEdit()
        self.xml_browse_button = QPushButton("Browse...")
        self.xml_browse_button.clicked.connect(self.browse_xml_file)
        xml_layout = QHBoxLayout()
        xml_layout.addWidget(self.xml_file_edit)
        xml_layout.addWidget(self.xml_browse_button)
        layout.addRow("Cameras XML File:", xml_layout)
        
        group_box.setLayout(layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box)
        self.setLayout(main_layout)
    
    def setup_buttons_layout(self):
        """Setup the layout for the import and cancel buttons."""
        layout = self.layout()
        button_layout = QHBoxLayout()
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.import_cameras)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def browse_xml_file(self):
        """Open a file dialog to select the cameras XML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Metashape Cameras XML File",
            "",
            "XML Files (*.xml);;All Files (*)",
            options=options
        )
        if file_path:
            self.xml_file_edit.setText(file_path)
    
    def import_cameras(self):
        """Import camera intrinsics and extrinsics from Metashape XML file into the current project."""
        # Get file path from the text field
        xml_file = self.xml_file_edit.text()
        
        # Validate that file is selected
        if not xml_file:
            QMessageBox.warning(
                self,
                "Missing File",
                "Please select a cameras XML file."
            )
            return
        
        # Validate that file exists
        if not os.path.exists(xml_file):
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected XML file does not exist:\n{xml_file}"
            )
            return
        
        # Check if images are loaded
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(
                self,
                "No Images Loaded",
                "Please load images into the project before importing camera parameters."
            )
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Read Metashape XML file
            sensors, cameras = read_metashape_xml(xml_file)
            
            if not sensors:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self,
                    "No Sensors Found",
                    "No sensor calibration data found in the XML file."
                )
                return
            
            if not cameras:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self,
                    "No Cameras Found",
                    "No camera data found in the XML file."
                )
                return
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Error Reading XML",
                f"Failed to read Metashape XML file:\n{str(e)}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        
        # Match Metashape camera labels to loaded rasters
        # Camera labels are image basenames (without extension)
        image_path_map = {os.path.splitext(os.path.basename(path))[0]: path 
                         for path in self.image_window.raster_manager.image_paths}
        matched_cameras = {}
        
        for cam_id, cam in cameras.items():
            # Match by basename without extension
            if cam.label in image_path_map:
                matched_cameras[cam_id] = cam
        
        if not matched_cameras:
            QMessageBox.warning(
                self,
                "No Matching Images",
                "No camera labels from the XML file match the loaded image basenames.\n\n"
                "Camera labels should match image filenames (without extension)."
            )
            return
        
        # Check for existing camera parameters in matched rasters
        rasters_with_cameras = []
        for cam_id, cam in matched_cameras.items():
            image_basename = cam.label
            image_path = image_path_map[image_basename]
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and (raster.intrinsics is not None or raster.extrinsics is not None):
                rasters_with_cameras.append(image_basename)
        
        # Alert user if existing camera data found
        if rasters_with_cameras:
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Camera Data?",
                f"Found existing camera parameters for {len(rasters_with_cameras)} image(s).\n\n"
                "Do you want to overwrite them with data from the Metashape XML file?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # Extract intrinsics and extrinsics
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing Metashape Camera Parameters")
        progress_bar.show()
        progress_bar.start_progress(len(matched_cameras))
        
        updated_count = 0
        skipped_count = 0
        
        try:
            # Extract all intrinsics and extrinsics
            intrinsics_all, extrinsics_all, camera_labels = extract_intrinsics_extrinsics_from_metashape(
                sensors, cameras
            )
            
            if len(intrinsics_all) == 0:
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self,
                    "No Valid Camera Data",
                    "No valid camera calibration or transform data found in the XML file."
                )
                return
            
            # Create label to index mapping
            label_to_idx = {label: idx for idx, label in enumerate(camera_labels)}
            
            # Update matched rasters
            for cam_id, cam in matched_cameras.items():
                image_basename = cam.label
                
                # Skip if not in extracted data
                if image_basename not in label_to_idx:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                
                idx = label_to_idx[image_basename]
                intrinsics = intrinsics_all[idx]
                extrinsics = extrinsics_all[idx]
                
                # Get the raster
                image_path = image_path_map[image_basename]
                raster = self.image_window.raster_manager.get_raster(image_path)
                
                if raster is None:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                
                # Set intrinsics and extrinsics
                try:
                    # Add or update camera parameters
                    if raster.intrinsics is not None or raster.extrinsics is not None:
                        raster.update_intrinsics(intrinsics)
                        raster.update_extrinsics(extrinsics)
                    else:
                        raster.add_intrinsics(intrinsics)
                        raster.add_extrinsics(extrinsics)
                    updated_count += 1
                except Exception as e:
                    print(f"Error setting camera parameters for {image_basename}: {e}")
                    skipped_count += 1
                
                progress_bar.update_progress()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"An error occurred during import:\n{str(e)}"
            )
        
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()
        
        # Show summary
        summary_msg = f"Import complete!\n\n"
        summary_msg += f"Updated: {updated_count} image(s)\n"
        if skipped_count > 0:
            summary_msg += f"Skipped: {skipped_count} image(s)"
        
        QMessageBox.information(
            self,
            "Import Complete",
            summary_msg
        )
        
        # Close dialog
        self.accept()
