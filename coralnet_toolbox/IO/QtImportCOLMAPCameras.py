import warnings


import os
import struct
import collections

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog, QGroupBox, 
                             QFormLayout, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# COLMAP Data Structures
# ----------------------------------------------------------------------------------------------------------------------

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in CAMERA_MODELS}


# ----------------------------------------------------------------------------------------------------------------------
# COLMAP I/O Functions
# ----------------------------------------------------------------------------------------------------------------------


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    
    Args:
        fid: File handle
        num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        endian_character: Any of {@, =, <, >, !}
    
    Returns:
        Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """Read COLMAP cameras from text file.
    
    See: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    
    Args:
        path: Path to cameras.txt file
    
    Returns:
        Dictionary of Camera objects keyed by camera_id
    """
    cameras = {}
    with open(path) as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """Read COLMAP cameras from binary file.
    
    See: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    
    Args:
        path_to_model_file: Path to cameras.bin file
    
    Returns:
        Dictionary of Camera objects keyed by camera_id
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """Read COLMAP images from text file.
    
    See: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    
    Args:
        path: Path to images.txt file
    
    Returns:
        Dictionary of Image objects keyed by image_id
    """
    images = {}
    with open(path) as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """Read COLMAP images from binary file.
    
    See: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    
    Args:
        path_to_model_file: Path to images.bin file
    
    Returns:
        Dictionary of Image objects keyed by image_id
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix.
    
    Args:
        qvec: Quaternion as numpy array [qw, qx, qy, qz]
    
    Returns:
        3x3 rotation matrix as numpy array
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def extract_intrinsics_extrinsics_from_colmap(cameras, images):
    """Extract intrinsics and extrinsics from COLMAP cameras and images dicts.
    
    Handles multiple camera models (PINHOLE, SIMPLE_PINHOLE, OPENCV, etc.).
    Extrinsics are w2c (world-to-camera) transformations.
    
    Args:
        cameras: Dictionary of Camera objects from COLMAP
        images: Dictionary of Image objects from COLMAP
    
    Returns:
        Tuple of (intrinsics, extrinsics) as numpy arrays:
        - intrinsics: (N, 3, 3) camera calibration matrices
        - extrinsics: (N, 4, 4) world-to-camera transformation matrices
    """
    intrinsics_list = []
    extrinsics_list = []

    for img_id, img in images.items():
        cam = cameras[img.camera_id]
        
        # Extract intrinsics based on camera model
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        elif cam.model in ["OPENCV", "RADIAL", "SIMPLE_RADIAL"]:
            # These models start with focal length and principal point
            if len(cam.params) >= 4:
                fx, fy, cx, cy = cam.params[:4]
            elif len(cam.params) >= 3:
                f, cx, cy = cam.params[:3]
                fx = fy = f
            else:
                raise ValueError(f"Unsupported parameter count for {cam.model}")
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics_list.append(K)

        # Extrinsics: R (from qvec) and tvec form w2c
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        extrinsics_list.append(w2c)

    return np.array(intrinsics_list), np.array(extrinsics_list)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportCOLMAPCameras(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        
        self.setWindowTitle("Import COLMAP Camera Parameters")
        self.resize(700, 150)
        
        # Setup the file selection layout
        self.setup_file_selection_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
    
    def setup_file_selection_layout(self):
        """Setup the layout for selecting COLMAP files."""
        group_box = QGroupBox("COLMAP Files")
        layout = QFormLayout()
        
        # Cameras file
        self.cameras_file_edit = QLineEdit()
        self.cameras_browse_button = QPushButton("Browse...")
        self.cameras_browse_button.clicked.connect(self.browse_cameras_file)
        cameras_layout = QHBoxLayout()
        cameras_layout.addWidget(self.cameras_file_edit)
        cameras_layout.addWidget(self.cameras_browse_button)
        layout.addRow("Cameras File:", cameras_layout)
        
        # Images file
        self.images_file_edit = QLineEdit()
        self.images_browse_button = QPushButton("Browse...")
        self.images_browse_button.clicked.connect(self.browse_images_file)
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.images_file_edit)
        images_layout.addWidget(self.images_browse_button)
        layout.addRow("Images File:", images_layout)
        
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
    
    def browse_cameras_file(self):
        """Open a file dialog to select the cameras file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COLMAP Cameras File",
            "",
            "COLMAP Files (*.bin *.txt);;Binary Files (*.bin);;Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            self.cameras_file_edit.setText(file_path)
            
            # Auto-fill images file if it exists in the same directory
            parent_dir = os.path.dirname(file_path)
            file_ext = os.path.splitext(file_path)[1]  # .bin or .txt
            images_file = os.path.join(parent_dir, f"images{file_ext}")
            
            if os.path.exists(images_file) and not self.images_file_edit.text():
                self.images_file_edit.setText(images_file)
    
    def browse_images_file(self):
        """Open a file dialog to select the images file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COLMAP Images File",
            "",
            "COLMAP Files (*.bin *.txt);;Binary Files (*.bin);;Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            self.images_file_edit.setText(file_path)
            
            # Auto-fill cameras file if it exists in the same directory
            parent_dir = os.path.dirname(file_path)
            file_ext = os.path.splitext(file_path)[1]  # .bin or .txt
            cameras_file = os.path.join(parent_dir, f"cameras{file_ext}")
            
            if os.path.exists(cameras_file) and not self.cameras_file_edit.text():
                self.cameras_file_edit.setText(cameras_file)
    
    def import_cameras(self):
        """Import camera intrinsics and extrinsics from COLMAP files into the current project."""
        # Get file paths from the text fields
        cameras_file = self.cameras_file_edit.text()
        images_file = self.images_file_edit.text()
        
        # Validate that both files are selected
        if not cameras_file or not images_file:
            QMessageBox.warning(self,
                                "Missing Files",
                                "Please select both cameras and images files.")
            return
        
        # Validate that files exist
        if not os.path.exists(cameras_file):
            QMessageBox.warning(self,
                                "File Not Found",
                                f"Cameras file not found: {cameras_file}")
            return
        
        if not os.path.exists(images_file):
            QMessageBox.warning(self,
                                "File Not Found",
                                f"Images file not found: {images_file}")
            return
        
        # Check if images are loaded
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing camera parameters.")
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Read cameras file based on extension
            cameras_ext = os.path.splitext(cameras_file)[1].lower()
            if cameras_ext == '.bin':
                cameras = read_cameras_binary(cameras_file)
            elif cameras_ext == '.txt':
                cameras = read_cameras_text(cameras_file)
            else:
                raise ValueError(f"Unsupported cameras file format: {cameras_ext}")
            
            # Read images file based on extension
            images_ext = os.path.splitext(images_file)[1].lower()
            if images_ext == '.bin':
                images = read_images_binary(images_file)
            elif images_ext == '.txt':
                images = read_images_text(images_file)
            else:
                raise ValueError(f"Unsupported images file format: {images_ext}")
            
            if not cameras or not images:
                raise Exception("Failed to read COLMAP files or files are empty.")
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self,
                                "Error Reading COLMAP Files",
                                f"An error occurred while reading COLMAP files: {str(e)}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        
        # Match COLMAP images to loaded rasters
        image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}
        matched_images = {}
        
        for img_id, img in images.items():
            img_basename = os.path.basename(img.name)
            if img_basename in image_path_map:
                matched_images[img_id] = img
        
        if not matched_images:
            QMessageBox.warning(self,
                                "No Matching Images",
                                "No images from the COLMAP files match the loaded images in the project.")
            return
        
        # Check for existing camera parameters in matched rasters
        rasters_with_cameras = []
        for img_id, img in matched_images.items():
            img_basename = os.path.basename(img.name)
            image_path = image_path_map[img_basename]
            raster = self.image_window.raster_manager.get_raster(image_path)
            
            if raster and (raster.intrinsics is not None or raster.extrinsics is not None):
                rasters_with_cameras.append(img_basename)
        
        # Alert user if existing camera data found
        if rasters_with_cameras:
            count = len(rasters_with_cameras)
            response = QMessageBox.question(
                self,
                "Existing Camera Parameters Found",
                f"Found {count} image(s) with existing camera parameters.\n\n"
                f"Do you want to proceed and overwrite them?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            
            if response == QMessageBox.Cancel:
                return
        
        # Extract intrinsics and extrinsics using the centralized function
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing COLMAP Camera Parameters")
        progress_bar.show()
        progress_bar.start_progress(len(matched_images))
        
        updated_count = 0
        skipped_count = 0
        
        try:
            # Extract all intrinsics and extrinsics at once
            all_intrinsics, all_extrinsics = extract_intrinsics_extrinsics_from_colmap(cameras, matched_images)
            
            # Map them back to the matched images
            for idx, (img_id, img) in enumerate(matched_images.items()):
                try:
                    # Get the intrinsics and extrinsics for this image
                    K = all_intrinsics[idx]
                    w2c = all_extrinsics[idx]
                    
                    # Get the raster and update camera parameters
                    img_basename = os.path.basename(img.name)
                    image_path = image_path_map[img_basename]
                    raster = self.image_window.raster_manager.get_raster(image_path)
                    
                    if raster:
                        # Add or update camera parameters
                        if raster.intrinsics is not None or raster.extrinsics is not None:
                            raster.update_intrinsics(K)
                            raster.update_extrinsics(w2c)
                        else:
                            raster.add_intrinsics(K)
                            raster.add_extrinsics(w2c)
                        
                        updated_count += 1
                    else:
                        skipped_count += 1
                    
                except Exception as e:
                    print(f"Error processing image {img.name}: {str(e)}")
                    skipped_count += 1
                
                progress_bar.update_progress()
            
            # Show summary
            summary_msg = f"Successfully imported camera parameters for {updated_count} image(s)."
            if skipped_count > 0:
                summary_msg += f"\n\nSkipped {skipped_count} image(s) due to errors."
            
            QMessageBox.information(self,
                                    "Import Complete",
                                    summary_msg)
            
            # Close dialog on success
            self.accept()
            
        except Exception as e:
            QMessageBox.warning(self,
                                "Error Importing Camera Parameters",
                                f"An error occurred while importing camera parameters: {str(e)}")
        
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
