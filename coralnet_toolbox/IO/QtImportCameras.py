import os
import struct
import warnings
import collections
import xml.etree.ElementTree as ET

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QMessageBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
    QTabWidget,
)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Shared utilities & data structures
# ----------------------------------------------------------------------------------------------------------------------

def normalize_extension(name):
    """Return filename with lower-cased extension for case-insensitive matching."""
    base, ext = os.path.splitext(name)
    return base + ext.lower()


# COLMAP namedtuples
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        """Convert this image's quaternion to a 3x3 rotation matrix."""
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


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file using struct format."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """Parse COLMAP cameras.txt and return a dict of Camera namedtuples."""
    cameras = {}
    with open(path) as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """Parse COLMAP cameras.bin and return a dict of Camera namedtuples."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
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
    """Parse COLMAP images.txt and return a dict of Image namedtuples."""
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
                # second line contains 2D points
                elems = fid.readline().split()
                if len(elems) > 0:
                    xys = np.column_stack(
                        [
                            tuple(map(float, elems[0::3])),
                            tuple(map(float, elems[1::3])),
                        ]
                    )
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                else:
                    xys = np.zeros((0, 2))
                    point3D_ids = np.zeros((0,), dtype=int)

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
    """Parse COLMAP images.bin and return a dict of Image namedtuples."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            # read null-terminated string
            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.extend(c)
            image_name = name_bytes.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            if num_points2D > 0:
                x_y_id_s = read_next_bytes(
                    fid,
                    num_bytes=24 * num_points2D,
                    format_char_sequence=("ddq" * num_points2D),
                )
                xys = np.column_stack(
                    [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            else:
                xys = np.zeros((0, 2))
                point3D_ids = np.zeros((0,), dtype=int)
            images[image_id] = Image(id=image_id, 
                                     qvec=qvec, 
                                     tvec=tvec, 
                                     camera_id=camera_id, 
                                     name=image_name, 
                                     xys=xys, 
                                     point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    """Convert quaternion [qw,qx,qy,qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx ** 2 - 2 * qy ** 2],
    ])


def extract_intrinsics_extrinsics_from_colmap(cameras, images):
    """Extract intrinsics (K) and extrinsics (w2c) arrays from COLMAP data.

    Returns (intrinsics, extrinsics, labels).
    """
    intrinsics_list = []
    extrinsics_list = []
    labels = []

    for img_id, img in images.items():
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue

        # Parse parameters based on the specific COLMAP camera model
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
            
        # FIX: OpenCV models use two distinct focal lengths (fx, fy) followed by the principal point (cx, cy)
        elif cam.model in ["OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
            
        # FIX: Radial models use a single focal length (f) followed by the principal point (cx, cy)
        elif cam.model in ["RADIAL", "SIMPLE_RADIAL", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE"]:
            f = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
            fx = fy = f
            
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

        # Construct the Intrinsic Matrix (K)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics_list.append(K)

        # Construct the Extrinsic Matrix (World-to-Camera, 4x4)
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        extrinsics_list.append(w2c)
        
        # Store image label for matching (lowercased without extension)
        labels.append(os.path.splitext(os.path.basename(img.name))[0].lower())

    if len(intrinsics_list) == 0:
        return np.array([]), np.array([]), []

    return np.array(intrinsics_list), np.array(extrinsics_list), labels


# ----------------------------------------------------------------------------------------------------------------------
# Metashape structures and functions (kept from original)
# ----------------------------------------------------------------------------------------------------------------------

Sensor = collections.namedtuple("Sensor", ["id", "label", "width", "height", "calibration"])
M_Camera = collections.namedtuple("Camera", ["id", "sensor_id", "label", "transform"])
Calibration = collections.namedtuple("Calibration", ["f", "cx", "cy", "fx", "fy"])


def read_metashape_xml(xml_path):
    """Read Metashape cameras XML and return sensors and cameras dictionaries."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensors = {}
    cameras = {}

    sensors_element = root.find('.//sensors')
    if sensors_element is not None:
        for sensor_elem in sensors_element.findall('sensor'):
            sensor_id = int(sensor_elem.get('id'))
            label = sensor_elem.get('label', 'unknown')
            resolution = sensor_elem.find('resolution')
            if resolution is not None:
                width = int(resolution.get('width'))
                height = int(resolution.get('height'))
            else:
                width = height = 0
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
            sensors[sensor_id] = Sensor(id=sensor_id, label=label, width=width, height=height, calibration=calibration)

    cameras_element = root.find('.//cameras')
    if cameras_element is not None:
        for camera_elem in cameras_element.findall('camera'):
            camera_id = int(camera_elem.get('id'))
            sensor_id = int(camera_elem.get('sensor_id'))
            label = camera_elem.get('label', '')
            transform_elem = camera_elem.find('transform')
            transform = None
            if transform_elem is not None and transform_elem.text:
                try:
                    values = [float(v) for v in transform_elem.text.split()]
                    if len(values) == 16:
                        transform = np.array(values).reshape(4, 4)
                except (ValueError, AttributeError):
                    pass
            cameras[camera_id] = M_Camera(id=camera_id, sensor_id=sensor_id, label=label, transform=transform)

    return sensors, cameras


def extract_intrinsics_extrinsics_from_metashape(sensors, cameras):
    """Extract intrinsics and world-to-camera extrinsics from Metashape data."""
    intrinsics_list = []
    extrinsics_list = []
    camera_labels = []

    T_metashape_to_opencv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    for cam_id, cam in cameras.items():
        if cam.transform is None:
            continue
        sensor = sensors.get(cam.sensor_id)
        if sensor is None or sensor.calibration is None:
            continue
        calib = sensor.calibration
        if calib.fx is not None and calib.fy is not None:
            fx = calib.fx
            fy = calib.fy
        elif calib.f is not None:
            fx = fy = calib.f
        else:
            continue
        image_center_x = sensor.width / 2.0
        image_center_y = sensor.height / 2.0
        if calib.cx is not None and calib.cy is not None:
            cx = image_center_x + calib.cx
            cy = image_center_y + calib.cy
        else:
            cx = image_center_x
            cy = image_center_y
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics_list.append(K)
        try:
            c2w_metashape = cam.transform
            c2w_opencv = c2w_metashape @ T_metashape_to_opencv
            w2c = np.linalg.inv(c2w_opencv)
            extrinsics_list.append(w2c)
        except np.linalg.LinAlgError:
            intrinsics_list.pop()
            continue
        # store label as basename without extension, lowercased for matching
        camera_labels.append(os.path.splitext(cam.label)[0].lower())

    if len(intrinsics_list) == 0:
        return np.array([]), np.array([]), []

    return np.array(intrinsics_list), np.array(extrinsics_list), camera_labels


# ----------------------------------------------------------------------------------------------------------------------
# UI: combined tabbed dialog
# ----------------------------------------------------------------------------------------------------------------------


class ColmapTab(QWidget):
    def __init__(self, parent_dialog):
        """Initialize the COLMAP import tab UI and bindings."""
        super().__init__(parent_dialog)
        self.parent_dialog = parent_dialog
        self.main_window = parent_dialog.main_window
        self.image_window = self.main_window.image_window

        layout = QFormLayout()

        self.cameras_file_edit = QLineEdit()
        self.cameras_browse_button = QPushButton("Browse...")
        self.cameras_browse_button.clicked.connect(self.browse_cameras_file)
        cameras_layout = QHBoxLayout()
        cameras_layout.addWidget(self.cameras_file_edit)
        cameras_layout.addWidget(self.cameras_browse_button)
        layout.addRow("Cameras File:", cameras_layout)

        self.images_file_edit = QLineEdit()
        self.images_browse_button = QPushButton("Browse...")
        self.images_browse_button.clicked.connect(self.browse_images_file)
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.images_file_edit)
        images_layout.addWidget(self.images_browse_button)
        layout.addRow("Images File:", images_layout)

        self.import_button = QPushButton("Import COLMAP")
        self.import_button.clicked.connect(self.import_cameras)
        btns = QHBoxLayout()
        btns.addWidget(self.import_button)
        layout.addRow(btns)

        self.setLayout(layout)

    def browse_cameras_file(self):
        """Open a file dialog to select a COLMAP cameras file and autofill images file if present."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Select COLMAP Cameras File", "", 
                                                   "COLMAP Files (*.bin *.txt);;All Files (*)", options=options)
        if file_path:
            self.cameras_file_edit.setText(file_path)
            parent_dir = os.path.dirname(file_path)
            file_ext = os.path.splitext(file_path)[1]
            images_file = f"{parent_dir}/images{file_ext}"
            if os.path.exists(images_file) and not self.images_file_edit.text():
                self.images_file_edit.setText(images_file)

    def browse_images_file(self):
        """Open a file dialog to select a COLMAP images file and autofill cameras file if present."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Select COLMAP Images File", 
                                                   "", 
                                                   "COLMAP Files (*.bin *.txt);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.images_file_edit.setText(file_path)
            parent_dir = os.path.dirname(file_path)
            file_ext = os.path.splitext(file_path)[1]
            cameras_file = f"{parent_dir}/cameras{file_ext}"
            if os.path.exists(cameras_file) and not self.cameras_file_edit.text():
                self.cameras_file_edit.setText(cameras_file)

    def import_cameras(self):
        """Import camera intrinsics and extrinsics from selected COLMAP files into project rasters."""
        cameras_file = self.cameras_file_edit.text()
        images_file = self.images_file_edit.text()
        if not cameras_file or not images_file:
            QMessageBox.warning(self, "Missing Files", "Please select both cameras and images files.")
            return
        if not os.path.exists(cameras_file) or not os.path.exists(images_file):
            QMessageBox.warning(self, "File Not Found", "Cameras or images file not found.")
            return
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self, 
                                "No Images Loaded", 
                                "Please load images first before importing camera parameters.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cameras_ext = os.path.splitext(cameras_file)[1].lower()
            if cameras_ext == '.bin':
                cameras = read_cameras_binary(cameras_file)
            else:
                cameras = read_cameras_text(cameras_file)

            images_ext = os.path.splitext(images_file)[1].lower()
            if images_ext == '.bin':
                images = read_images_binary(images_file)
            else:
                images = read_images_text(images_file)

            if not cameras or not images:
                QMessageBox.warning(self, 
                                    "No Data", 
                                    "No cameras or images parsed from COLMAP files.")
                return
        except Exception as e:
            QMessageBox.warning(self, 
                                "Error Reading COLMAP Files", 
                                f"An error occurred while reading COLMAP files: {str(e)}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            
        # Build a case-insensitive map of loaded images by basename (no extension)
        image_path_map = {}
        for path in self.image_window.raster_manager.image_paths:
            base = os.path.splitext(os.path.basename(path))[0].lower()
            image_path_map[base] = path

        matched_images = {}
        for img_id, img in images.items():
            img_basename = os.path.splitext(os.path.basename(img.name))[0].lower()
            if img_basename in image_path_map:
                matched_images[img_id] = img

        if not matched_images:
            QMessageBox.warning(self,
                                "No Matching Images", 
                                "No images from the COLMAP files match the loaded images in the project.")
            return

        rasters_with_cameras = []
        for img_id, img in matched_images.items():
            img_basename = os.path.splitext(os.path.basename(img.name))[0].lower()
            image_path = image_path_map[img_basename]
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and (raster.intrinsics is not None or raster.extrinsics is not None):
                rasters_with_cameras.append(image_path)

        if rasters_with_cameras:
            count = len(rasters_with_cameras)
            response = QMessageBox.question(
                self,
                "Existing Camera Parameters Found",
                f"Found {count} image(s) with existing camera parameters.\n\n"
                "Do you want to proceed and overwrite them?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if response == QMessageBox.Cancel:
                return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing COLMAP Camera Parameters")
        progress_bar.show()
        progress_bar.start_progress(len(matched_images))

        updated_count = 0
        skipped_count = 0

        try:
            intrinsics_all, extrinsics_all, labels = extract_intrinsics_extrinsics_from_colmap(cameras, matched_images)
            label_to_idx = {label: idx for idx, label in enumerate(labels)}

            for img_id, img in matched_images.items():
                image_basename = os.path.splitext(os.path.basename(img.name))[0].lower()
                if image_basename not in label_to_idx:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                idx = label_to_idx[image_basename]
                intrinsics = intrinsics_all[idx]
                extrinsics = extrinsics_all[idx]
                image_path = image_path_map[image_basename]
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                try:
                    if raster.intrinsics is not None or raster.extrinsics is not None:
                        raster.update_intrinsics(intrinsics)
                        raster.update_extrinsics(extrinsics)
                    else:
                        raster.add_intrinsics(intrinsics)
                        raster.add_extrinsics(extrinsics)
                    updated_count += 1
                except Exception:
                    skipped_count += 1
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self, 
                                "Error Importing Camera Parameters", 
                                f"An error occurred while importing camera parameters: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        summary_msg = f"Successfully imported camera parameters for {updated_count} image(s)."
        if skipped_count > 0:
            summary_msg += f" Skipped {skipped_count} image(s)."
        QMessageBox.information(self, "Import Complete", summary_msg)
        self.parent_dialog.accept()


class MetashapeTab(QWidget):
    def __init__(self, parent_dialog):
        """Initialize the Metashape import tab UI and bindings."""
        super().__init__(parent_dialog)
        self.parent_dialog = parent_dialog
        self.main_window = parent_dialog.main_window
        self.image_window = self.main_window.image_window

        layout = QFormLayout()
        self.xml_file_edit = QLineEdit()
        self.xml_browse_button = QPushButton("Browse...")
        self.xml_browse_button.clicked.connect(self.browse_xml_file)
        xml_layout = QHBoxLayout()
        xml_layout.addWidget(self.xml_file_edit)
        xml_layout.addWidget(self.xml_browse_button)
        layout.addRow("Cameras XML File:", xml_layout)

        self.import_button = QPushButton("Import Metashape")
        self.import_button.clicked.connect(self.import_cameras)
        btns = QHBoxLayout()
        btns.addWidget(self.import_button)
        layout.addRow(btns)

        self.setLayout(layout)

    def browse_xml_file(self):
        """Open a file dialog to select a Metashape cameras XML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Select Metashape Cameras XML File", 
                                                   "", 
                                                   "XML Files (*.xml);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.xml_file_edit.setText(file_path)

    def import_cameras(self):
        """Import camera intrinsics and extrinsics from the selected Metashape XML into project rasters."""
        xml_file = self.xml_file_edit.text()
        if not xml_file:
            QMessageBox.warning(self, "Missing File", "Please select a cameras XML file.")
            return
        if not os.path.exists(xml_file):
            QMessageBox.warning(self, "File Not Found", f"The selected XML file does not exist:\n{xml_file}")
            return
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self, 
                                "No Images Loaded", 
                                "Please load images into the project before importing camera parameters.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sensors, cameras = read_metashape_xml(xml_file)
            if not sensors:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "No Sensors Found", "No sensor calibration data found in the XML file.")
                return
            if not cameras:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "No Cameras Found", "No camera data found in the XML file.")
                return
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error Reading XML", f"Failed to read Metashape XML file:\n{str(e)}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # Case-insensitive basename map for loaded images
        image_path_map = {}
        for path in self.image_window.raster_manager.image_paths:
            base = os.path.splitext(os.path.basename(path))[0].lower()
            image_path_map[base] = path

        matched_cameras = {}
        for cam_id, cam in cameras.items():
            if not cam.label:
                continue
            cam_basename = os.path.splitext(cam.label)[0].lower()
            if cam_basename in image_path_map:
                matched_cameras[cam_id] = cam

        if not matched_cameras:
            QMessageBox.warning(
                self,
                "No Matching Images",
                "No camera labels from the XML file match the loaded image basenames.\n\n"
                "Camera labels should match image filenames (without extension).",
            )
            return

        rasters_with_cameras = []
        for cam_id, cam in matched_cameras.items():
            image_basename = os.path.splitext(cam.label)[0].lower()
            image_path = image_path_map[image_basename]
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and (raster.intrinsics is not None or raster.extrinsics is not None):
                rasters_with_cameras.append(image_basename)

        if rasters_with_cameras:
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Camera Data?",
                f"Found existing camera parameters for {len(rasters_with_cameras)} image(s).\n\n"
                "Do you want to overwrite them with data from the Metashape XML file?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing Metashape Camera Parameters")
        progress_bar.show()
        progress_bar.start_progress(len(matched_cameras))

        updated_count = 0
        skipped_count = 0

        try:
            (
                intrinsics_all,
                extrinsics_all,
                camera_labels,
            ) = extract_intrinsics_extrinsics_from_metashape(sensors, cameras)
            if len(intrinsics_all) == 0:
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self,
                                    "No Valid Camera Data", 
                                    "No valid camera calibration or transform data found in the XML file.")
                return
            # Use lowercase labels for case-insensitive matching
            label_to_idx = {label.lower(): idx for idx, label in enumerate(camera_labels)}

            for cam_id, cam in matched_cameras.items():
                image_basename = os.path.splitext(cam.label)[0].lower()
                if image_basename not in label_to_idx:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                idx = label_to_idx[image_basename]
                intrinsics = intrinsics_all[idx]
                extrinsics = extrinsics_all[idx]
                image_path = image_path_map[image_basename]
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster is None:
                    skipped_count += 1
                    progress_bar.update_progress()
                    continue
                try:
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
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{str(e)}")
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

        summary_msg = f"Import complete!\n\nUpdated: {updated_count} image(s)\n"
        if skipped_count > 0:
            summary_msg += f"Skipped: {skipped_count} image(s)"
        QMessageBox.information(self, "Import Complete", summary_msg)
        self.parent_dialog.accept()


class ImportCameras(QDialog):
    def __init__(self, main_window):
        """Create the tabbed import dialog containing COLMAP and Metashape tabs."""
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Import Camera Parameters")
        self.resize(800, 180)

        tabs = QTabWidget()
        self.colmap_tab = ColmapTab(self)
        self.metashape_tab = MetashapeTab(self)
        tabs.addTab(self.colmap_tab, "COLMAP")
        # Metashape tab is disabled for now
        tabs.addTab(self.metashape_tab, "Metashape")
        try:
            idx = tabs.indexOf(self.metashape_tab)
            tabs.setTabEnabled(idx, False)
            tabs.setTabToolTip(idx, "Disabled for now")
        except Exception:
            # safety: if something goes wrong, leave tab as-is
            pass

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
