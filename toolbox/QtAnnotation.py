import uuid
import warnings

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QMessageBox)

from toolbox.QtLabel import Label

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Annotation(QObject):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)
    annotation_deleted = pyqtSignal(object)
    annotation_updated = pyqtSignal(object)

    def __init__(self, short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=True):
        super().__init__()
        self.id = str(uuid.uuid4())
        self.label = Label(short_label_code, long_label_code, color, label_id)
        self.image_path = image_path
        self.is_selected = False
        self.graphics_item = None
        self.transparency = transparency
        self.user_confidence = {self.label: 1.0}
        self.machine_confidence = {}
        self.data = {}
        self.rasterio_src = None
        self.cropped_image = None

        self.show_message = show_msg

    def show_warning_message(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Altering an annotation with predictions will remove the machine suggestions.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        # Only show once
        self.show_message = False

    def select(self):
        self.is_selected = True
        self.update_graphics_item()

    def deselect(self):
        self.is_selected = False
        self.update_graphics_item()

    def delete(self):
        self.annotation_deleted.emit(self)
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None

    def update_machine_confidence(self, prediction: dict):
        # Set user confidence to None
        self.user_confidence = {}
        # Update machine confidence
        self.machine_confidence = prediction
        # Pass the label with the largest confidence as the label
        self.label = max(prediction, key=prediction.get)
        # Create the graphic
        self.update_graphics_item()

    def update_user_confidence(self, new_label: 'Label'):
        # Set machine confidence to None
        self.machine_confidence = {}
        # Update user confidence
        self.user_confidence = {new_label: 1.0}
        # Pass the label with the largest confidence as the label
        self.label = new_label
        # Create the graphic
        self.update_graphics_item()

    def update_label(self, new_label: 'Label'):
        self.label = new_label
        self.update_graphics_item()

    def update_transparency(self, transparency: int):
        self.transparency = transparency
        self.update_graphics_item()

    def to_dict(self):
        return {
            'id': self.id,
            'label_short_code': self.label.short_label_code,
            'label_long_code': self.label.long_label_code,
            'annotation_color': self.label.color.getRgb(),
            'image_path': self.image_path,
            'label_id': self.label.id,
            'data': self.data,
            'machine_confidence': self.machine_confidence
        }

    @classmethod
    def from_dict(cls, data):
        annotation = cls(data['label_short_code'],
                         data['label_long_code'],
                         QColor(*data['annotation_color']),
                         data['image_path'],
                         data['label_id'])
        annotation.data = data.get('data', {})
        annotation.machine_confidence = data.get('machine_confidence', {})
        return annotation

    def __repr__(self):
        return (f"Annotation(id={self.id}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")