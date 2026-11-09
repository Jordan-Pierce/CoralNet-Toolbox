from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox, QComboBox)

from coralnet_toolbox.MachineLearning.TrainModel.QtBase import Base
from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
from coralnet_toolbox.paths import find_weights
from coralnet_toolbox.pyspacer import CORALNET_WEIGHTS_NAME


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    # Largest Ultralytics `scale` that keeps the patch centre, where the point
    # label sits, inside every training crop. RandomResizedCrop takes an area of
    # (1 - scale, 1) of the patch at an aspect ratio of 3/4 to 4/3, so at 0.5 the
    # shortest crop side is sqrt(0.5 * 3/4) = 0.61 of the patch and the centre
    # stays at least 18% in from the crop edge, enough to survive a RandAugment
    # translate (about 14%). It is also the Ultralytics default.
    MAX_CROP_SCALE = 0.5

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Classification Model")

    def point_safe_enabled(self):
        """True when Point Safe Augmentations is "True" (the default)."""
        combo = getattr(self, 'point_safe_combo', None)
        return True if combo is None else combo.currentText() == "True"

    def get_parameters(self):
        """Training parameters; with Point Safe on, minus augmentations that can cut away the label.

        A patch is labeled by its center point. Random erasing can blank it, and
        a strong random crop (`scale`) can crop it out, so erasing is zeroed and
        scale capped. Forced, since an imported args.yaml brings erasing back.
        """
        params = super().get_parameters()

        if not self.point_safe_enabled():
            return params

        erasing = params.get('erasing')
        if isinstance(erasing, (int, float)) and erasing > 0:
            print(f"Note: erasing={erasing} ignored; random erasing can cover "
                  f"the point each patch is labeled by.")
        params['erasing'] = 0.0

        scale = params.get('scale', self.MAX_CROP_SCALE)
        if not isinstance(scale, (int, float)) or scale > self.MAX_CROP_SCALE:
            print(f"Note: scale={scale} capped at {self.MAX_CROP_SCALE}; a "
                  f"stronger random crop can cut the labeled point out of the patch.")
            scale = self.MAX_CROP_SCALE
        params['scale'] = scale

        return params

    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        
        self.task = "classify"
        self.imgsz = 256
        self.batch = 64

        group_box = QGroupBox("Dataset")
        layout = QFormLayout()

        # Dataset Directory
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_dir)
        self.dataset_edit.setToolTip("Path to classification dataset with train/val/test splits.\nEach split should contain subdirectories named after class labels.")
        self.dataset_button.setToolTip("Browse for a classification dataset directory.")

        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(self.dataset_edit)
        dataset_dir_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset Directory:", dataset_dir_layout)

        # Class Mapping
        self.mapping_edit = QLineEdit()
        self.mapping_button = QPushButton("Browse...")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)
        self.mapping_edit.setToolTip("Optional YAML file mapping class IDs to class names.\nUsed during inference to label predictions.")
        self.mapping_button.setToolTip("Browse for a class mapping YAML file.")

        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(self.mapping_edit)
        class_mapping_layout.addWidget(self.mapping_button)
        layout.addRow("Class Mapping:", class_mapping_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

        # Point Safe Augmentations
        aug_box = QGroupBox("Augmentation")
        aug_layout = QFormLayout()
        self.point_safe_combo = QComboBox()
        self.point_safe_combo.addItems(["True", "False"])
        self.point_safe_combo.setToolTip(
            "Keep the patch's center point, which is its label, in view.\n"
            f"Turns off random erasing and caps random crop at scale {self.MAX_CROP_SCALE}.")
        aug_layout.addRow("Point Safe Augmentations:", self.point_safe_combo)
        aug_box.setLayout(aug_layout)
        self.layout.addWidget(aug_box)

    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)

        standard_models = ['yolov8n-cls.pt',
                           'yolov8s-cls.pt',
                           'yolov8m-cls.pt',
                           'yolov8l-cls.pt',
                           'yolov8x-cls.pt',
                           'yolo11n-cls.pt',
                           'yolo11s-cls.pt',
                           'yolo11m-cls.pt',
                           'yolo11l-cls.pt',
                           'yolo11x-cls.pt',
                           'yolo26n-cls.pt',
                           'yolo26s-cls.pt',
                           'yolo26m-cls.pt',
                           'yolo26l-cls.pt',
                           'yolo26x-cls.pt']
        
        # CoralNet's weights are not distributed, so they are offered only when
        # the user has put a copy in the weights folder, and are then the default.
        # The bare name is resolved to that copy when training starts.
        if find_weights(CORALNET_WEIGHTS_NAME):
            self.model_combo.addItem(CORALNET_WEIGHTS_NAME)
            self.model_combo.setItemData(
                0, "CoralNet's EfficientNet-B0, trained on CoralNet's patches.\n"
                   "Trained here as a new classifier on your labels.", Qt.ToolTipRole)
            self.model_combo.insertSeparator(1)

        coralnet_first = self.model_combo.count() > 0
        self.model_combo.addItems(standard_models)

        # Add community models
        community_configs = get_available_configs(task=self.task)
        if community_configs:
            self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItems(list(community_configs.keys()))

        # Set the default model
        self.model_combo.setCurrentIndex(0 if coralnet_first else standard_models.index('yolo11n-cls.pt'))