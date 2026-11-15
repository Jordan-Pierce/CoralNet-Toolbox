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
    # Ultralytics augmentations that can cut the labeled point out of the patch,
    # forced off under Point Safe. `scale` 0 makes RandomResizedCrop's area
    # (1 - scale, 1) = (1, 1), i.e. no random crop. `auto_augment` None skips
    # RandAugment, which includes translate, shear and cutout.
    POINT_SAFE_SCALE = 0.0
    POINT_SAFE_ERASING = 0.0
    POINT_SAFE_AUTO_AUGMENT = None

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Classification Model")

    def point_safe_enabled(self):
        """True when Point Safe Augmentations is "True" (the default)."""
        combo = getattr(self, 'point_safe_combo', None)
        return True if combo is None else combo.currentText() == "True"

    def get_parameters(self):
        """Training parameters; with Point Safe on, minus augmentations that can cut away the label.

        A patch is labeled by its center point. Random erasing (`erasing`) can
        blank it, a random crop (`scale`) can crop it out, and RandAugment
        (`auto_augment`) includes translate, shear and cutout. All three are
        forced off. Forced, since an imported args.yaml or a custom parameter
        can bring them back.
        """
        params = super().get_parameters()

        if not self.point_safe_enabled():
            return params

        for key, forced in (('erasing', self.POINT_SAFE_ERASING),
                            ('scale', self.POINT_SAFE_SCALE),
                            ('auto_augment', self.POINT_SAFE_AUTO_AUGMENT)):
            value = params.get(key)
            if value != forced:
                print(f"Note: {key}={value} overridden to {forced}; this augmentation can "
                      f"move or cover the point each patch is labeled by.")
            params[key] = forced

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

    def setup_model_layout(self):
        """Model Selection, then the augmentation switch directly beneath it.

        Augmentation is a statement about how the chosen model is trained rather
        than about the dataset, so it belongs under Model Selection. The base
        class calls this after the dataset and output groups, which is what puts
        the switch at the bottom of the left column.
        """
        super().setup_model_layout()
        self.setup_augmentation_layout()

    def setup_augmentation_layout(self):
        """Point Safe Augmentations, at the bottom of the left column."""
        aug_box = QGroupBox("Augmentation")
        aug_layout = QFormLayout()
        self.point_safe_combo = QComboBox()
        self.point_safe_combo.addItems(["True", "False"])
        self.point_safe_combo.setToolTip(
            "Keep the patch's center point, which is its label, in view.\n"
            "Turns off random erasing, random crop and RandAugment (auto_augment).")
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