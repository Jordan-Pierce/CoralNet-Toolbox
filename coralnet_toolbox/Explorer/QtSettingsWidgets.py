import os
import warnings

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, 
                             QWidget, QGroupBox, QSlider, QListWidget, QTabWidget, 
                             QLineEdit, QFileDialog, QFormLayout, QSpinBox)

from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------------------------------------------------


class UncertaintySettingsWidget(QWidget):
    """A widget for configuring uncertainty sampling parameters."""
    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        # Set a strong focus policy to prevent the menu from closing on interaction
        self.confidence_slider.setFocusPolicy(Qt.StrongFocus)
        self.margin_slider.setFocusPolicy(Qt.StrongFocus)

    def setup_ui(self):
        """Creates the UI controls for the parameters."""
        main_layout = QFormLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # 1. Confidence Threshold
        confidence_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(60)
        self.confidence_slider.setToolTip(
            "Find annotations where the model's top guess\n"
            "has a confidence BELOW this threshold."
        )
        self.confidence_label = QLabel("60%")
        self.confidence_label.setMinimumWidth(40)
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        main_layout.addRow("Max Confidence:", confidence_layout)

        # 2. Margin Threshold
        margin_layout = QHBoxLayout()
        self.margin_slider = QSlider(Qt.Horizontal)
        self.margin_slider.setMinimum(0)
        self.margin_slider.setMaximum(50)
        self.margin_slider.setValue(10)
        self.margin_slider.setToolTip(
            "Find annotations where the confidence difference\n"
            "between the top two guesses is BELOW this threshold."
        )
        self.margin_label = QLabel("10%")
        self.margin_label.setMinimumWidth(40)
        margin_layout.addWidget(self.margin_slider)
        margin_layout.addWidget(self.margin_label)
        main_layout.addRow("Min Margin:", margin_layout)

        # Connect signals
        self.confidence_slider.valueChanged.connect(self._emit_parameters)
        self.margin_slider.valueChanged.connect(self._emit_parameters)
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v}%")
        )
        self.margin_slider.valueChanged.connect(
            lambda v: self.margin_label.setText(f"{v}%")
        )

    @pyqtSlot()
    def _emit_parameters(self):
        """Gathers current values and emits them in a dictionary."""
        params = self.get_parameters()
        self.parameters_changed.emit(params)

    def get_parameters(self):
        """Returns the current parameters as a dictionary."""
        return {
            'confidence': self.confidence_slider.value() / 100.0,
            'margin': self.margin_slider.value() / 100.0
        }
        

class MislabelSettingsWidget(QWidget):
    """A widget for configuring mislabel detection parameters."""
    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        # Set a default value to prevent the menu from closing on interaction
        self.k_spinbox.setFocusPolicy(Qt.StrongFocus)
        self.threshold_slider.setFocusPolicy(Qt.StrongFocus)

    def setup_ui(self):
        """Creates the UI controls for the parameters."""
        main_layout = QFormLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # 1. K (Number of Neighbors)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(50)
        self.k_spinbox.setValue(5)
        self.k_spinbox.setToolTip("Number of neighbors to check for each point (K).")
        main_layout.addRow("Neighbors (K):", self.k_spinbox)

        # 2. Agreement Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(60)
        self.threshold_slider.setToolTip(
            "A point is flagged if the percentage of neighbors\n"
            "with the same label is BELOW this threshold."
        )

        self.threshold_label = QLabel("60%")
        self.threshold_label.setMinimumWidth(40)

        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        main_layout.addRow("Agreement:", threshold_layout)
        
        # Connect signals
        self.k_spinbox.valueChanged.connect(self._emit_parameters)
        self.threshold_slider.valueChanged.connect(self._emit_parameters)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v}%")
        )
    
    @pyqtSlot()
    def _emit_parameters(self):
        """Gathers current values and emits them in a dictionary."""
        params = self.get_parameters()
        self.parameters_changed.emit(params)

    def get_parameters(self):
        """Returns the current parameters as a dictionary."""
        return {
            'k': self.k_spinbox.value(),
            'threshold': self.threshold_slider.value() / 100.0
        }
        

class SimilaritySettingsWidget(QWidget):
    """A widget for configuring similarity search parameters (number of neighbors)."""
    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.k_spinbox.setFocusPolicy(Qt.StrongFocus)

    def setup_ui(self):
        """Creates the UI controls for the parameters."""
        main_layout = QFormLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # K (Number of Neighbors)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(1)
        self.k_spinbox.setMaximum(200)
        self.k_spinbox.setValue(10)
        self.k_spinbox.setToolTip("Number of similar items to find (K).")
        main_layout.addRow("Neighbors (K):", self.k_spinbox)

        # Connect signals
        self.k_spinbox.valueChanged.connect(self._emit_parameters)

    @pyqtSlot()
    def _emit_parameters(self):
        params = self.get_parameters()
        self.parameters_changed.emit(params)

    def get_parameters(self):
        return {
            'k': self.k_spinbox.value()
        }
        

class AnnotationSettingsWidget(QGroupBox):
    """Widget for filtering annotations by image, type, and label in a multi-column layout."""

    def __init__(self, main_window, parent=None):
        super(AnnotationSettingsWidget, self).__init__("Annotation Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent  # Store reference to ExplorerWindow
        self.setup_ui()

    def setup_ui(self):
        # The main layout is vertical, to hold the top columns, the stretch, and the bottom buttons
        layout = QVBoxLayout(self)

        # A horizontal layout to contain the filter columns
        conditions_layout = QHBoxLayout()

        # Images column
        images_column = QVBoxLayout()
        images_label = QLabel("Images:")
        images_label.setStyleSheet("font-weight: bold;")
        images_column.addWidget(images_label)

        self.images_list = QListWidget()
        self.images_list.setSelectionMode(QListWidget.MultiSelection)
        self.images_list.setMaximumHeight(50)

        if hasattr(self.main_window, 'image_window') and hasattr(self.main_window.image_window, 'raster_manager'):
            for path in self.main_window.image_window.raster_manager.image_paths:
                self.images_list.addItem(os.path.basename(path))

        images_column.addWidget(self.images_list)

        images_buttons_layout = QHBoxLayout()
        self.images_select_all_btn = QPushButton("Select All")
        self.images_select_all_btn.clicked.connect(self.select_all_images)
        images_buttons_layout.addWidget(self.images_select_all_btn)

        self.images_deselect_all_btn = QPushButton("Deselect All")
        self.images_deselect_all_btn.clicked.connect(self.deselect_all_images)
        images_buttons_layout.addWidget(self.images_deselect_all_btn)
        images_column.addLayout(images_buttons_layout)

        conditions_layout.addLayout(images_column)

        # Annotation Type column
        type_column = QVBoxLayout()
        type_label = QLabel("Annotation Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_column.addWidget(type_label)

        self.annotation_type_list = QListWidget()
        self.annotation_type_list.setSelectionMode(QListWidget.MultiSelection)
        self.annotation_type_list.setMaximumHeight(50)
        self.annotation_type_list.addItems(["PatchAnnotation",
                                            "RectangleAnnotation",
                                            "PolygonAnnotation",
                                            "MultiPolygonAnnotation"])

        type_column.addWidget(self.annotation_type_list)

        type_buttons_layout = QHBoxLayout()
        self.type_select_all_btn = QPushButton("Select All")
        self.type_select_all_btn.clicked.connect(self.select_all_annotation_types)
        type_buttons_layout.addWidget(self.type_select_all_btn)

        self.type_deselect_all_btn = QPushButton("Deselect All")
        self.type_deselect_all_btn.clicked.connect(self.deselect_all_annotation_types)
        type_buttons_layout.addWidget(self.type_deselect_all_btn)
        type_column.addLayout(type_buttons_layout)

        conditions_layout.addLayout(type_column)

        # Label column
        label_column = QVBoxLayout()
        label_label = QLabel("Label:")
        label_label.setStyleSheet("font-weight: bold;")
        label_column.addWidget(label_label)

        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.MultiSelection)
        self.label_list.setMaximumHeight(50)

        if hasattr(self.main_window, 'label_window') and hasattr(self.main_window.label_window, 'labels'):
            for label in self.main_window.label_window.labels:
                self.label_list.addItem(label.short_label_code)

        label_column.addWidget(self.label_list)

        label_buttons_layout = QHBoxLayout()
        self.label_select_all_btn = QPushButton("Select All")
        self.label_select_all_btn.clicked.connect(self.select_all_labels)
        label_buttons_layout.addWidget(self.label_select_all_btn)

        self.label_deselect_all_btn = QPushButton("Deselect All")
        self.label_deselect_all_btn.clicked.connect(self.deselect_all_labels)
        label_buttons_layout.addWidget(self.label_deselect_all_btn)
        label_column.addLayout(label_buttons_layout)

        conditions_layout.addLayout(label_column)

        # Add the horizontal layout of columns to the main vertical layout
        layout.addLayout(conditions_layout)

        # Add a stretch item to push the columns to the top
        layout.addStretch(1)

        # Bottom buttons layout with Apply and Clear buttons on the right
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()  # Push buttons to the right

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_conditions)
        bottom_layout.addWidget(self.apply_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all_conditions)
        bottom_layout.addWidget(self.clear_button)

        # Add the bottom buttons layout to the main layout, keeping it at the bottom
        layout.addLayout(bottom_layout)

        # Set defaults
        self.set_defaults()

    def select_all_images(self):
        """Select all items in the images list."""
        self.images_list.selectAll()

    def deselect_all_images(self):
        """Deselect all items in the images list."""
        self.images_list.clearSelection()

    def select_all_annotation_types(self):
        """Select all items in the annotation types list."""
        self.annotation_type_list.selectAll()

    def deselect_all_annotation_types(self):
        """Deselect all items in the annotation types list."""
        self.annotation_type_list.clearSelection()

    def select_all_labels(self):
        """Select all items in the labels list."""
        self.label_list.selectAll()

    def deselect_all_labels(self):
        """Deselect all items in the labels list."""
        self.label_list.clearSelection()

    def set_defaults(self):
        """Set default selections."""
        self.set_default_to_current_image()
        self.select_all_annotation_types()
        self.select_all_labels()

    def set_default_to_current_image(self):
        """Set the current image as the default selection."""
        if hasattr(self.main_window, 'annotation_window'):
            current_image_path = self.main_window.annotation_window.current_image_path
            if current_image_path:
                current_image_name = os.path.basename(current_image_path)
                items = self.images_list.findItems(current_image_name, Qt.MatchExactly)
                if items:
                    items[0].setSelected(True)
                    return
        self.select_all_images()

    def clear_all_conditions(self):
        """Reset all conditions to their defaults."""
        self.images_list.clearSelection()
        self.annotation_type_list.clearSelection()
        self.label_list.clearSelection()
        self.set_defaults()
        if self.explorer_window and hasattr(self.explorer_window, 'refresh_filters'):
            self.explorer_window.refresh_filters()

    def apply_conditions(self):
        """Apply the current filter conditions."""
        if self.explorer_window and hasattr(self.explorer_window, 'refresh_filters'):
            self.explorer_window.refresh_filters()

    def get_selected_images(self):
        """Get selected image names."""
        selected_items = self.images_list.selectedItems()
        if not selected_items:
            return []
        return [item.text() for item in selected_items]

    def get_selected_annotation_types(self):
        """Get selected annotation types."""
        selected_items = self.annotation_type_list.selectedItems()
        if not selected_items:
            return []
        return [item.text() for item in selected_items]

    def get_selected_labels(self):
        """Get selected labels."""
        selected_items = self.label_list.selectedItems()
        if not selected_items:
            return []
        return [item.text() for item in selected_items]


class ModelSettingsWidget(QGroupBox):
    """Widget containing model selection with tabs for different model sources."""
    selection_changed = pyqtSignal()

    def __init__(self, main_window, parent=None):
        super(ModelSettingsWidget, self).__init__("Model Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI with a tabbed interface for model selection."""
        main_layout = QVBoxLayout(self)

        # --- Tabbed Interface for Model Selection ---
        self.tabs = QTabWidget()

        # Tab 1: Select Model
        model_select_tab = QWidget()
        model_select_layout = QFormLayout(model_select_tab)
        model_select_layout.setContentsMargins(5, 10, 5, 5)  # Add some top margin

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Color Features"])
        self.model_combo.insertSeparator(1)  # Add a separator

        standard_models = [
            'yolov8n-cls.pt',
            'yolov8s-cls.pt',
            'yolov8m-cls.pt',
            'yolov8l-cls.pt',
            'yolov8x-cls.pt',
            'yolo11n-cls.pt',
            'yolo11s-cls.pt',
            'yolo11m-cls.pt',
            'yolo11l-cls.pt',
            'yolo11x-cls.pt',
            'yolo12n-cls.pt',
            'yolo12s-cls.pt',
            'yolo12m-cls.pt',
            'yolo12l-cls.pt',
            'yolo12x-cls.pt'
        ]

        self.model_combo.addItems(standard_models)

        community_configs = get_available_configs(task='classify')
        if community_configs:
            self.model_combo.insertSeparator(len(standard_models) + 2)
            self.model_combo.addItems(list(community_configs.keys()))

        self.model_combo.setCurrentText('Color Features')
        model_select_layout.addRow("Model:", self.model_combo)
        
        self.tabs.addTab(model_select_tab, "Select Model")

        # Tab 2: Existing Model from File
        model_existing_tab = QWidget()
        model_existing_layout = QFormLayout(model_existing_tab)
        model_existing_layout.setContentsMargins(5, 10, 5, 5)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to a existing .pt model file...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_for_model)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(browse_button)
        model_existing_layout.addRow("Model Path:", path_layout)
        
        self.tabs.addTab(model_existing_tab, "Use Existing Model")

        main_layout.addWidget(self.tabs)
        
        # Connect all relevant widgets to a single slot that emits the new signal
        self.model_combo.currentTextChanged.connect(self._on_selection_changed)
        self.tabs.currentChanged.connect(self._on_selection_changed)
        self.model_path_edit.textChanged.connect(self._on_selection_changed)
        
        # Add feature extraction mode selection outside of tabs
        feature_mode_layout = QFormLayout()
        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems(["Predictions", "Embed Features"])
        self.feature_mode_combo.currentTextChanged.connect(self._on_selection_changed)
        feature_mode_layout.addRow("Feature Mode:", self.feature_mode_combo)
        main_layout.addLayout(feature_mode_layout)
        
        # Initialize the feature mode state and emit the first signal
        self._on_selection_changed()

    def browse_for_model(self):
        """Open a file dialog to browse for model files."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "PyTorch Models (*.pt);;All Files (*)",
            options=options
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    @pyqtSlot()
    def _on_selection_changed(self):
        """Central slot to handle any change in model selection and emit a single signal."""
        self._update_feature_mode_state()
        self.selection_changed.emit()

    def _update_feature_mode_state(self, *args):
        """Update the enabled state of the feature mode field based on the current model selection."""
        current_tab_index = self.tabs.currentIndex()
        is_color_features = False
        
        if current_tab_index == 0:
            # Select Model tab - check if Color Features is selected
            current_model = self.model_combo.currentText()
            is_color_features = current_model == "Color Features"
        elif current_tab_index == 1:
            # Use Existing Model tab - feature mode should always be enabled
            is_color_features = False
        
        # Enable feature mode only if not Color Features
        self.feature_mode_combo.setEnabled(not is_color_features)
        
        # Update the tooltip based on state
        if is_color_features:
            self.feature_mode_combo.setToolTip("Feature Mode is not available for Color Features")
        else:
            self.feature_mode_combo.setToolTip("Select the feature extraction mode")
    
    def get_selected_model(self):
        """Get the currently selected model name/path and feature mode."""
        current_tab_index = self.tabs.currentIndex()
        
        # Get model name/path and feature mode based on the active tab
        if current_tab_index == 0:
            model_name = self.model_combo.currentText()
        elif current_tab_index == 1:
            model_name = self.model_path_edit.text()
        else:
            return "", None
        
        feature_mode = self.feature_mode_combo.currentText() if self.feature_mode_combo.isEnabled() else "N/A"
        return model_name, feature_mode


class EmbeddingSettingsWidget(QGroupBox):
    """Widget containing settings with tabs for models and embedding."""

    def __init__(self, main_window, parent=None):
        super(EmbeddingSettingsWidget, self).__init__("Embedding Settings", parent)
        self.main_window = main_window
        self.explorer_window = parent

        self.setup_ui()

        # Initial call to set the sliders correctly for the default technique
        self._update_parameter_sliders()

    def setup_ui(self):
        """Set up the UI with embedding technique parameters."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 10, 5, 5)

        # Form layout for embedding settings
        settings_layout = QFormLayout()

        self.embedding_technique_combo = QComboBox()
        self.embedding_technique_combo.addItems(["PCA", "TSNE", "UMAP"])
        self.embedding_technique_combo.currentTextChanged.connect(self._update_parameter_sliders)
        settings_layout.addRow("Technique:", self.embedding_technique_combo)

        # Slider 1
        self.param1_label = QLabel("Parameter 1:")
        param1_layout = QHBoxLayout()
        self.param1_slider = QSlider(Qt.Horizontal)
        self.param1_value_label = QLabel("0")
        self.param1_value_label.setMinimumWidth(25)
        param1_layout.addWidget(self.param1_slider)
        param1_layout.addWidget(self.param1_value_label)
        settings_layout.addRow(self.param1_label, param1_layout)
        self.param1_slider.valueChanged.connect(lambda v: self.param1_value_label.setText(str(v)))

        # Slider 2
        self.param2_label = QLabel("Parameter 2:")
        param2_layout = QHBoxLayout()
        self.param2_slider = QSlider(Qt.Horizontal)
        self.param2_value_label = QLabel("0.0")
        self.param2_value_label.setMinimumWidth(35)  # Increased width for larger numbers
        param2_layout.addWidget(self.param2_slider)
        param2_layout.addWidget(self.param2_value_label)
        settings_layout.addRow(self.param2_label, param2_layout)

        self.apply_embedding_button = QPushButton("Apply Embedding")
        self.apply_embedding_button.clicked.connect(self.apply_embedding)
        settings_layout.addRow("", self.apply_embedding_button)

        main_layout.addLayout(settings_layout)

    def _update_parameter_sliders(self):
        """Enable, disable, and configure sliders based on the selected technique."""
        technique = self.embedding_technique_combo.currentText()

        # Disconnect any existing connections to prevent conflicts
        try:
            self.param2_slider.valueChanged.disconnect()
        except TypeError:
            pass  # No connection existed

        if technique == "UMAP":
            # Enable Row 1 for n_neighbors
            self.param1_label.setEnabled(True)
            self.param1_slider.setEnabled(True)
            self.param1_value_label.setEnabled(True)
            self.param1_label.setText("n_neighbors:")
            self.param1_slider.setRange(2, 150)
            self.param1_slider.setValue(15)

            # Enable Row 2 for min_dist
            self.param2_label.setEnabled(True)
            self.param2_slider.setEnabled(True)
            self.param2_value_label.setEnabled(True)
            self.param2_label.setText("min_dist:")
            self.param2_slider.setRange(0, 99)
            self.param2_slider.setValue(10)
            self.param2_slider.valueChanged.connect(lambda v: self.param2_value_label.setText(f"{v/100.0:.2f}"))

        elif technique == "TSNE":
            # Enable Row 1 for Perplexity
            self.param1_label.setEnabled(True)
            self.param1_slider.setEnabled(True)
            self.param1_value_label.setEnabled(True)
            self.param1_label.setText("Perplexity:")
            self.param1_slider.setRange(5, 50)
            self.param1_slider.setValue(30)

            # --- MODIFIED: Enable Row 2 for Early Exaggeration ---
            self.param2_label.setEnabled(True)
            self.param2_slider.setEnabled(True)
            self.param2_value_label.setEnabled(True)
            self.param2_label.setText("Exaggeration:")
            self.param2_slider.setRange(50, 600)  # Represents 5.0 to 60.0
            self.param2_slider.setValue(120)      # Represents 12.0
            self.param2_slider.valueChanged.connect(lambda v: self.param2_value_label.setText(f"{v/10.0:.1f}"))

        elif technique == "PCA":
            # Disable both rows for PCA and reset to minimum values
            self.param1_label.setEnabled(False)
            self.param1_slider.setEnabled(False)
            self.param1_value_label.setEnabled(False)
            self.param1_label.setText(" ")
            self.param1_slider.setValue(self.param1_slider.minimum())
            self.param1_value_label.setText(str(self.param1_slider.minimum()))

            self.param2_label.setEnabled(False)
            self.param2_slider.setEnabled(False)
            self.param2_value_label.setEnabled(False)
            self.param2_label.setText(" ")
            self.param2_slider.setValue(self.param2_slider.minimum())
            self.param2_value_label.setText(str(self.param2_slider.minimum()))

    def get_embedding_parameters(self):
        """Returns a dictionary of the current embedding parameters."""
        params = {
            'technique': self.embedding_technique_combo.currentText(),
        }
        if params['technique'] == 'UMAP':
            params['n_neighbors'] = self.param1_slider.value()
            params['min_dist'] = self.param2_slider.value() / 100.0
        elif params['technique'] == 'TSNE':
            params['perplexity'] = self.param1_slider.value()
            params['early_exaggeration'] = self.param2_slider.value() / 10.0
        return params

    def apply_embedding(self):
        if self.explorer_window and hasattr(self.explorer_window, 'run_embedding_pipeline'):
            # Clear all selections before running embedding pipeline
            if hasattr(self.explorer_window, 'handle_selection_change'):
                self.explorer_window.handle_selection_change([])

            self.explorer_window.run_embedding_pipeline()
