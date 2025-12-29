import warnings

from PyQt5.QtGui import QFont, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout,
                             QLabel, QPushButton, QRadioButton, QSlider,
                             QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
                             QDialog, QGroupBox, QComboBox, QDoubleSpinBox,
                             QTextEdit, QProgressBar, QFormLayout, QTabWidget,
                             QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

REVIEW_LABEL = 'Review'

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ClickablePredictionRow(QWidget):
    """Clickable widget for prediction rows that applies label when clicked."""
    clicked = pyqtSignal(str)  # Emits the label code when clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_code = None
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.is_hovered = False
        
    def set_label_code(self, label_code):
        """Store the label code associated with this row."""
        self.label_code = label_code
        
    def mousePressEvent(self, event):
        """Handle mouse clicks."""
        if event.button() == Qt.LeftButton and self.label_code:
            self.clicked.emit(self.label_code)
        super().mousePressEvent(event)
    
    def enterEvent(self, event):
        """Handle mouse enter for hover effect."""
        self.is_hovered = True
        self.setStyleSheet("ClickablePredictionRow { background-color: #f0f0f0; border-radius: 4px; }")
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave to remove hover effect."""
        self.is_hovered = False
        self.setStyleSheet("")
        super().leaveEvent(event)


class AutoAnnotationError(Exception):
    """Raised when the wizard cannot complete a step."""


class AutoAnnotationWizard(QDialog):
    """
    Interactive modeless wizard for ML-assisted annotation.
    Trains scikit-learn models on features and provides intelligent label suggestions.
    """
    
    annotations_updated = pyqtSignal(list)  # Emitted when annotations are labeled
    
    def __init__(self, explorer_window, parent=None):
        super().__init__(parent)
        self.explorer_window = explorer_window
        self.main_window = explorer_window.main_window
        
        # Make dialog modeless and stay on top
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setModal(False)
        
        self.setWindowTitle("Auto-Annotation Wizard")
        self.setMinimumSize(800, 600)
        
        # Wizard state
        self.current_page = 0
        self.trained_model = None
        self.scaler = None
        self.feature_type = 'full'  # 'full' or 'reduced'
        self.model_type = 'random_forest'
        self.label_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.training_score = 0.0
        self.labeled_count = 0
        self.auto_labeled_count = 0
        self.current_annotation_item = None
        
        # Track annotations completed in this session
        self.completed_annotation_ids = set()
        
        # Annotation mode: 'active_learning' or 'bulk_labeling'
        self.annotation_mode = 'active_learning'
        
        # Bulk labeling state
        self.bulk_predictions = {}  # annotation_id -> prediction dict
        self.bulk_confidence_threshold = 0.95  # Reset each session
        self.manual_bulk_overrides = {}  # annotation_id -> 'forced_review' or 'forced_accept'
        self.bulk_preview_timer = QTimer()
        self.bulk_preview_timer.setSingleShot(True)
        self.bulk_preview_timer.timeout.connect(self._apply_bulk_preview_labels)
        
        # Model parameters
        self.model_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'svc': {'C': 1.0, 'kernel': 'rbf', 'probability': True},
            'knn': {'n_neighbors': 5},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 10},
            'adaboost': {'n_estimators': 50, 'learning_rate': 1.0},
            'extra_trees': {'n_estimators': 100, 'max_depth': 10}
        }
        
        # Thresholds
        self.uncertainty_threshold = 0.60
        
        self.setup_ui()
        
        # Connect to label selection signal to capture manual label changes
        self.main_window.label_window.labelSelected.connect(self._on_label_manually_selected)
        
    def setup_ui(self):
        """Create the wizard interface with 3 pages."""
        main_layout = QVBoxLayout(self)
        
        # Page stack
        self.page_stack = QStackedWidget()
        main_layout.addWidget(self.page_stack)
        
        # Create pages
        self.setup_page = self._create_setup_page()
        self.training_page = self._create_training_page()
        self.annotation_page = self._create_annotation_page()
        
        self.page_stack.addWidget(self.setup_page)
        self.page_stack.addWidget(self.training_page)
        self.page_stack.addWidget(self.annotation_page)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("< Back")
        self.next_button = QPushButton("Next >")
        self.cancel_button = QPushButton("Cancel")
        
        # Retrain button in bottom-left
        self.retrain_now_button = QPushButton("🔄 Retrain Model")
        self.retrain_now_button.setToolTip("Manually trigger model retraining with current labels")
        self.retrain_now_button.clicked.connect(self._retrain_now)
        self.retrain_now_button.setVisible(False)  # Only show on annotation page
        
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        self.cancel_button.clicked.connect(self.close)
        
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.retrain_now_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_button)
        nav_layout.addWidget(self.next_button)
        
        main_layout.addLayout(nav_layout)
        
        self._update_navigation_buttons()
        
    def _create_setup_page(self):
        """Page 1: Model selection, feature choice, parameters."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Setup: Model & Feature Selection</h2>")
        layout.addWidget(title)
        
        # Information box
        info_box = QGroupBox("Information")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "Configure the machine learning model to assist with annotation labeling. "
            "Choose between full features (more accurate) or reduced embeddings (faster). "
            "Select a model algorithm and adjust hyperparameters based on your dataset size and complexity."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        
        # Create a more structured layout
        stats_layout = QFormLayout()
        
        self.dataset_total_label = QLabel("0")
        self.dataset_labeled_label = QLabel("0")
        self.dataset_review_label = QLabel("0")
        self.dataset_classes_label = QLabel("0")
        
        stats_layout.addRow("<b>Total annotations:</b>", self.dataset_total_label)
        stats_layout.addRow("<b>Labeled:</b>", self.dataset_labeled_label)
        stats_layout.addRow("<b>Review (unlabeled):</b>", self.dataset_review_label)
        stats_layout.addRow("<b>Unique classes:</b>", self.dataset_classes_label)
        
        info_layout.addLayout(stats_layout)
        
        # Warning label
        self.dataset_warning_label = QLabel()
        self.dataset_warning_label.setWordWrap(True)
        self.dataset_warning_label.setStyleSheet("QLabel { color: #ff9800; font-weight: bold; }")
        info_layout.addWidget(self.dataset_warning_label)
        
        layout.addWidget(info_group)
        
        # Feature selection
        feature_group = QGroupBox("Feature Type")
        feature_layout = QHBoxLayout(feature_group)
        
        self.feature_button_group = QButtonGroup()
        self.full_features_radio = QRadioButton("Full Features (Higher accuracy, slower)")
        self.reduced_features_radio = QRadioButton("Reduced Embeddings (Faster, less accurate)")
        
        self.full_features_radio.setToolTip("Use high-dimensional features from the feature store")
        self.reduced_features_radio.setToolTip("Use 2D/3D embeddings from visualization")
        
        self.feature_button_group.addButton(self.full_features_radio)
        self.feature_button_group.addButton(self.reduced_features_radio)
        self.full_features_radio.setChecked(True)
        
        # Connect to reset training when feature type changes
        self.full_features_radio.toggled.connect(self._on_setup_changed)
        self.reduced_features_radio.toggled.connect(self._on_setup_changed)
        
        feature_layout.addWidget(self.full_features_radio, 1)
        feature_layout.addWidget(self.reduced_features_radio, 1)
        layout.addWidget(feature_group)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest (Recommended)",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "Gradient Boosting",
            "AdaBoost",
            "Extra Trees"
        ])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_combo.currentIndexChanged.connect(self._on_setup_changed)
        model_layout.addRow("Algorithm:", self.model_combo)
        
        # Model-specific parameters
        self.param_stack = QStackedWidget()
        
        # Random Forest params
        rf_widget = QWidget()
        rf_layout = QFormLayout(rf_widget)
        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 500)
        self.rf_n_estimators.setValue(100)
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(3, 50)
        self.rf_max_depth.setValue(10)
        rf_layout.addRow("Trees:", self.rf_n_estimators)
        rf_layout.addRow("Max Depth:", self.rf_max_depth)
        self.param_stack.addWidget(rf_widget)
        
        # SVC params
        svc_widget = QWidget()
        svc_layout = QFormLayout(svc_widget)
        self.svc_c = QDoubleSpinBox()
        self.svc_c.setRange(0.01, 100.0)
        self.svc_c.setValue(1.0)
        self.svc_kernel = QComboBox()
        self.svc_kernel.addItems(['rbf', 'linear', 'poly'])
        svc_layout.addRow("C:", self.svc_c)
        svc_layout.addRow("Kernel:", self.svc_kernel)
        self.param_stack.addWidget(svc_widget)
        
        # KNN params
        knn_widget = QWidget()
        knn_layout = QFormLayout(knn_widget)
        self.knn_neighbors = QSpinBox()
        self.knn_neighbors.setRange(1, 50)
        self.knn_neighbors.setValue(5)
        knn_layout.addRow("Neighbors:", self.knn_neighbors)
        self.param_stack.addWidget(knn_widget)
        
        # Gradient Boosting params
        gb_widget = QWidget()
        gb_layout = QFormLayout(gb_widget)
        self.gb_n_estimators = QSpinBox()
        self.gb_n_estimators.setRange(10, 500)
        self.gb_n_estimators.setValue(100)
        self.gb_learning_rate = QDoubleSpinBox()
        self.gb_learning_rate.setRange(0.01, 1.0)
        self.gb_learning_rate.setValue(0.1)
        self.gb_max_depth = QSpinBox()
        self.gb_max_depth.setRange(3, 50)
        self.gb_max_depth.setValue(10)
        gb_layout.addRow("Trees:", self.gb_n_estimators)
        gb_layout.addRow("Learning Rate:", self.gb_learning_rate)
        gb_layout.addRow("Max Depth:", self.gb_max_depth)
        self.param_stack.addWidget(gb_widget)
        
        # AdaBoost params
        ab_widget = QWidget()
        ab_layout = QFormLayout(ab_widget)
        self.ab_n_estimators = QSpinBox()
        self.ab_n_estimators.setRange(10, 500)
        self.ab_n_estimators.setValue(50)
        self.ab_learning_rate = QDoubleSpinBox()
        self.ab_learning_rate.setRange(0.01, 2.0)
        self.ab_learning_rate.setValue(1.0)
        ab_layout.addRow("Trees:", self.ab_n_estimators)
        ab_layout.addRow("Learning Rate:", self.ab_learning_rate)
        self.param_stack.addWidget(ab_widget)
        
        # Extra Trees params
        et_widget = QWidget()
        et_layout = QFormLayout(et_widget)
        self.et_n_estimators = QSpinBox()
        self.et_n_estimators.setRange(10, 500)
        self.et_n_estimators.setValue(100)
        self.et_max_depth = QSpinBox()
        self.et_max_depth.setRange(3, 50)
        self.et_max_depth.setValue(10)
        et_layout.addRow("Trees:", self.et_n_estimators)
        et_layout.addRow("Max Depth:", self.et_max_depth)
        self.param_stack.addWidget(et_widget)
        
        model_layout.addRow(self.param_stack)
        layout.addWidget(model_group)
        
        # Training section
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout(training_group)
        
        # Progress
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        training_layout.addWidget(self.training_progress)
        
        # Train button
        self.train_button = QPushButton("🎯 Train Model")
        self.train_button.clicked.connect(self._train_model)
        self.train_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        training_layout.addWidget(self.train_button)
        
        layout.addWidget(training_group)
        
        layout.addStretch()
        
        return page
    
    def _create_training_page(self):
        """Page 2: Show training metrics and results."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Training Results: Model Performance</h2>")
        layout.addWidget(title)
        
        # Information box
        info_box = QGroupBox("Information")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "Review the model's accuracy metrics and confusion matrix to assess performance. "
            "A higher accuracy indicates better predictions. The confusion matrix shows which classes "
            "the model confuses most often."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Metrics display
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        # Use monospace font for proper alignment of classification report
        font = QFont("Courier New", 9)
        self.metrics_text.setFont(font)
        metrics_layout.addWidget(self.metrics_text)
        
        layout.addWidget(metrics_group)
        
        # Confusion matrix
        confusion_group = QGroupBox("Confusion Matrix (Training Data)")
        confusion_layout = QVBoxLayout(confusion_group)
        
        self.confusion_table = QTableWidget()
        self.confusion_table.setMaximumHeight(300)
        confusion_layout.addWidget(self.confusion_table)
        
        layout.addWidget(confusion_group)
        layout.addStretch()
        
        return page
    
    def _create_annotation_page(self):
        """Page 3: Dual-mode intelligent labeling."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Intelligent Labeling</h2>")
        layout.addWidget(title)
        
        # Tabbed interface for two modes
        self.annotation_tabs = QTabWidget()
        self.annotation_tabs.currentChanged.connect(self._on_annotation_mode_changed)
        
        # Mode 1: Active Learning (uncertainty-driven one-by-one)
        active_learning_tab = self._create_active_learning_tab()
        self.annotation_tabs.addTab(active_learning_tab, "🎯 Active Learning")
        
        # Mode 2: Bulk Labeling (confidence-threshold-based)
        bulk_labeling_tab = self._create_bulk_labeling_tab()
        self.annotation_tabs.addTab(bulk_labeling_tab, "⚡ Bulk Labeling")
        
        layout.addWidget(self.annotation_tabs)
        layout.addStretch()
        
        return page
    
    def _create_active_learning_tab(self):
        """Create the Active Learning mode tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Information box
        info_box = QGroupBox("Active Learning Mode")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "Review annotations one-by-one, prioritized by model uncertainty. "
            "This focuses on difficult cases to improve the model most efficiently. "
            "Use the 'Retrain Model' button to update predictions with your new labels. "
            "Annotations are shown in the viewer below.<br><br>"
            "<b>💡 Click on a prediction:</b> Directly applies that label to the annotation.<br>"
            "<b>⊘ Skip:</b> Skips the current annotation without labeling (leaves as 'Review')."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Progress info
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("%p% Complete")
        progress_layout.addWidget(self.overall_progress_bar)
        
        stats_layout = QFormLayout()
        self.progress_label = QLabel("0 / 0")
        self.remaining_review_label = QLabel("0")
        self.model_accuracy_label = QLabel("N/A")
        
        stats_layout.addRow("<b>Total labeled:</b>", self.progress_label)
        stats_layout.addRow("<b>Remaining 'Review':</b>", self.remaining_review_label)
        stats_layout.addRow("<b>Model accuracy:</b>", self.model_accuracy_label)
        
        progress_layout.addLayout(stats_layout)
        layout.addWidget(progress_group)
        
        # Current annotation info
        current_group = QGroupBox("Current Annotation")
        current_layout = QVBoxLayout(current_group)
        
        self.current_annotation_label = QLabel("No annotation selected")
        self.current_annotation_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11pt; }")
        current_layout.addWidget(self.current_annotation_label)
        
        current_label_layout = QHBoxLayout()
        current_label_layout.addWidget(QLabel("<b>Current Label:</b>"))
        self.current_label_display = QLabel("-")
        self.current_label_display.setStyleSheet("QLabel { color: #d32f2f; }")
        current_label_layout.addWidget(self.current_label_display)
        current_label_layout.addStretch()
        current_layout.addLayout(current_label_layout)
        
        # Top 3 predictions with confidence bars
        pred_group = QGroupBox("Top 3 Predictions")
        pred_group_layout = QVBoxLayout(pred_group)
        
        hint_label = QLabel("<i><small>💡 Click on a prediction to apply that label</small></i>")
        hint_label.setStyleSheet("QLabel { color: #666; }")
        pred_group_layout.addWidget(hint_label)
        
        # Create 3 prediction rows with labels and bars
        self.prediction_rows = []
        for i in range(3):
            row_widget = ClickablePredictionRow()
            row_widget.clicked.connect(self._apply_label_from_prediction)
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 6, 8, 6)
            row_layout.setSpacing(2)
            
            # Rank and label on one line
            label_layout = QHBoxLayout()
            rank_label = QLabel(f"<b>{i+1}.</b>")
            rank_label.setFixedWidth(20)
            label_layout.addWidget(rank_label)
            
            pred_label = QLabel("-")
            pred_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11pt; }")
            label_layout.addWidget(pred_label)
            
            conf_label = QLabel("-")
            conf_label.setStyleSheet("QLabel { color: #666; }")
            conf_label.setFixedWidth(60)
            conf_label.setAlignment(Qt.AlignRight)
            label_layout.addWidget(conf_label)
            row_layout.addLayout(label_layout)
            
            # Confidence bar
            conf_bar = QProgressBar()
            conf_bar.setMaximumHeight(18)
            conf_bar.setTextVisible(False)
            row_layout.addWidget(conf_bar)
            
            pred_group_layout.addWidget(row_widget)
            
            # Store references
            self.prediction_rows.append({
                'widget': row_widget,
                'label': pred_label,
                'confidence_label': conf_label,
                'confidence_bar': conf_bar
            })
        
        current_layout.addWidget(pred_group)
        
        # Skip button at bottom of current group (full width)
        self.skip_button = QPushButton("⊘ Skip Annotation")
        self.skip_button.setToolTip("Skip this annotation without labeling (leaves as 'Review')")
        self.skip_button.clicked.connect(self._skip_annotation)
        self.skip_button.setStyleSheet("QPushButton { padding: 8px; }")
        current_layout.addWidget(self.skip_button)
        
        layout.addWidget(current_group)
        layout.addStretch()
        
        return tab
    
    def _create_bulk_labeling_tab(self):
        """Create the Bulk Labeling mode tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Information box
        info_box = QGroupBox("Bulk Labeling Mode")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "Interactively adjust the confidence threshold to preview automatic label assignments. "
            "Annotations are sorted by predicted label in the viewer below. "
            "Adjust the slider to see which annotations will be auto-labeled at different thresholds.<br><br>"
            "<b>Click Annotations:</b> Click any annotation to manually override the threshold. "
            "Clicking an annotation above the threshold moves it to 'Review'; "
            "clicking an annotation in 'Review' accepts it at the predicted label. "
            "All manual overrides are cleared when you adjust the threshold slider.<br><br>"
            "<b>Apply Auto-Labels:</b> Permanently applies all previewed labels to annotations. "
            "Once applied, labels cannot be undone within the wizard."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Progress display
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.bulk_overall_progress_bar = QProgressBar()
        self.bulk_overall_progress_bar.setTextVisible(True)
        self.bulk_overall_progress_bar.setFormat("%p% Complete")
        progress_layout.addWidget(self.bulk_overall_progress_bar)
        
        stats_layout = QFormLayout()
        self.bulk_progress_label = QLabel("0 / 0")
        self.bulk_remaining_review_label = QLabel("0")
        self.bulk_model_accuracy_label = QLabel("N/A")
        
        stats_layout.addRow("<b>Total labeled:</b>", self.bulk_progress_label)
        stats_layout.addRow("<b>Remaining 'Review':</b>", self.bulk_remaining_review_label)
        stats_layout.addRow("<b>Model accuracy:</b>", self.bulk_model_accuracy_label)
        
        progress_layout.addLayout(stats_layout)
        layout.addWidget(progress_group)
        
        # Confidence threshold control
        threshold_group = QGroupBox("Confidence Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # Slider with labels
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("50%"))
        
        self.bulk_threshold_slider = QSlider(Qt.Horizontal)
        self.bulk_threshold_slider.setMinimum(50)
        self.bulk_threshold_slider.setMaximum(100)
        self.bulk_threshold_slider.setValue(95)
        self.bulk_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.bulk_threshold_slider.setTickInterval(10)
        self.bulk_threshold_slider.valueChanged.connect(self._on_bulk_threshold_changed)
        slider_layout.addWidget(self.bulk_threshold_slider)
        
        slider_layout.addWidget(QLabel("100%"))
        threshold_layout.addLayout(slider_layout)
        
        # Current value and count display
        info_layout2 = QHBoxLayout()
        self.bulk_threshold_label = QLabel("<b>Current Threshold: 95%</b>")
        self.bulk_threshold_label.setStyleSheet("QLabel { font-size: 11pt; }")
        info_layout2.addWidget(self.bulk_threshold_label)
        info_layout2.addStretch()
        threshold_layout.addLayout(info_layout2)
        
        self.bulk_count_label = QLabel("Annotations to be auto-labeled: 0")
        self.bulk_count_label.setStyleSheet("QLabel { color: #1976d2; font-weight: bold; }")
        threshold_layout.addWidget(self.bulk_count_label)
        
        layout.addWidget(threshold_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.apply_bulk_button = QPushButton("✓ Apply Auto-Labels")
        self.apply_bulk_button.setToolTip("Permanently apply all preview label assignments shown below")
        self.apply_bulk_button.clicked.connect(self._apply_bulk_labels_permanently)
        self.apply_bulk_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        actions_layout.addWidget(self.apply_bulk_button)
        
        layout.addWidget(actions_group)
        layout.addStretch()
        
        return tab
    
    def _on_model_changed(self, index):
        """Update parameter widget when model selection changes."""
        self.param_stack.setCurrentIndex(index)
    
    def _on_setup_changed(self):
        """Reset training state when setup parameters change."""
        # Clear trained model and results
        self.trained_model = None
        self.scaler = None
        self.label_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.training_score = 0.0
        
        # Clear bulk predictions since they're model-specific
        self.bulk_predictions = {}
        
        # Reset completed annotations tracking
        self.completed_annotation_ids = set()
        
        # Reset training page UI
        self.metrics_text.clear()
        self.confusion_table.setRowCount(0)
        self.confusion_table.setColumnCount(0)
        
        # Disable navigation to annotation page
        if self.current_page == 1:
            self._update_navigation_buttons()
    
    def _on_annotation_mode_changed(self, index):
        """Handle switching between Active Learning and Bulk Labeling modes."""
        # Clear animations and selections when switching modes
        self._clear_animations_and_selections()
        
        # Rescan for any manually labeled annotations before entering new mode
        self._rescan_for_manual_labels()
        
        if index == 0:
            self.annotation_mode = 'active_learning'
            self._enter_active_learning_mode()
        elif index == 1:
            self.annotation_mode = 'bulk_labeling'
            self._enter_bulk_labeling_mode()
    
    def _rescan_for_manual_labels(self):
        """Scan all annotations and mark any that are no longer 'Review' as completed.
        
        This ensures that manually labeled annotations (changed outside the wizard)
        are properly tracked and excluded from future selections.
        
        Also removes items from completed set if they've been relabeled back to 'Review'.
        
        Returns:
            int: Number of newly detected manually labeled annotations
        """
        newly_labeled_count = 0
        relabeled_to_review_count = 0
        
        for item in self.explorer_window.current_data_items:
            current_label = getattr(item.annotation.label, 'short_label_code', '')
            ann_id = item.annotation.id
            
            # Check if this annotation was completed but is now 'Review' again
            if ann_id in self.completed_annotation_ids and current_label == REVIEW_LABEL:
                # Remove from completed set - user relabeled it back to Review
                self.completed_annotation_ids.remove(ann_id)
                relabeled_to_review_count += 1
                
                # Clear any existing prediction for this item
                if ann_id in self.bulk_predictions:
                    del self.bulk_predictions[ann_id]
            
            # Check if this annotation is not in completed set but is also not 'Review'
            elif ann_id not in self.completed_annotation_ids:
                # If it's not Review, it was manually labeled
                if current_label and current_label != REVIEW_LABEL:
                    # Mark as completed
                    self.completed_annotation_ids.add(ann_id)
                    newly_labeled_count += 1
                    
                    # Remove from bulk predictions if it was there
                    if item.annotation.id in self.bulk_predictions:
                        del self.bulk_predictions[item.annotation.id]
                    
                    # Update tooltip and visuals
                    if hasattr(item, 'widget') and item.widget:
                        item.widget.update_tooltip()
                    if hasattr(item, 'graphics_item') and item.graphics_item:
                        item.graphics_item.update_tooltip()
        
        return newly_labeled_count
    
    def _clear_animations_and_selections(self):
        """Clear all animations and selections from viewers."""
        # Clear animations
        for item in self.explorer_window.current_data_items:
            if hasattr(item, 'graphics_item') and item.graphics_item:
                if hasattr(item.graphics_item, 'deanimate'):
                    item.graphics_item.deanimate()
            if hasattr(item, 'widget') and item.widget:
                if hasattr(item.widget, 'deanimate'):
                    item.widget.deanimate()
        
        # Clear selection in embedding viewer
        if hasattr(self.explorer_window, 'embedding_viewer'):
            self.explorer_window.embedding_viewer.graphics_scene.clearSelection()
    
    def _update_navigation_buttons(self):
        """Update button states based on current page."""
        self.back_button.setEnabled(self.current_page > 0)
        
        if self.current_page == 0:
            # Setup page - train button handles progression, no next button needed
            self.next_button.setVisible(False)
            self.retrain_now_button.setVisible(False)
        elif self.current_page == 1:
            # Results page - can proceed to annotation
            self.next_button.setVisible(True)
            self.next_button.setText("Start Annotating >")
            self.next_button.setEnabled(True)
            self.retrain_now_button.setVisible(False)
        elif self.current_page == 2:
            # Annotation page
            self.next_button.setVisible(True)
            self.next_button.setText("Exit")
            self.next_button.setEnabled(True)
            self.retrain_now_button.setVisible(True)
    
    def _go_back(self):
        """Navigate to previous page."""
        if self.current_page > 0:
            # If going back from annotation page, restore original selection handler
            if self.current_page == 2:
                if hasattr(self, '_original_selection_handler') and hasattr(self, '_bulk_click_handler_connected'):
                    if self._bulk_click_handler_connected:
                        handler = self.explorer_window.annotation_viewer.handle_annotation_selection
                        self.explorer_window.annotation_viewer.handle_annotation_selection = handler
                        self._bulk_click_handler_connected = False
                        delattr(self, '_original_selection_handler')
                
                # Clear manual overrides
                self.manual_bulk_overrides.clear()
                
                # Re-enable sort combo if it was disabled
                if hasattr(self.explorer_window.annotation_viewer, 'sort_combo'):
                    self.explorer_window.annotation_viewer.sort_combo.setEnabled(True)
            
            self.current_page -= 1
            self.page_stack.setCurrentIndex(self.current_page)
            self._update_navigation_buttons()
    
    def _go_to_page(self, page_index):
        """Navigate to a specific page."""
        if 0 <= page_index < self.page_stack.count():
            self.current_page = page_index
            self.page_stack.setCurrentIndex(self.current_page)
            self._update_navigation_buttons()
    
    def _go_next(self):
        """Navigate to next page."""
        if self.current_page == 0:
            # Moving from setup to training
            self._initialize_from_setup()
            self.current_page = 1
            self.page_stack.setCurrentIndex(self.current_page)
        elif self.current_page == 1:
            # Moving from training to annotation
            if self.trained_model is None:
                QMessageBox.warning(self, 
                                    "Not Ready", 
                                    "Please train the model before proceeding.")
                return
            self._start_annotation()
            self.current_page = 2
            self.page_stack.setCurrentIndex(self.current_page)
        elif self.current_page == 2:
            # Exit (user wants to leave early or is done)
            self._exit()
            
        self._update_navigation_buttons()
    
    def _initialize_from_setup(self):
        """Initialize wizard state from setup page selections."""
        # Get feature type
        self.feature_type = 'full' if self.full_features_radio.isChecked() else 'reduced'
        
        # Get model type
        model_index = self.model_combo.currentIndex()
        model_types = ['random_forest', 'svc', 'knn', 'gradient_boosting', 'adaboost', 'extra_trees']
        self.model_type = model_types[model_index]
        
        # Get model parameters
        if self.model_type == 'random_forest':
            self.model_params['random_forest']['n_estimators'] = self.rf_n_estimators.value()
            self.model_params['random_forest']['max_depth'] = self.rf_max_depth.value()
        elif self.model_type == 'svc':
            self.model_params['svc']['C'] = self.svc_c.value()
            self.model_params['svc']['kernel'] = self.svc_kernel.currentText()
        elif self.model_type == 'knn':
            self.model_params['knn']['n_neighbors'] = self.knn_neighbors.value()
        elif self.model_type == 'gradient_boosting':
            self.model_params['gradient_boosting']['n_estimators'] = self.gb_n_estimators.value()
            self.model_params['gradient_boosting']['learning_rate'] = self.gb_learning_rate.value()
            self.model_params['gradient_boosting']['max_depth'] = self.gb_max_depth.value()
        elif self.model_type == 'adaboost':
            self.model_params['adaboost']['n_estimators'] = self.ab_n_estimators.value()
            self.model_params['adaboost']['learning_rate'] = self.ab_learning_rate.value()
        elif self.model_type == 'extra_trees':
            self.model_params['extra_trees']['n_estimators'] = self.et_n_estimators.value()
            self.model_params['extra_trees']['max_depth'] = self.et_max_depth.value()
        
        # Update dataset info
        self._update_dataset_info()
    
    def _update_dataset_info(self):
        """Update dataset information display."""
        data_items = self.explorer_window.current_data_items
        total = len(data_items)
        review_count = sum(
            1 for item in data_items
            if getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        )
        labeled_count = total - review_count
        
        # Count unique labels (excluding Review)
        unique_labels = set()
        for item in data_items:
            label_code = getattr(item.effective_label, 'short_label_code', '')
            if label_code and label_code != REVIEW_LABEL:
                unique_labels.add(label_code)
        
        # Update labels
        self.dataset_total_label.setText(str(total))
        self.dataset_labeled_label.setText(str(labeled_count))
        self.dataset_review_label.setText(str(review_count))
        self.dataset_classes_label.setText(str(len(unique_labels)))
        
        # Update warnings
        warning_text = ""
        if len(unique_labels) < 2:
            warning_text = "⚠️ Warning: Need at least 2 classes to train model"
        elif labeled_count < 10:
            warning_text = f"⚠️ Warning: Only {labeled_count} labeled examples (cold start)"
        
        self.dataset_warning_label.setText(warning_text)
        self.dataset_warning_label.setVisible(bool(warning_text))
    
    def _train_model(self):
        """Train the ML model."""
        self.training_progress.setVisible(True)
        self.training_progress.setRange(0, 0)  # Indeterminate
        self.train_button.setEnabled(False)
        
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            # Call explorer's training method
            result = self.explorer_window.train_annotation_model(
                feature_type=self.feature_type,
                model_type=self.model_type,
                model_params=self.model_params[self.model_type]
            )
            
            if result is None:
                raise AutoAnnotationError("Training failed - insufficient data or invalid configuration")
            
            self.trained_model = result['model']
            self.scaler = result['scaler']
            self.label_classes = result['classes']
            self.class_to_idx = result['class_to_idx']
            self.idx_to_class = result['idx_to_class']
            self.training_score = result['accuracy']
            
            # Display metrics
            self._display_training_metrics(result)
            
            # Update ALL annotations with sklearn predictions for confidence display
            self.explorer_window.update_all_sklearn_predictions(
                self.trained_model,
                self.scaler,
                self.class_to_idx,
                self.feature_type
            )
            
            # Auto-advance to results page
            QTimer.singleShot(500, lambda: self._go_to_page(1))
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
        finally:
            self.training_progress.setVisible(False)
            self.train_button.setEnabled(True)
            # Restore cursor
            QApplication.restoreOverrideCursor()
        
        self._update_navigation_buttons()
    
    def _display_training_metrics(self, result):
        """Display training metrics and confusion matrix."""
        # Format metrics text
        metrics_text = f"Overall Accuracy: {result['accuracy']:.2%}\n\n"
        metrics_text += "Per-Class Metrics:\n"
        metrics_text += result['report']
        
        self.metrics_text.setText(metrics_text)
        
        # Display confusion matrix
        cm = result['confusion_matrix']
        classes = result['classes']
        
        self.confusion_table.setRowCount(len(classes))
        self.confusion_table.setColumnCount(len(classes))
        self.confusion_table.setHorizontalHeaderLabels(classes)
        self.confusion_table.setVerticalHeaderLabels(classes)
        
        for i in range(len(classes)):
            for j in range(len(classes)):
                item = QTableWidgetItem(str(cm[i, j]))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight diagonal
                if i == j:
                    item.setBackground(Qt.lightGray)
                self.confusion_table.setItem(i, j, item)
        
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.confusion_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
    
    def _start_annotation(self):
        """Start the annotation process."""
        # Enable selection blocking to keep user focused on wizard workflow
        self.explorer_window.embedding_viewer.selection_blocked = True
        self.explorer_window.annotation_viewer.selection_blocked = True
        
        # Generate predictions for all Review annotations (if not already done)
        if not self.bulk_predictions:
            self._generate_all_predictions()
        
        # Initialize based on current mode (from the active tab)
        current_tab = self.annotation_tabs.currentIndex()
        if current_tab == 0:
            self.annotation_mode = 'active_learning'
            self._enter_active_learning_mode()
        else:
            self.annotation_mode = 'bulk_labeling'
            self._enter_bulk_labeling_mode()
    
    def _enter_active_learning_mode(self):
        """Enter Active Learning mode."""
        # Check if model is trained
        if self.trained_model is None:
            return
        
        # Rescan for manually labeled annotations before entering mode
        self._rescan_for_manual_labels()
        
        # Clear any preview labels from bulk mode
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        # Regenerate all predictions to ensure they're current
        review_items = [
            item for item in self.explorer_window.current_data_items
            if item.annotation.id not in self.completed_annotation_ids
            and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        ]
        
        if review_items:
            self._generate_predictions_for_items(review_items)
        else:
            self.current_annotation_item = None
        
        # Exit isolation mode if active and show all data items
        if self.explorer_window.annotation_viewer.isolated_mode:
            self.explorer_window.annotation_viewer.show_all_annotations()
        
        # Update viewer with all data items to show all label bins
        self.explorer_window.annotation_viewer.update_annotations(self.explorer_window.current_data_items)
        
        # Set annotation viewer to sort by label and lock it
        self.explorer_window.annotation_viewer.sort_combo.setCurrentText("Label")
        self.explorer_window.annotation_viewer.sort_combo.setEnabled(False)
        
        self._update_progress_display()
        self._get_next_uncertain_annotation()
        self._show_current_annotation()
    
    def _enter_bulk_labeling_mode(self):
        """Enter Bulk Labeling mode."""
        
        # First, rescan for manually labeled annotations
        self._rescan_for_manual_labels()
        
        # Clear all existing preview labels before entering mode
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        # Get current review annotations, excluding completed ones from this session
        review_items = [
            item for item in self.explorer_window.current_data_items
            if item.annotation.id not in self.completed_annotation_ids
            and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        ]
        
        # Clear any stale predictions for items that are no longer Review
        review_ids = {item.annotation.id for item in review_items}
        stale_ids = [ann_id for ann_id in self.bulk_predictions.keys() if ann_id not in review_ids]
        if stale_ids:
            for ann_id in stale_ids:
                del self.bulk_predictions[ann_id]
        
        # Regenerate predictions for all review items to ensure they're current
        if review_items and self.trained_model is not None:
            self._generate_predictions_for_items(review_items)
        elif not review_items:
            self.current_annotation_item = None
        
        # Exit isolation mode if active
        if self.explorer_window.annotation_viewer.isolated_mode:
            self.explorer_window.annotation_viewer.show_all_annotations()
        
        # Set annotation viewer to sort by label and lock it
        self.explorer_window.annotation_viewer.sort_combo.setCurrentText("Label")
        self.explorer_window.annotation_viewer.sort_combo.setEnabled(False)
        
        # Connect click handler for manual override functionality
        # Store reference to disconnect later
        if not hasattr(self, '_bulk_click_handler_connected'):
            self._bulk_click_handler_connected = False
        
        if not self._bulk_click_handler_connected:
            # We'll override the selection handler temporarily
            self._original_selection_handler = self.explorer_window.annotation_viewer.handle_annotation_selection
            self.explorer_window.annotation_viewer.handle_annotation_selection = self._on_bulk_annotation_clicked
            self._bulk_click_handler_connected = True
        
        # Update the viewer with only Review data items to show proper label bins
        self.explorer_window.annotation_viewer.update_annotations(review_items)
        
        # Update statistics (will recount from scratch)
        self._update_bulk_statistics()
        
        # Apply threshold to show preview labels
        self._apply_bulk_preview_labels()
    
    def _generate_predictions_for_items(self, items, progress_bar=None):
        """Generate predictions for specific items."""
        if not items:
            return
        
        total = len(items)
        for i, item in enumerate(items):
            try:
                # predict_with_model returns None but stores predictions in item.ml_prediction
                self.explorer_window.predict_with_model(
                    [item],
                    self.trained_model,
                    self.scaler,
                    self.class_to_idx,
                    self.idx_to_class,
                    self.feature_type
                )
                
                # Collect the predictions that were stored in the items
                if hasattr(item, 'ml_prediction') and item.ml_prediction:
                    self.bulk_predictions[item.annotation.id] = item.ml_prediction
                
                if progress_bar:
                    percentage = int((i + 1) / total * 100)
                    if percentage % 10 == 0 or i == total - 1:
                        progress_bar.update_progress_percentage(percentage)
                        
            except Exception as e:
                print(f"Error generating prediction for annotation ID {item.annotation.id}: {e}")
    
    def _generate_all_predictions(self):
        """Generate predictions for all Review annotations, excluding completed ones."""
        # Validate model and scaler
        if self.trained_model is None or self.scaler is None:
            return
        
        # Get review items excluding those completed in this session
        # Note: We rely on completed_annotation_ids as the source of truth
        # Even if an annotation appears as 'Review' in the cache, if it's in completed_annotation_ids, skip it
        review_items = [
            item for item in self.explorer_window.current_data_items
            if item.annotation.id not in self.completed_annotation_ids
            and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        ]
        
        if not review_items:
            return
        
        progress_bar = ProgressBar(self, "Generating Predictions")
        progress_bar.start_progress(100)
        progress_bar.show()
        self._generate_predictions_for_items(review_items, progress_bar)
        progress_bar.finish_progress()
        progress_bar.close()
    
    def _update_progress_display(self):
        """Update progress statistics for Active Learning mode."""
        data_items = self.explorer_window.current_data_items
        total = len(data_items)
        
        # Count review items excluding completed annotations
        review_count = sum(1 for item in data_items
                           if item.annotation.id not in self.completed_annotation_ids
                           and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL)
        
        labeled = total - review_count
        
        self.progress_label.setText(f"{labeled} / {total}")
        self.remaining_review_label.setText(str(review_count))
        self.model_accuracy_label.setText(f"{self.training_score:.1%}")
        
        # Update progress bar (0-100%)
        if total > 0:
            progress_percent = int((labeled / total) * 100)
            self.overall_progress_bar.setValue(progress_percent)
    
    def _update_bulk_statistics(self):
        """Update statistics for Bulk Labeling mode.
        
        Recounts from scratch to ensure accurate display after mode switches.
        """
        data_items = self.explorer_window.current_data_items
        total = len(data_items)
        
        # Recount review items from scratch, excluding completed annotations
        # and checking actual current label state
        review_count = 0
        for item in data_items:
            if item.annotation.id not in self.completed_annotation_ids:
                current_label = getattr(item.annotation.label, 'short_label_code', '')
                if current_label == REVIEW_LABEL:
                    review_count += 1
        
        labeled = total - review_count
        
        self.bulk_progress_label.setText(f"{labeled} / {total}")
        self.bulk_remaining_review_label.setText(str(review_count))
        self.bulk_model_accuracy_label.setText(f"{self.training_score:.1%}")
        
        # Update progress bar (0-100%)
        if total > 0:
            progress_percent = int((labeled / total) * 100)
            self.bulk_overall_progress_bar.setValue(progress_percent)
    
    def _on_bulk_threshold_changed(self, value):
        """Handle bulk threshold slider change with debouncing."""
        threshold = value / 100.0
        self.bulk_confidence_threshold = threshold
        
        # Clear all manual overrides when threshold changes
        self.manual_bulk_overrides.clear()
        
        # Update label
        self.bulk_threshold_label.setText(f"<b>Current Threshold: {value}%</b>")
        
        # Debounce: restart timer
        self.bulk_preview_timer.stop()
        self.bulk_preview_timer.start(400)  # 400ms debounce
    
    def _apply_bulk_preview_labels(self):
        """Apply preview labels based on current confidence threshold and manual overrides."""
        threshold = self.bulk_confidence_threshold
        count = 0
        above_threshold = 0
        manual_review_count = 0
        manual_accept_count = 0
        
        # Clear all previews first
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        # Apply previews for annotations above threshold, respecting manual overrides
        for item in self.explorer_window.current_data_items:
            ann_id = item.annotation.id
            
            # Only process Review annotations with predictions
            if getattr(item.effective_label, 'short_label_code', '') != REVIEW_LABEL:
                continue
            
            if ann_id not in self.bulk_predictions:
                continue
            
            pred = self.bulk_predictions[ann_id]
            
            # Check for manual override
            override_state = self.manual_bulk_overrides.get(ann_id)
            
            if override_state == 'forced_review':
                # User manually moved this to Review - skip it
                manual_review_count += 1
                continue
            elif override_state == 'forced_accept':
                # User manually accepted this - apply label regardless of threshold
                manual_accept_count += 1
                label_obj = self._get_label_by_code(pred['label'])
                if label_obj:
                    item.set_preview_label(label_obj)
                    count += 1
            elif pred['confidence'] >= threshold:
                # Normal threshold-based labeling
                above_threshold += 1
                label_obj = self._get_label_by_code(pred['label'])
                if label_obj:
                    item.set_preview_label(label_obj)
                    count += 1
        
        # Update count display
        override_text = ""
        if manual_accept_count > 0 or manual_review_count > 0:
            override_text = f" ({manual_accept_count} manual accepts, {manual_review_count} forced reviews)"
        self.bulk_count_label.setText(f"Annotations to be auto-labeled: {count}{override_text}")
        
        # Refresh annotation viewer to show preview changes
        self.explorer_window.annotation_viewer.recalculate_layout()
    
    def _clear_bulk_previews(self):
        """Clear all bulk preview labels."""
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        self.bulk_count_label.setText("Annotations to be auto-labeled: 0")
        self.explorer_window.annotation_viewer.recalculate_layout()
    
    def _on_bulk_annotation_clicked(self, widget, event):
        """Handle annotation clicks in bulk labeling mode to toggle manual overrides."""
        # Only handle left-click
        if event.button() != Qt.LeftButton:
            # Restore original behavior for other buttons
            if hasattr(self, '_original_selection_handler'):
                self._original_selection_handler(widget, event)
            return
        
        data_item = widget.data_item
        ann_id = data_item.annotation.id
        
        # Only process annotations that have predictions
        if ann_id not in self.bulk_predictions:
            return
                
        # Determine current state
        has_preview = data_item.has_preview_changes()
        
        # Toggle logic:
        # - If annotation has a preview label (above threshold or manually accepted): move to Review
        # - If annotation is in Review (no preview): accept at predicted label
        
        if has_preview:
            # Currently shown as labeled - move to Review
            self.manual_bulk_overrides[ann_id] = 'forced_review'
        else:
            # Currently in Review - accept at predicted label
            self.manual_bulk_overrides[ann_id] = 'forced_accept'
        
        # Reapply preview labels with the new override
        self._apply_bulk_preview_labels()
        
        event.accept()
    
    def _apply_bulk_labels_permanently(self):
        """Apply all bulk preview labels permanently."""
        # Count preview changes
        preview_count = sum(1 for item in self.explorer_window.current_data_items if item.has_preview_changes())
        
        if preview_count == 0:
            QMessageBox.information(self, "No Changes", "No preview labels to apply.")
            return
        
        reply = QMessageBox.question(
            self,
            "Apply Auto-Labels",
            f"Permanently apply {preview_count} auto-labels?\n\nThis action cannot be undone within the wizard.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Apply all previews permanently and track completed IDs
            for item in self.explorer_window.current_data_items:
                if item.has_preview_changes():
                    # Apply the preview label permanently
                    item.apply_preview_permanently()
                    
                    # Update the data item's effective_label cache
                    item._effective_label = item.annotation.label
                    
                    # Track as completed
                    self.completed_annotation_ids.add(item.annotation.id)
                    
                    # Remove from predictions
                    if item.annotation.id in self.bulk_predictions:
                        del self.bulk_predictions[item.annotation.id]
                    
                    # Update tooltip and visuals
                    if hasattr(item, 'widget') and item.widget:
                        item.widget.update_tooltip()
                    if hasattr(item, 'graphics_item') and item.graphics_item:
                        item.graphics_item.update_tooltip()
            
            # Get remaining review items (excluding completed ones)
            # completed_annotation_ids is the authoritative exclusion list
            review_items = [
                item for item in self.explorer_window.current_data_items
                if item.annotation.id not in self.completed_annotation_ids
                and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
            ]
            
            # Regenerate predictions for remaining items
            if review_items:
                self._generate_predictions_for_items(review_items)
            
            # Update viewer with remaining items
            self.explorer_window.annotation_viewer.update_annotations(review_items)
            
            # Update statistics
            self._update_bulk_statistics()
            self.bulk_count_label.setText("Annotations to be auto-labeled: 0")
            
            # Check if all annotations are now labeled
            if len(review_items) == 0:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully applied {preview_count} labels.\n\n✓ All annotations have been labeled!"
                )
                self._cleanup_on_completion(auto_close=True)
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully applied {preview_count} labels.\n\n{len(review_items)} Review annotations remaining."
                )
    
    def _get_next_uncertain_annotation(self):
        """Get the next most uncertain annotation to review."""
        # Validate model and scaler
        if self.trained_model is None or self.scaler is None:
            self.current_annotation_item = None
            return
        
        try:
            # Get all review items that haven't been completed
            review_items = [
                item for item in self.explorer_window.current_data_items
                if item.annotation.id not in self.completed_annotation_ids
                and getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
            ]
            
            if not review_items:
                self.current_annotation_item = None
                return
            
            # Find the most uncertain annotation (lowest confidence)
            most_uncertain = None
            lowest_confidence = 1.0
            
            for item in review_items:
                if hasattr(item, 'ml_prediction') and item.ml_prediction:
                    confidence = item.ml_prediction['confidence']
                    if confidence < lowest_confidence:
                        lowest_confidence = confidence
                        most_uncertain = item
            
            if most_uncertain is None:
                # No predictions available, just take first review item
                most_uncertain = review_items[0]
            
            self.current_annotation_item = most_uncertain
            
            # Highlight in embedding viewer
            self.explorer_window.embedding_viewer.render_selection_from_ids([most_uncertain.annotation.id])
            
        except Exception as e:
            self.current_annotation_item = None
    
    def _show_current_annotation(self):
        """Display information about current annotation."""
        if not self.current_annotation_item:
            # Check if there are any remaining review annotations at all
            total_review = sum(1 for i in self.explorer_window.current_data_items
                               if i.annotation.id not in self.completed_annotation_ids
                               and getattr(i.effective_label, 'short_label_code', '') == REVIEW_LABEL)
            
            if total_review == 0:
                # Truly done - no more review annotations
                self.current_annotation_label.setText(
                    "✓ All Done! No more uncertain annotations to label.<br>"
                    "<small>All annotations have been processed.</small>"
                )
                self._cleanup_on_completion(auto_close=True)
            else:
                # Still have review annotations - need to get next one
                self.current_annotation_label.setText(
                    f"{total_review} 'Review' annotations remain.<br>"
                    "<small>Click 'Exit' to close the wizard.</small>"
                )
                self._cleanup_on_completion(auto_close=False)
            return
        
        item = self.current_annotation_item
        
        # Show annotation info
        total_review = sum(1 for i in self.explorer_window.current_data_items
                           if i.annotation.id not in self.completed_annotation_ids
                           and getattr(i.effective_label, 'short_label_code', '') == REVIEW_LABEL)
        
        annotation_text = "📍 Current Annotation"
        annotation_text += f"<br><small>ID: {item.annotation.id[:12]}... | {total_review} total remaining</small>"
        self.current_annotation_label.setText(annotation_text)
        
        # Show current label
        current_label = getattr(item.effective_label, 'short_label_code', 'Unknown')
        if current_label == REVIEW_LABEL:
            self.current_label_display.setText(f"{current_label} (Needs labeling)")
            self.current_label_display.setStyleSheet("QLabel { color: #d32f2f; font-weight: bold; }")
        else:
            self.current_label_display.setText(current_label)
            self.current_label_display.setStyleSheet("QLabel { color: #388e3c; font-weight: bold; }")
        
        # Show top 3 predictions with confidence bars
        if hasattr(item, 'ml_prediction') and item.ml_prediction:
            pred = item.ml_prediction
            top_predictions = pred['top_predictions'][:3]  # Get top 3
            
            # Update each prediction row
            for i, row in enumerate(self.prediction_rows):
                if i < len(top_predictions):
                    p = top_predictions[i]
                    row['widget'].setVisible(True)
                    row['widget'].set_label_code(p['label'])  # Store label code for clicking
                    
                    # Get label object to access its color
                    label_obj = self._get_label_by_code(p['label'])
                    label_color = label_obj.color.name() if label_obj else "#999999"
                    
                    # Set label text (grey color, not colored by label)
                    row['label'].setText(p['label'])
                    row['label'].setStyleSheet("QLabel { font-weight: bold; font-size: 11pt; color: #666; }")
                    
                    # Color the percentage based on confidence level (matches badge colors)
                    confidence = p['confidence']
                    if confidence >= 0.83:
                        conf_color = "#228B22"  # Dark green (34, 139, 34)
                    elif confidence >= 0.67:
                        conf_color = "#90EE90"  # Light green (144, 238, 144)
                    elif confidence >= 0.50:
                        conf_color = "#FFD700"  # Gold (255, 215, 0)
                    elif confidence >= 0.33:
                        conf_color = "#FFA500"  # Orange (255, 165, 0)
                    elif confidence >= 0.17:
                        conf_color = "#FF6347"  # Tomato (255, 99, 71)
                    else:
                        conf_color = "#DC143C"  # Crimson red (220, 20, 60)
                    
                    row['confidence_label'].setText(f"{confidence:.1%}")
                    row['confidence_label'].setStyleSheet(f"QLabel {{ color: {conf_color}; font-weight: bold; }}")
                    
                    # Use label color for the confidence bar
                    row['confidence_bar'].setValue(int(confidence * 100))
                    row['confidence_bar'].setStyleSheet(f"QProgressBar::chunk {{ background-color: {label_color}; }}")
                else:
                    # Hide unused rows
                    row['widget'].setVisible(False)
        else:
            # No prediction available - hide all rows
            for row in self.prediction_rows:
                row['widget'].setVisible(False)
        
        # Select and highlight in viewers
        self.explorer_window.embedding_viewer.render_selection_from_ids([item.annotation.id])
        
        # Isolate the current annotation in the annotation viewer
        self.explorer_window.annotation_viewer.isolate_and_select_from_ids([item.annotation.id])
        
        # Highlight in viewers - call animate on the item's graphics representations
        if hasattr(item, 'graphics_item') and item.graphics_item:
            item.graphics_item.animate()
        if hasattr(item, 'widget') and item.widget:
            item.widget.animate()
        
        self._enable_annotation_buttons(True)
    
    def _enable_annotation_buttons(self, enabled):
        """Enable/disable annotation action buttons."""
        self.skip_button.setEnabled(enabled)
    
    def _apply_label_from_prediction(self, label_code):
        """Apply a label when a prediction row is clicked."""
        if not self.current_annotation_item:
            return
        
        item = self.current_annotation_item
        label_obj = self._get_label_by_code(label_code)
        
        if label_obj:
            # Update machine confidence from prediction if available
            if hasattr(item, 'ml_prediction') and item.ml_prediction:
                if 'probabilities' in item.ml_prediction:
                    item.annotation.update_machine_confidence(
                        item.ml_prediction['probabilities'],
                        from_import=True
                    )
            
            # Apply label directly and permanently
            item.annotation.label = label_obj
            item._effective_label = label_obj
            
            # Track as completed
            self.completed_annotation_ids.add(item.annotation.id)
            
            # Remove from predictions
            if item.annotation.id in self.bulk_predictions:
                del self.bulk_predictions[item.annotation.id]
            
            # Emit update signal
            self.annotations_updated.emit([item])
            
            # Update progress
            self.labeled_count += 1
            self._update_progress_display()
            
            # Move to next annotation
            self._move_to_next_annotation()
    
    def _on_label_manually_selected(self, label_widget):
        """Handle when user manually selects a label from LabelWindow."""
        # Ignore if we don't have a current annotation or label was deselected
        if not self.current_annotation_item or not label_widget:
            return
        
        # Ignore if wizard is not in annotation mode
        if self.current_page != 2:  # Not on annotation page
            return
        
        item = self.current_annotation_item
        
        # Only update if the label actually changed
        if item.annotation.label.id == label_widget.id:
            return
        
        # Apply the label using the proper annotation API (same as bulk labeling)
        item.annotation.update_label(label_widget)
        item.annotation.update_user_confidence(label_widget)
        
        # Update the data item's effective_label cache
        item._effective_label = label_widget
        
        # Update tooltips
        if hasattr(item, 'widget') and item.widget:
            item.widget.update_tooltip()
        if hasattr(item, 'graphics_item') and item.graphics_item:
            item.graphics_item.update_tooltip()
        
        # Refresh UI to show the change immediately
        self._show_current_annotation()
        
        # Move to next annotation automatically
        self._move_to_next_annotation()
    
    def _cleanup_on_completion(self, auto_close=False):
        """Clean up UI state when annotation process is complete.
        
        Args:
            auto_close (bool): If True, automatically close the wizard after showing success message
        """
        
        # Restore original selection handler if we replaced it
        if hasattr(self, '_original_selection_handler') and hasattr(self, '_bulk_click_handler_connected'):
            if self._bulk_click_handler_connected:
                self.explorer_window.annotation_viewer.handle_annotation_selection = self._original_selection_handler
                self._bulk_click_handler_connected = False
                delattr(self, '_original_selection_handler')
        
        # Disconnect signal to prevent issues during cleanup
        try:
            self.main_window.label_window.labelSelected.disconnect(self._on_label_manually_selected)
        except (TypeError, RuntimeError):
            pass  # Signal was already disconnected or never connected
        
        # Clear all selections in both viewers
        if hasattr(self.explorer_window.embedding_viewer, 'graphics_scene'):
            self.explorer_window.embedding_viewer.graphics_scene.clearSelection()
        
        if hasattr(self.explorer_window.annotation_viewer, 'selected_widgets'):
            for widget in list(self.explorer_window.annotation_viewer.selected_widgets):
                widget.data_item.set_selected(False)
                widget.update_selection_visuals()
            self.explorer_window.annotation_viewer.selected_widgets.clear()
        
        # Reset annotation viewer sort to None and unlock it
        if hasattr(self.explorer_window.annotation_viewer, 'sort_combo'):
            self.explorer_window.annotation_viewer.sort_combo.setEnabled(True)
            self.explorer_window.annotation_viewer.sort_combo.setCurrentText("None")
        
        # Update viewer to show all annotations in unsorted order
        self.explorer_window.annotation_viewer.update_annotations(self.explorer_window.current_data_items)
        
        # Clear all annotation info displays
        self.current_label_display.setText("---")
        self.current_label_display.setStyleSheet("")
        
        # Clear prediction rows
        for row in self.prediction_rows:
            row['widget'].setVisible(False)
            row['label'].setText("-")
            row['confidence_label'].setText("-")
            row['confidence_bar'].setValue(0)
        
        # Disable annotation action buttons
        self._enable_annotation_buttons(False)
        
        # Hide retrain button
        self.retrain_now_button.setVisible(False)
        
        # Disable mode switching tabs
        if hasattr(self, 'annotation_tabs'):
            self.annotation_tabs.setEnabled(False)
        
        # If auto-close is requested, show completion message and close
        if auto_close:
            
            QMessageBox.information(
                self,
                "Annotation Complete",
                "All 'Review'' annotations have been labeled!\n\n"
                "No more 'Review' annotations require attention.\n"
                "The wizard will now close.",
                QMessageBox.Ok
            )
            
            # Emit signal with all completed annotations
            updated_items = [
                item for item in self.explorer_window.current_data_items
                if item.annotation.id in self.completed_annotation_ids
            ]
            self.annotations_updated.emit(updated_items)
            
            # Reset wizard to initial state and close
            self._reset_wizard()
            self.close()
    
    def _skip_annotation(self):
        """Skip the current annotation without labeling it (leaves as 'Review')."""
        if not self.current_annotation_item:
            return
                    
        self._move_to_next_annotation()
    
    def _move_to_next_annotation(self):
        """Move to next annotation."""
        # Get next uncertain annotation
        self._get_next_uncertain_annotation()
        
        # Update UI
        self._update_progress_display()
        self._show_current_annotation()
    
    def _retrain_now(self):
        """Manually trigger model retraining."""
        # Navigate back to training page
        self.current_page = 1
        self.page_stack.setCurrentIndex(1)
        self._update_navigation_buttons()
        
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            result = self.explorer_window.train_annotation_model(
                feature_type=self.feature_type,
                model_type=self.model_type,
                model_params=self.model_params[self.model_type]
            )
            
            if result:
                self.trained_model = result['model']
                self.scaler = result['scaler']
                self.training_score = result['accuracy']
                
                # Update class mappings in case new classes were added
                self.label_classes = result['classes']
                self.class_to_idx = result['class_to_idx']
                self.idx_to_class = result['idx_to_class']
                
                # Display training results
                self._display_training_metrics(result)
                
                # Update ALL annotations with sklearn predictions for confidence display
                self.explorer_window.update_all_sklearn_predictions(
                    self.trained_model,
                    self.scaler,
                    self.class_to_idx,
                    self.feature_type
                )
                
                # Regenerate predictions for remaining Review annotations
                self._generate_all_predictions()
                
                QMessageBox.information(
                    self,
                    "Retrain Complete",
                    f"Model retrained successfully!\nNew accuracy: {self.training_score:.2%}\n\n"
                    "Review the metrics and click 'Start Annotating >' to continue."
                )
                
        except Exception as e:
            QMessageBox.warning(self, "Retrain Error", f"Failed to retrain: {str(e)}")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
    
    def _get_label_by_code(self, label_code):
        """Get Label object from code."""
        for label in self.main_window.label_window.labels:
            if label.short_label_code == label_code:
                return label
        return None
    
    def _exit(self):
        """Exit wizard early (before completion)."""
        # Count completed annotations
        completed_count = len(self.completed_annotation_ids)
        
        reply = QMessageBox.question(
            self,
            "Exit Wizard",
            f"Exit the Auto-Annotation Wizard?\n\n{completed_count} annotations were labeled in this session."
            f"\nAll changes have been saved.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Emit signal with all completed annotations from this session
            updated_items = [
                item for item in self.explorer_window.current_data_items
                if item.annotation.id in self.completed_annotation_ids
            ]
            self.annotations_updated.emit(updated_items)
            
            # Reset wizard state for next use
            self._reset_wizard()
            
            # Close wizard
            self.close()
    
    def _reset_wizard(self):
        """Nuclear option: Destroy this wizard instance and tell ExplorerWindow to recreate it."""
        
        # Restore original selection handler if we replaced it
        if hasattr(self, '_original_selection_handler') and hasattr(self, '_bulk_click_handler_connected'):
            if self._bulk_click_handler_connected:
                self.explorer_window.annotation_viewer.handle_annotation_selection = self._original_selection_handler
                self._bulk_click_handler_connected = False
                delattr(self, '_original_selection_handler')
        
        # Clear all preview labels from data items
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        # Clear any animations on data items
        for item in self.explorer_window.current_data_items:
            # Stop animations on graphics representations
            if hasattr(item, 'graphics_item') and item.graphics_item:
                if hasattr(item.graphics_item, 'deanimate'):
                    item.graphics_item.deanimate()
            if hasattr(item, 'widget') and item.widget:
                if hasattr(item.widget, 'deanimate'):
                    item.widget.deanimate()
        
        # Restore normal selection behavior
        self.explorer_window.embedding_viewer.selection_blocked = False
        self.explorer_window.annotation_viewer.selection_blocked = False
        
        # Clear selections in both viewers
        if hasattr(self.explorer_window.embedding_viewer, 'graphics_scene'):
            self.explorer_window.embedding_viewer.graphics_scene.clearSelection()
        
        if hasattr(self.explorer_window.annotation_viewer, 'selected_widgets'):
            for widget in list(self.explorer_window.annotation_viewer.selected_widgets):
                widget.data_item.set_selected(False)
                widget.update_selection_visuals()
            self.explorer_window.annotation_viewer.selected_widgets.clear()
        
        # Re-enable and reset sort combo
        self.explorer_window.annotation_viewer.sort_combo.setEnabled(True)
        self.explorer_window.annotation_viewer.sort_combo.setCurrentText("None")
        
        # Clear any isolated/ordered views
        if hasattr(self.explorer_window.annotation_viewer, 'isolated_mode'):
            if self.explorer_window.annotation_viewer.isolated_mode:
                self.explorer_window.annotation_viewer.show_all_annotations()
        if hasattr(self.explorer_window.annotation_viewer, 'active_ordered_ids'):
            self.explorer_window.annotation_viewer.active_ordered_ids = []
        
        # Update viewer to show all annotations
        self.explorer_window.annotation_viewer.update_annotations(self.explorer_window.current_data_items)
        
        # Stop any timers
        self.bulk_preview_timer.stop()
        
        # Disconnect all signals to prevent memory leaks
        try:
            self.main_window.label_window.labelSelected.disconnect(self._on_label_manually_selected)
        except TypeError:
            pass  # Already disconnected
        
        # Tell the explorer window to destroy this wizard instance
        self.explorer_window._destroy_and_recreate_wizard()
    
    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        self._update_dataset_info()
    
    def closeEvent(self, event):
        """Handle close event."""
        # Disconnect signal handler to prevent issues
        try:
            self.main_window.label_window.labelSelected.disconnect(self._on_label_manually_selected)
        except (TypeError, RuntimeError):
            pass  # Signal was already disconnected or never connected
        
        # Restore normal selection behavior
        self.explorer_window.embedding_viewer.selection_blocked = False
        self.explorer_window.annotation_viewer.selection_blocked = False
        
        # Re-enable sort combo
        self.explorer_window.annotation_viewer.sort_combo.setEnabled(True)
        
        # Stop any timers
        self.bulk_preview_timer.stop()
        
        # Clear any animations
        for item in self.explorer_window.current_data_items:
            # Stop animations on graphics representations
            if hasattr(item, 'graphics_item') and item.graphics_item:
                if hasattr(item.graphics_item, 'deanimate'):
                    item.graphics_item.deanimate()
            if hasattr(item, 'widget') and item.widget:
                if hasattr(item.widget, 'deanimate'):
                    item.widget.deanimate()
        
        # Reset viewers and restore settings
        if hasattr(self.explorer_window, 'annotation_viewer'):
            self.explorer_window.annotation_viewer.reset_view_requested.emit()
        
        event.accept()
