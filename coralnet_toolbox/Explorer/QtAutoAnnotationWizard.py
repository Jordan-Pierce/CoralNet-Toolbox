import warnings

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QHBoxLayout,
                             QLabel, QListWidget, QPushButton, QRadioButton,
                             QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
                             QDialog, QGroupBox, QComboBox, QDoubleSpinBox,
                             QTextEdit, QProgressBar, QFormLayout, QGridLayout,
                             QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QTabWidget, QSlider)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Auto-annotation wizard will not be available.")
    

warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    
# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


REVIEW_LABEL = 'Review'


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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
        self.bulk_preview_timer = QTimer()
        self.bulk_preview_timer.setSingleShot(True)
        self.bulk_preview_timer.timeout.connect(self._apply_bulk_preview_labels)
        
        # Model parameters
        self.model_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'svc': {'C': 1.0, 'kernel': 'rbf', 'probability': True},
            'knn': {'n_neighbors': 5}
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
        info_group = QGroupBox("📊 Dataset Information")
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
        feature_layout = QVBoxLayout(feature_group)
        
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
        
        feature_layout.addWidget(self.full_features_radio)
        feature_layout.addWidget(self.reduced_features_radio)
        layout.addWidget(feature_group)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest (Recommended)",
            "Support Vector Machine",
            "K-Nearest Neighbors"
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
        
        model_layout.addRow(self.param_stack)
        layout.addWidget(model_group)
        
        layout.addStretch()
        
        return page
    
    def _create_training_page(self):
        """Page 2: Train model and show metrics."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Training: Model Performance</h2>")
        layout.addWidget(title)
        
        # Information box
        info_box = QGroupBox("Information")
        info_layout = QVBoxLayout(info_box)
        info_text = QLabel(
            "Train the model on your labeled annotations. The model will learn patterns from your existing labels "
            "to predict labels for unlabeled annotations. Review the accuracy metrics and confusion matrix to assess "
            "model performance before proceeding to annotation."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        
        # Status
        self.training_status_label = QLabel("Press 'Train Model' to begin...")
        layout.addWidget(self.training_status_label)
        
        # Progress
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)
        
        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self._train_model)
        layout.addWidget(self.train_button)
        
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
            "<b>✓ Accept Prediction:</b> Applies the predicted label to the current annotation.<br>"
            "<b>⊘ Skip:</b> Skips the current annotation without labeling (leaves as 'Review').<br>"
            "<b>→ Next:</b> Advances after you manually changed the label in the Label Window."
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
        
        pred_layout = QFormLayout()
        self.predicted_label_label = QLabel("-")
        self.predicted_label_label.setStyleSheet("QLabel { font-weight: bold; color: #1976d2; font-size: 12pt; }")
        self.confidence_label = QLabel("-")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
        self.top_predictions_label = QLabel("-")
        self.top_predictions_label.setWordWrap(True)
        
        pred_layout.addRow("<b>Predicted Label:</b>", self.predicted_label_label)
        pred_layout.addRow("<b>Confidence:</b>", self.confidence_label)
        pred_layout.addRow("", self.confidence_bar)
        pred_layout.addRow("<b>Alternatives:</b>", self.top_predictions_label)
        current_layout.addLayout(pred_layout)
        layout.addWidget(current_group)
        
        # Annotation actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        row1 = QHBoxLayout()
        self.accept_button = QPushButton("✓ Accept Prediction")
        self.accept_button.setToolTip("Apply the predicted label to this annotation")
        self.accept_button.clicked.connect(self._accept_prediction)
        row1.addWidget(self.accept_button)
        actions_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        self.skip_button = QPushButton("⊘ Skip")
        self.skip_button.setToolTip("Skip this annotation without labeling (leaves as 'Review')")
        self.skip_button.clicked.connect(self._skip_annotation)
        self.next_button = QPushButton("→ Next")
        self.next_button.setToolTip("Advance to next annotation after manually labeling in Label Window")
        self.next_button.clicked.connect(self._next_annotation)
        row2.addWidget(self.skip_button)
        row2.addWidget(self.next_button)
        actions_layout.addLayout(row2)
        
        layout.addWidget(actions_group)
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
        self.training_status_label.setText("Model configuration changed. Please train the model again.")
        
        # Disable navigation to annotation page
        if self.current_page == 1:
            self._update_navigation_buttons()
    
    def _on_annotation_mode_changed(self, index):
        """Handle switching between Active Learning and Bulk Labeling modes."""
        # Clear animations and selections when switching modes
        self._clear_animations_and_selections()
        
        # Rescan for any manually labeled annotations before entering new mode
        manually_labeled_count = self._rescan_for_manual_labels()
        if manually_labeled_count > 0:
            print(f"✓ Detected {manually_labeled_count} manually labeled annotation(s) before mode switch")
        
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
        
        Returns:
            int: Number of newly detected manually labeled annotations
        """
        newly_labeled_count = 0
        
        for item in self.explorer_window.current_data_items:
            # Check if this annotation is not in completed set but is also not 'Review'
            if item.annotation.id not in self.completed_annotation_ids:
                current_label = getattr(item.annotation.label, 'short_label_code', '')
                
                # If it's not Review, it was manually labeled
                if current_label and current_label != REVIEW_LABEL:
                    # Mark as completed
                    self.completed_annotation_ids.add(item.annotation.id)
                    newly_labeled_count += 1
                    
                    # Update the data item's effective_label cache
                    item._effective_label = item.annotation.label
                    
                    # Remove from bulk predictions if it was there
                    if item.annotation.id in self.bulk_predictions:
                        del self.bulk_predictions[item.annotation.id]
                    
                    # Update tooltip and visuals
                    if hasattr(item, 'widget') and item.widget:
                        item.widget.update_tooltip()
                    if hasattr(item, 'graphics_item') and item.graphics_item:
                        item.graphics_item.update_tooltip()
                    
                    print(f"  Detected manual label: {current_label} for annotation {item.annotation.id[:12]}...")
        
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
            self.next_button.setText("Next >")
            self.next_button.setEnabled(True)
            self.retrain_now_button.setVisible(False)
        elif self.current_page == 1:
            self.next_button.setText("Start Annotating >")
            self.next_button.setEnabled(self.trained_model is not None)
            self.retrain_now_button.setVisible(False)
        elif self.current_page == 2:
            self.next_button.setText("Exit")
            self.next_button.setEnabled(True)
            self.retrain_now_button.setVisible(True)
    
    def _go_back(self):
        """Navigate to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
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
        model_types = ['random_forest', 'svc', 'knn']
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
        
        # Update dataset info
        self._update_dataset_info()
    
    def _update_dataset_info(self):
        """Update dataset information display."""
        data_items = self.explorer_window.current_data_items
        total = len(data_items)
        
        review_count = sum(1 for item in data_items 
                          if getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL)
        
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
        self.training_status_label.setText("Training model...")
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
            
            self.training_status_label.setText(
                f"✓ Model trained successfully! Accuracy: {self.training_score:.2%}"
            )
            self.next_button.setEnabled(True)
            
        except Exception as e:
            self.training_status_label.setText(f"❌ Training failed: {str(e)}")
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
            print("Warning: Model not trained yet")
            return
        
        print("\n=== Entering Active Learning Mode ===")
        
        # Rescan for manually labeled annotations before entering mode
        manually_labeled_count = self._rescan_for_manual_labels()
        if manually_labeled_count > 0:
            print(f"Found {manually_labeled_count} manually labeled annotations")
        
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
            print(f"Generating predictions for {len(review_items)} review items")
            self._generate_predictions_for_items(review_items)
        else:
            print("No review items found - all annotations labeled")
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
        print("=== Active Learning Mode Ready ===")
    
    def _enter_bulk_labeling_mode(self):
        """Enter Bulk Labeling mode."""
        print("\n=== Entering Bulk Labeling Mode ===")
        
        # First, rescan for manually labeled annotations
        manually_labeled_count = self._rescan_for_manual_labels()
        if manually_labeled_count > 0:
            print(f"Found {manually_labeled_count} manually labeled annotations")
        
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
        
        print(f"Found {len(review_items)} review items (excluded {len(self.completed_annotation_ids)} completed)")
        
        # Clear any stale predictions for items that are no longer Review
        review_ids = {item.annotation.id for item in review_items}
        stale_ids = [ann_id for ann_id in self.bulk_predictions.keys() if ann_id not in review_ids]
        if stale_ids:
            print(f"Clearing {len(stale_ids)} stale predictions")
            for ann_id in stale_ids:
                del self.bulk_predictions[ann_id]
        
        # Regenerate predictions for all review items to ensure they're current
        if review_items and self.trained_model is not None:
            print(f"Generating/updating predictions for {len(review_items)} review items")
            self._generate_predictions_for_items(review_items)
        elif not review_items:
            print("No review items found - all annotations labeled")
            self.current_annotation_item = None
        
        print(f"Have {len(self.bulk_predictions)} total predictions")
        
        # Exit isolation mode if active
        if self.explorer_window.annotation_viewer.isolated_mode:
            self.explorer_window.annotation_viewer.show_all_annotations()
        
        # Set annotation viewer to sort by label and lock it
        self.explorer_window.annotation_viewer.sort_combo.setCurrentText("Label")
        self.explorer_window.annotation_viewer.sort_combo.setEnabled(False)
        
        # Update the viewer with only Review data items to show proper label bins
        self.explorer_window.annotation_viewer.update_annotations(review_items)
        
        # Update statistics (will recount from scratch)
        self._update_bulk_statistics()
        
        # Apply threshold to show preview labels
        self._apply_bulk_preview_labels()
        print("=== Bulk Labeling Mode Ready ===")
    
    def _generate_predictions_for_items(self, items):
        """Generate predictions for specific items."""
        if not items:
            return
        
        try:
            # predict_with_model returns None but stores predictions in item.ml_prediction
            self.explorer_window.predict_with_model(
                items,
                self.trained_model,
                self.scaler,
                self.class_to_idx,
                self.idx_to_class,
                self.feature_type
            )
            
            # Collect the predictions that were stored in the items
            for item in items:
                if hasattr(item, 'ml_prediction') and item.ml_prediction:
                    self.bulk_predictions[item.annotation.id] = item.ml_prediction
                
        except Exception as e:
            print(f"Failed to generate predictions: {str(e)}")
    
    def _generate_all_predictions(self):
        """Generate predictions for all Review annotations, excluding completed ones."""
        # Validate model and scaler
        if self.trained_model is None or self.scaler is None:
            print("Warning: Model or scaler not initialized. Skipping prediction generation.")
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
            print("No review items found")
            return
        
        print(f"Generating predictions for {len(review_items)} review items...")
        self._generate_predictions_for_items(review_items)
        print(f"Successfully generated {len(self.bulk_predictions)} predictions")
    
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
        
        # Update label
        self.bulk_threshold_label.setText(f"<b>Current Threshold: {value}%</b>")
        
        # Debounce: restart timer
        self.bulk_preview_timer.stop()
        self.bulk_preview_timer.start(400)  # 400ms debounce
    
    def _apply_bulk_preview_labels(self):
        """Apply preview labels based on current confidence threshold."""
        threshold = self.bulk_confidence_threshold
        count = 0
        above_threshold = 0
        
        # Clear all previews first
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        # Apply previews for annotations above threshold
        for item in self.explorer_window.current_data_items:
            ann_id = item.annotation.id
            
            # Only process Review annotations with predictions
            if getattr(item.effective_label, 'short_label_code', '') != REVIEW_LABEL:
                continue
            
            if ann_id not in self.bulk_predictions:
                continue
            
            pred = self.bulk_predictions[ann_id]
            if pred['confidence'] >= threshold:
                above_threshold += 1
                label_obj = self._get_label_by_code(pred['label'])
                if label_obj:
                    item.set_preview_label(label_obj)
                    count += 1
        
        print(f"Bulk Labeling: Threshold {threshold:.2f}, {above_threshold} above threshold, {count} labels applied")
        
        # Update count display
        self.bulk_count_label.setText(f"Annotations to be auto-labeled: {count}")
        
        # Refresh annotation viewer to show preview changes
        self.explorer_window.annotation_viewer.recalculate_layout()
    
    def _clear_bulk_previews(self):
        """Clear all bulk preview labels."""
        for item in self.explorer_window.current_data_items:
            if item.has_preview_changes():
                item.clear_preview_label()
        
        self.bulk_count_label.setText("Annotations to be auto-labeled: 0")
        self.explorer_window.annotation_viewer.recalculate_layout()
    
    def _apply_bulk_labels_permanently(self):
        """Apply all bulk preview labels permanently."""
        # Count preview changes
        preview_count = sum(1 for item in self.explorer_window.current_data_items
                           if item.has_preview_changes())
        
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
            
            QMessageBox.information(
                self,
                "Success",
                f"Successfully applied {preview_count} labels.\n\n{len(review_items)} Review annotations remaining."
            )
    
    def _get_next_uncertain_annotation(self):
        """Get the next most uncertain annotation to review."""
        # Validate model and scaler
        if self.trained_model is None or self.scaler is None:
            print("Warning: Model or scaler not initialized")
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
                print("⚠ No more uncertain annotations available")
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
                print(f"No predictions found, using first review item")
            
            self.current_annotation_item = most_uncertain
            print(f"✓ Selected annotation with {lowest_confidence:.1%} confidence")
            
            # Highlight in embedding viewer
            self.explorer_window.embedding_viewer.render_selection_from_ids([most_uncertain.annotation.id])
            
        except Exception as e:
            print(f"Error getting uncertain annotation: {str(e)}")
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
        
        annotation_text = f"📍 Current Annotation"
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
        
        # Show prediction
        if hasattr(item, 'ml_prediction') and item.ml_prediction:
            pred = item.ml_prediction
            self.predicted_label_label.setText(pred['label'])
            self.confidence_label.setText(f"{pred['confidence']:.1%}")
            self.confidence_bar.setValue(int(pred['confidence'] * 100))
            
            # Color code confidence bar
            if pred['confidence'] >= 0.9:
                self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            elif pred['confidence'] >= 0.7:
                self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
            else:
                self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            
            # Format top predictions
            top_3_text = "<br>".join([
                f"<b>{i+1}.</b> {p['label']}: {p['confidence']:.1%}"
                for i, p in enumerate(pred['top_predictions'][:3])
            ])
            self.top_predictions_label.setText(top_3_text)
        else:
            self.predicted_label_label.setText("N/A")
            self.confidence_label.setText("N/A")
            self.confidence_bar.setValue(0)
            self.top_predictions_label.setText("N/A")
        
        # Select and highlight in viewers
        self.explorer_window.embedding_viewer.render_selection_from_ids([item.annotation.id])
        
        # Highlight in viewers - call animate on the item's graphics representations
        if hasattr(item, 'graphics_item') and item.graphics_item:
            item.graphics_item.animate()
        if hasattr(item, 'widget') and item.widget:
            item.widget.animate()
        
        self._enable_annotation_buttons(True)
    
    def _enable_annotation_buttons(self, enabled):
        """Enable/disable annotation action buttons."""
        self.accept_button.setEnabled(enabled)
        self.skip_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
    
    def _accept_prediction(self):
        """Accept the predicted label permanently."""
        if not self.current_annotation_item:
            return
        
        item = self.current_annotation_item
        
        if hasattr(item, 'ml_prediction') and item.ml_prediction:
            # Apply predicted label
            predicted_label = item.ml_prediction['label']
            label_obj = self._get_label_by_code(predicted_label)
            
            if label_obj:
                # Update machine confidence from prediction first (before applying label)
                if 'probabilities' in item.ml_prediction:
                    item.annotation.update_machine_confidence(
                        item.ml_prediction['probabilities'],
                        from_import=True  # Don't overwrite the label we're about to apply
                    )
                
                # Apply label directly and permanently (no preview)
                item.annotation.label = label_obj
                
                # Update the data item's effective_label cache
                item._effective_label = label_obj
                
                # Track this annotation as completed
                self.completed_annotation_ids.add(item.annotation.id)
                
                # Remove from predictions since it's now labeled
                if item.annotation.id in self.bulk_predictions:
                    del self.bulk_predictions[item.annotation.id]
                
                # Update tooltip and visuals
                if hasattr(item, 'widget') and item.widget:
                    item.widget.update_tooltip()
                if hasattr(item, 'graphics_item') and item.graphics_item:
                    item.graphics_item.update_tooltip()
    
        # Move to next BEFORE refreshing viewers to avoid blocking
        self._move_to_next_annotation()
    
    def _on_label_manually_selected(self, label_widget):
        """Handle when user manually selects a label from LabelWindow."""
        if not self.current_annotation_item:
            return
        
        if not label_widget:  # Label was deselected
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
        
        print(f"✓ Manual label applied: '{label_widget.short_label_code}' for annotation {item.annotation.id[:12]}...")
    
    def _cleanup_on_completion(self, auto_close=False):
        """Clean up UI state when annotation process is complete.
        
        Args:
            auto_close (bool): If True, automatically close the wizard after showing success message
        """
        print("\n=== Cleaning up on completion ===")
        
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
        self.predicted_label_label.setText("---")
        self.confidence_label.setText("---")
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("")
        self.top_predictions_label.setText("---")
        
        # Disable annotation action buttons
        self._enable_annotation_buttons(False)
        
        # Hide retrain button
        self.retrain_now_button.setVisible(False)
        
        # Disable mode switching tabs
        if hasattr(self, 'annotation_tabs'):
            self.annotation_tabs.setEnabled(False)
        
        print("Cleanup complete - ready to finish")
        
        # If auto-close is requested, show completion message and close
        if auto_close:
            # Count how many were labeled
            completed_count = len(self.completed_annotation_ids)
            total_review = sum(1 for i in self.explorer_window.current_data_items
                             if getattr(i.effective_label, 'short_label_code', '') == REVIEW_LABEL)
            
            QMessageBox.information(
                self,
                "Annotation Complete! 🎉",
                f"✓ All uncertain annotations have been labeled!\n\n"
                f"Labeled in this session: {completed_count}\n"
                f"Remaining 'Review' annotations: {total_review}\n\n"
                f"No more uncertain annotations require attention.\n"
                f"The wizard will now close.",
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
            
        item = self.current_annotation_item
        print(f"⊘ Skipped annotation {item.annotation.id[:12]}... (no label applied)")
        
        self._move_to_next_annotation()
    
    def _next_annotation(self):
        """Advance to next annotation after manually labeling the current one.
        
        Since labels are now applied immediately via _on_label_manually_selected(),
        this just verifies the label was changed and moves to the next annotation.
        """
        if not self.current_annotation_item:
            return
            
        item = self.current_annotation_item
        current_label = getattr(item.annotation.label, 'short_label_code', '')
        
        # Check if the annotation was manually labeled (changed from Review)
        if current_label and current_label != REVIEW_LABEL:
            # The label was already applied via _on_label_manually_selected()
            # Just track it as completed and move on
            self.completed_annotation_ids.add(item.annotation.id)
            
            # Remove from predictions
            if item.annotation.id in self.bulk_predictions:
                del self.bulk_predictions[item.annotation.id]
            
            print(f"  → Confirmed and moving to next (total completed: {len(self.completed_annotation_ids)})")
        else:
            # Warning: Next button clicked but label is still 'Review'
            QMessageBox.warning(
                self,
                "No Label Change Detected",
                f"The annotation label is still '{current_label}'.\n\n"
                "Please select a label in the Label Window before clicking 'Next', "
                "or click 'Skip' to move on without labeling.",
                QMessageBox.Ok
            )
            return  # Don't move to next annotation
        
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
        
        self.training_status_label.setText("Retraining model...")
        
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
                
                # Regenerate predictions for remaining Review annotations
                self._generate_all_predictions()
                
                self.training_status_label.setText(
                    f"✓ Model retrained successfully! Accuracy: {self.training_score:.2%}\n\n"
                    "Review the updated metrics below, then click 'Start Annotating >' to continue."
                )
                
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
        print("\n=== NUCLEAR RESET: Destroying and recreating wizard ===")
        
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
        
        print("Wizard destruction complete - new instance will be created on next open")
    
    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        self._update_dataset_info()
    
    def closeEvent(self, event):
        """Handle close event."""
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
