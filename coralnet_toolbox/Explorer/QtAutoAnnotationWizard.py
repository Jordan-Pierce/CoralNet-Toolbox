import warnings
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QHBoxLayout,
                             QLabel, QListWidget, QPushButton, QRadioButton,
                             QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
                             QDialog, QGroupBox, QComboBox, QDoubleSpinBox,
                             QTextEdit, QProgressBar, QFormLayout, QGridLayout,
                             QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)

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


REVIEW_LABEL = 'Review'


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
        self.retrain_interval = 10
        self.labels_since_retrain = 0
        self.current_batch = []
        self.current_batch_index = 0
        
        # Model parameters
        self.model_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'svc': {'C': 1.0, 'kernel': 'rbf', 'probability': True},
            'knn': {'n_neighbors': 5}
        }
        
        # Thresholds
        self.auto_label_threshold = 0.95
        self.uncertainty_threshold = 0.60
        self.batch_size = 20
        
        self.setup_ui()
        
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
        
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        self.cancel_button.clicked.connect(self.close)
        
        nav_layout.addWidget(self.back_button)
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
        
        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        self.dataset_info_label = QLabel()
        info_layout.addWidget(self.dataset_info_label)
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
        
        # Training parameters
        param_group = QGroupBox("Training Parameters")
        param_layout = QFormLayout(param_group)
        
        self.auto_label_spin = QDoubleSpinBox()
        self.auto_label_spin.setRange(0.5, 1.0)
        self.auto_label_spin.setSingleStep(0.05)
        self.auto_label_spin.setValue(0.95)
        self.auto_label_spin.setToolTip("Automatically apply labels when confidence exceeds this threshold")
        param_layout.addRow("Auto-label threshold:", self.auto_label_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(5, 100)
        self.batch_size_spin.setValue(20)
        self.batch_size_spin.setToolTip("Number of annotations to suggest in each batch")
        param_layout.addRow("Batch size:", self.batch_size_spin)
        
        self.retrain_interval_spin = QSpinBox()
        self.retrain_interval_spin.setRange(1, 50)
        self.retrain_interval_spin.setValue(10)
        self.retrain_interval_spin.setToolTip("Retrain model after this many labels")
        param_layout.addRow("Retrain interval:", self.retrain_interval_spin)
        
        layout.addWidget(param_group)
        layout.addStretch()
        
        return page
    
    def _create_training_page(self):
        """Page 2: Train model and show metrics."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Training: Model Performance</h2>")
        layout.addWidget(title)
        
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
        """Page 3: Iterative annotation with predictions."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("<h2>Annotation: Intelligent Labeling</h2>")
        layout.addWidget(title)
        
        # Progress info
        progress_group = QGroupBox("Progress")
        progress_layout = QFormLayout(progress_group)
        
        self.progress_label = QLabel("0 / 0")
        self.auto_label_progress_label = QLabel("0")
        self.remaining_review_label = QLabel("0")
        self.model_accuracy_label = QLabel("N/A")
        
        progress_layout.addRow("Manually labeled:", self.progress_label)
        progress_layout.addRow("Auto-labeled:", self.auto_label_progress_label)
        progress_layout.addRow("Remaining 'Review':", self.remaining_review_label)
        progress_layout.addRow("Est. accuracy:", self.model_accuracy_label)
        
        layout.addWidget(progress_group)
        
        # Current annotation info
        current_group = QGroupBox("Current Annotation")
        current_layout = QVBoxLayout(current_group)
        
        self.current_annotation_label = QLabel("No annotation selected")
        current_layout.addWidget(self.current_annotation_label)
        
        # Prediction display
        pred_layout = QFormLayout()
        self.predicted_label_label = QLabel("-")
        self.confidence_label = QLabel("-")
        self.confidence_bar = QProgressBar()
        self.top_predictions_label = QLabel("-")
        
        pred_layout.addRow("Predicted Label:", self.predicted_label_label)
        pred_layout.addRow("Confidence:", self.confidence_label)
        pred_layout.addRow("", self.confidence_bar)
        pred_layout.addRow("Top 3 Alternatives:", self.top_predictions_label)
        
        current_layout.addLayout(pred_layout)
        layout.addWidget(current_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        button_row1 = QHBoxLayout()
        self.accept_button = QPushButton("Accept Prediction")
        self.reject_button = QPushButton("Reject / Manual Label")
        self.skip_button = QPushButton("Skip")
        
        self.accept_button.clicked.connect(self._accept_prediction)
        self.reject_button.clicked.connect(self._reject_prediction)
        self.skip_button.clicked.connect(self._skip_annotation)
        
        button_row1.addWidget(self.accept_button)
        button_row1.addWidget(self.reject_button)
        button_row1.addWidget(self.skip_button)
        action_layout.addLayout(button_row1)
        
        button_row2 = QHBoxLayout()
        self.retrain_now_button = QPushButton("Retrain Now")
        self.auto_label_all_button = QPushButton("Auto-Label All Confident")
        self.validate_button = QPushButton("Validate Training Labels")
        
        self.retrain_now_button.clicked.connect(self._retrain_now)
        self.auto_label_all_button.clicked.connect(self._auto_label_all)
        self.validate_button.clicked.connect(self._validate_labels)
        
        button_row2.addWidget(self.retrain_now_button)
        button_row2.addWidget(self.auto_label_all_button)
        button_row2.addWidget(self.validate_button)
        action_layout.addLayout(button_row2)
        
        layout.addWidget(action_group)
        layout.addStretch()
        
        return page
    
    def _on_model_changed(self, index):
        """Update parameter widget when model selection changes."""
        self.param_stack.setCurrentIndex(index)
    
    def _update_navigation_buttons(self):
        """Update button states based on current page."""
        self.back_button.setEnabled(self.current_page > 0)
        
        if self.current_page == 0:
            self.next_button.setText("Next >")
            self.next_button.setEnabled(True)
        elif self.current_page == 1:
            self.next_button.setText("Start Annotating >")
            self.next_button.setEnabled(self.trained_model is not None)
        elif self.current_page == 2:
            self.next_button.setText("Finish")
            self.next_button.setEnabled(True)
    
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
            # Finish
            self._finish()
            
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
        
        # Get training parameters
        self.auto_label_threshold = self.auto_label_spin.value()
        self.batch_size = self.batch_size_spin.value()
        self.retrain_interval = self.retrain_interval_spin.value()
        
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
        
        info_text = (
            f"Total annotations: {total}\n"
            f"Labeled: {labeled_count}\n"
            f"Review (unlabeled): {review_count}\n"
            f"Unique classes: {len(unique_labels)}"
        )
        
        if len(unique_labels) < 2:
            info_text += "\n\n⚠️ Warning: Need at least 2 classes to train model"
        elif labeled_count < 10:
            info_text += f"\n\n⚠️ Warning: Only {labeled_count} labeled examples (cold start)"
        
        self.dataset_info_label.setText(info_text)
    
    def _train_model(self):
        """Train the ML model."""
        self.training_status_label.setText("Training model...")
        self.training_progress.setVisible(True)
        self.training_progress.setRange(0, 0)  # Indeterminate
        self.train_button.setEnabled(False)
        
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
        self._update_progress_display()
        self._get_next_batch()
        self._show_current_annotation()
    
    def _update_progress_display(self):
        """Update progress statistics."""
        data_items = self.explorer_window.current_data_items
        total = len(data_items)
        
        review_count = sum(1 for item in data_items 
                          if getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL)
        
        self.progress_label.setText(f"{self.labeled_count} / {total}")
        self.auto_label_progress_label.setText(str(self.auto_labeled_count))
        self.remaining_review_label.setText(str(review_count))
        self.model_accuracy_label.setText(f"{self.training_score:.1%}")
    
    def _get_next_batch(self):
        """Get next batch of annotations to label."""
        try:
            self.current_batch = self.explorer_window.get_next_annotation_batch(
                model=self.trained_model,
                scaler=self.scaler,
                class_to_idx=self.class_to_idx,
                feature_type=self.feature_type,
                batch_size=self.batch_size
            )
            self.current_batch_index = 0
            
            # Display batch in viewers
            if self.current_batch:
                batch_ids = [item.annotation.id for item in self.current_batch]
                self.explorer_window.annotation_viewer.display_and_isolate_ordered_results(batch_ids)
                self.explorer_window.embedding_viewer.render_selection_from_ids(batch_ids)
                
        except Exception as e:
            QMessageBox.warning(self, "Batch Error", f"Failed to get next batch: {str(e)}")
            self.current_batch = []
    
    def _show_current_annotation(self):
        """Display information about current annotation."""
        if not self.current_batch or self.current_batch_index >= len(self.current_batch):
            self.current_annotation_label.setText("No more annotations in current batch")
            self._enable_annotation_buttons(False)
            return
        
        item = self.current_batch[self.current_batch_index]
        
        # Show annotation info
        annotation_text = f"Annotation {self.current_batch_index + 1} of {len(self.current_batch)}\n"
        annotation_text += f"ID: {item.annotation.id[:8]}..."
        self.current_annotation_label.setText(annotation_text)
        
        # Show prediction
        if hasattr(item, 'ml_prediction') and item.ml_prediction:
            pred = item.ml_prediction
            self.predicted_label_label.setText(pred['label'])
            self.confidence_label.setText(f"{pred['confidence']:.1%}")
            self.confidence_bar.setValue(int(pred['confidence'] * 100))
            
            # Format top predictions
            top_3_text = "<br>".join([
                f"{i+1}. {p['label']}: {p['confidence']:.1%}"
                for i, p in enumerate(pred['top_predictions'][:3])
            ])
            self.top_predictions_label.setText(top_3_text)
        else:
            self.predicted_label_label.setText("N/A")
            self.confidence_label.setText("N/A")
            self.confidence_bar.setValue(0)
            self.top_predictions_label.setText("N/A")
        
        # Highlight in viewers - call animate on the item's graphics representations
        if hasattr(item, 'graphics_item') and item.graphics_item:
            item.graphics_item.animate()
        if hasattr(item, 'widget') and item.widget:
            item.widget.animate()
        
        self._enable_annotation_buttons(True)
    
    def _enable_annotation_buttons(self, enabled):
        """Enable/disable annotation action buttons."""
        self.accept_button.setEnabled(enabled)
        self.reject_button.setEnabled(enabled)
        self.skip_button.setEnabled(enabled)
    
    def _accept_prediction(self):
        """Accept the predicted label."""
        if not self.current_batch or self.current_batch_index >= len(self.current_batch):
            return
        
        item = self.current_batch[self.current_batch_index]
        
        if hasattr(item, 'ml_prediction') and item.ml_prediction:
            # Apply predicted label
            predicted_label = item.ml_prediction['label']
            label_obj = self._get_label_by_code(predicted_label)
            
            if label_obj:
                item.annotation.update_label(label_obj)
                self.labeled_count += 1
                self.labels_since_retrain += 1
        
        self._move_to_next_annotation()
    
    def _reject_prediction(self):
        """Reject prediction - user will manually label via main UI."""
        if not self.current_batch or self.current_batch_index >= len(self.current_batch):
            return
        
        item = self.current_batch[self.current_batch_index]
        
        # Show message
        QMessageBox.information(
            self,
            "Manual Labeling",
            "Use the Label Window in the main interface to assign the correct label.\n\n"
            "The annotation will remain highlighted for your reference."
        )
        
        # Keep annotation highlighted but move to next
        self._move_to_next_annotation()
    
    def _skip_annotation(self):
        """Skip current annotation."""
        self._move_to_next_annotation()
    
    def _move_to_next_annotation(self):
        """Move to next annotation in batch."""
        self.current_batch_index += 1
        
        # Check if we need to retrain
        if self.labels_since_retrain >= self.retrain_interval:
            self._retrain_now()
        
        # Check if batch is done
        if self.current_batch_index >= len(self.current_batch):
            reply = QMessageBox.question(
                self,
                "Batch Complete",
                "Current batch complete. Get next batch?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._get_next_batch()
            else:
                self._enable_annotation_buttons(False)
                return
        
        self._update_progress_display()
        self._show_current_annotation()
    
    def _retrain_now(self):
        """Manually trigger model retraining."""
        self.training_status_label.setText("Retraining model...")
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
                self.labels_since_retrain = 0
                
                QMessageBox.information(
                    self,
                    "Retrain Complete",
                    f"Model retrained successfully!\nNew accuracy: {self.training_score:.2%}"
                )
                
                # Get fresh predictions for current batch
                self._get_next_batch()
                
        except Exception as e:
            QMessageBox.warning(self, "Retrain Error", f"Failed to retrain: {str(e)}")
    
    def _auto_label_all(self):
        """Auto-label all confident predictions."""
        reply = QMessageBox.question(
            self,
            "Auto-Label All",
            f"Apply predicted labels to all annotations with confidence > {self.auto_label_threshold:.0%}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                count = self.explorer_window.auto_label_confident_predictions(
                    model=self.trained_model,
                    scaler=self.scaler,
                    class_to_idx=self.class_to_idx,
                    idx_to_class=self.idx_to_class,
                    feature_type=self.feature_type,
                    threshold=self.auto_label_threshold
                )
                
                self.auto_labeled_count += count
                self._update_progress_display()
                
                QMessageBox.information(
                    self,
                    "Auto-Labeling Complete",
                    f"Successfully auto-labeled {count} annotations."
                )
                
                # Refresh batch
                self._get_next_batch()
                
            except Exception as e:
                QMessageBox.warning(self, "Auto-Label Error", f"Failed: {str(e)}")
    
    def _validate_labels(self):
        """Validate training labels using anomaly detection."""
        try:
            flagged_items = self.explorer_window.validate_training_labels()
            
            if flagged_items:
                QMessageBox.warning(
                    self,
                    "Validation Results",
                    f"Found {len(flagged_items)} potentially mislabeled annotations.\n\n"
                    "These have been highlighted in the viewers for your review."
                )
                
                # Display flagged items
                flagged_ids = [item.annotation.id for item in flagged_items]
                self.explorer_window.annotation_viewer.display_and_isolate_ordered_results(flagged_ids)
            else:
                QMessageBox.information(
                    self,
                    "Validation Results",
                    "No obvious labeling issues detected!"
                )
                
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", f"Failed: {str(e)}")
    
    def _get_label_by_code(self, label_code):
        """Get Label object from code."""
        for label in self.main_window.label_window.labels:
            if label.short_label_code == label_code:
                return label
        return None
    
    def _finish(self):
        """Finish wizard and apply changes."""
        reply = QMessageBox.question(
            self,
            "Finish Wizard",
            "Exit the Auto-Annotation Wizard?\n\nAll labeled annotations will be saved.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Emit signal with updated annotations
            updated_items = [item for item in self.explorer_window.current_data_items 
                            if item.has_preview_changes()]
            self.annotations_updated.emit(updated_items)
            
            # Close wizard
            self.close()
    
    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        self._update_dataset_info()
    
    def closeEvent(self, event):
        """Handle close event."""
        # Clear any animations
        for item in self.explorer_window.current_data_items:

            # Stop animations on graphics representations
            if hasattr(item, 'graphics_item') and item.graphics_item:
                if hasattr(item.graphics_item, 'deanimate'):
                    item.graphics_item.deanimate()
            if hasattr(item, 'widget') and item.widget:
                if hasattr(item.widget, 'deanimate'):
                    item.widget.deanimate()
        
        # Reset viewers
        if hasattr(self.explorer_window, 'annotation_viewer'):
            self.explorer_window.annotation_viewer.reset_view_requested.emit()
        
        event.accept()
