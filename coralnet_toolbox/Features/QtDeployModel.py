"""
Features deployment dialog for configuring and loading feature extraction models.

Allows users to select a transformer, YOLO, or color feature model, configure
extraction parameters, and manage the loaded model.
"""

import warnings
import os
import gc

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon

from coralnet_toolbox.Features import TRANSFORMER_MODELS, TIMM_MODELS, OPENCLIP_MODELS
from coralnet_toolbox.Features.Extractor import FeatureExtractor, model_supports_dense

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)


class FeaturesDeployModelDialog(QDialog):
    """
    Feature extraction deployment dialog.

    Modal dialog for configuring and loading feature extraction models.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the Features deployment dialog.

        Args:
            main_window: Reference to the main window.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        # Model management
        self.loaded_model: FeatureExtractor = None
        self.cuda_warning_shown = False

        # Setup UI
        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Feature Extractor")
        self.resize(200, 300)

        layout = QVBoxLayout()

        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()

        # Only dense-capable backbones are offered here: feature maps (and the
        # 3D Features tool) need a dense [h, w, C] map. YOLO and Color models are
        # pooled-only (image-level embeddings for the Explorer) and would produce
        # no feature map, so they are intentionally excluded.
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for registry in (TRANSFORMER_MODELS, TIMM_MODELS, OPENCLIP_MODELS):
            for display_name, model_id in registry.items():
                if model_supports_dense(model_id):
                    self.model_combo.addItem(display_name, model_id)

        default_idx = self.model_combo.findText("DINOv2+reg (Small)")
        self.model_combo.setCurrentIndex(default_idx if default_idx >= 0 else 0)
        self.model_combo.setToolTip("Choose a dense-capable backbone for feature extraction.\nDINOv2: Fast, strong general-purpose features.\nOpenCLIP/TIMM: Specialized vision models.")

        model_layout.addRow("Model:", self.model_combo)

        dense_note = QLabel(
            "Dense backbones only (ViT / ConvNext). YOLO and Color models are "
            "image-level only and don't produce feature maps."
        )
        dense_note.setWordWrap(True)
        dense_note.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addRow("", dense_note)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Options group
        options_group = QGroupBox("Extraction Options")
        options_layout = QFormLayout()

        # Input resolution: the square edge (px) the image is resized to before
        # the backbone. Larger → denser feature grid (e.g. 768 → ~48×48 for /16)
        # → finer detail on the mesh, at higher VRAM/time cost.
        self.resolution_combo = QComboBox()
        for px in (256, 384, 512, 768, 1024, 1536, 2560):
            self.resolution_combo.addItem(str(px), px)
        default_res_idx = self.resolution_combo.findText("768")
        self.resolution_combo.setCurrentIndex(default_res_idx if default_res_idx >= 0 else 0)
        self.resolution_combo.setToolTip(
            "Square input edge in pixels. Larger gives a denser feature grid and "
            "finer detail, but uses more VRAM/time (attention scales with tokens²)."
        )
        options_layout.addRow("Input Resolution:", self.resolution_combo)

        # AnyUp upsampling: densifies the patch grid (e.g. 48x48 -> 192x192 for
        # 4x) using the RGB image as guidance, capped at the input resolution.
        self.upsample_combo = QComboBox()
        self.upsample_combo.addItems(["Off", "2x", "4x", "8x"])
        self.upsample_combo.setCurrentText("2x")
        self.upsample_combo.setToolTip(
            "Densify the feature grid with AnyUp (https://github.com/wimmerth/anyup), "
            "using the RGB image as guidance. Capped at the input resolution. "
            "First use downloads the model via torch.hub (requires internet)."
        )
        options_layout.addRow("Upsample (AnyUp):", self.upsample_combo)

        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["True", "False"])
        self.normalize_combo.setCurrentText("True")
        self.normalize_combo.setToolTip("L2-normalize features for cosine similarity")
        options_layout.addRow("Normalize:", self.normalize_combo)

        self.store_pooled_combo = QComboBox()
        self.store_pooled_combo.addItems(["True", "False"])
        self.store_pooled_combo.setCurrentText("False")
        self.store_pooled_combo.setToolTip(
            "Cache pooled vector for Explorer (enables cheap image-level similarity)"
        )
        options_layout.addRow("Store Pooled Vector:", self.store_pooled_combo)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Output group — controls how the 2D Feature Select tool finalizes a
        # thresholded similarity selection into an annotation.
        output_group = QGroupBox("Feature Select Output")
        output_layout = QFormLayout()

        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(["Polygon", "Mask"])
        self.output_type_combo.setCurrentText("Mask")
        self.output_type_combo.setToolTip(
            "What the 2D Feature Select tool creates when you press Space: vector "
            "Polygon annotations, or a paint into the raster Mask annotation."
        )
        output_layout.addRow("Output Type:", self.output_type_combo)

        self.allow_holes_combo = QComboBox()
        self.allow_holes_combo.addItems(["True", "False"])
        self.allow_holes_combo.setCurrentText("True")
        self.allow_holes_combo.setToolTip(
            "Preserve interior holes when polygonizing the selection (Polygon output)."
        )
        output_layout.addRow("Allow Holes:", self.allow_holes_combo)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("No model loaded")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        self.load_button.setToolTip("Load the selected feature extraction model into VRAM.")
        button_layout.addWidget(self.load_button)

        self.deactivate_button = QPushButton("Deactivate Model")
        self.deactivate_button.clicked.connect(self.deactivate_model)
        self.deactivate_button.setEnabled(False)
        self.deactivate_button.setToolTip("Unload the current model and free GPU memory.")
        button_layout.addWidget(self.deactivate_button)

        button_layout.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setToolTip("Close this dialog.")
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_output_type(self):
        """Return the Feature Select output type ('Polygon' or 'Mask')."""
        return self.output_type_combo.currentText()

    def get_allow_holes(self):
        """Return whether polygonized selections should preserve interior holes."""
        return self.allow_holes_combo.currentText() == "True"

    def load_model(self):
        """Load the selected feature extraction model."""
        model_name = self.model_combo.currentData()
        if not model_name:
            # Editable combo with custom (typed) text has no associated data
            model_name = self.model_combo.currentText()

        # Guard against custom-typed pooled-only models (YOLO / Color): they
        # cannot produce the dense feature maps this workflow requires.
        if not model_supports_dense(model_name):
            QMessageBox.warning(
                self,
                "Unsupported Model",
                f"'{model_name}' is a pooled-only model and cannot produce dense "
                "feature maps.\n\nChoose a ViT or ConvNext backbone instead.",
            )
            return

        # Check for HF_TOKEN if using DINOv3
        if "dinov3" in model_name.lower():
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token or not hf_token.strip():
                QMessageBox.warning(
                    self,
                    "HuggingFace Access Required",
                    "DINOv3 models require HuggingFace token.\n\n"
                    "Please:\n"
                    "1. Request access to the DINOv3 model on HuggingFace\n"
                    "2. Set HF_TOKEN environment variable\n"
                    "3. Restart the application",
                )
                return

        # Warn if CUDA not available
        if not self.cuda_warning_shown and not torch.cuda.is_available():
            reply = QMessageBox.question(
                self,
                "No CUDA Detected",
                "Feature extraction will run slowly without CUDA. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
            self.cuda_warning_shown = True

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Loading Model")
        progress_bar.show()

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_size = self.resolution_combo.currentData()
            upsample_text = self.upsample_combo.currentText()
            upsample_factor = int(upsample_text[:-1]) if upsample_text != "Off" else None
            self.loaded_model = FeatureExtractor(
                model_name, device=device, input_size=input_size,
                upsample_factor=upsample_factor
            )

            display_name = self.model_combo.currentText()
            self.status_label.setText(f"Loaded: {display_name}")
            self.load_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.model_combo.setEnabled(False)
            self.resolution_combo.setEnabled(False)
            self.upsample_combo.setEnabled(False)

            progress_bar.finish_progress()
            QMessageBox.information(self, "Model Loaded", "Feature model loaded successfully")

        except Exception as e:
            self.status_label.setText("Model loading failed")
            QMessageBox.critical(self, "Error Loading Model", f"Error: {e}")

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def deactivate_model(self):
        """Deactivate and clear the loaded model from memory."""
        if self.loaded_model is not None:
            try:
                self.loaded_model.clear_cache()
                self.loaded_model = None

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.status_label.setText("Model deactivated")
                self.load_button.setEnabled(True)
                self.deactivate_button.setEnabled(False)
                self.model_combo.setEnabled(True)
                self.resolution_combo.setEnabled(True)
                self.upsample_combo.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error Deactivating Model", f"Error: {e}")
