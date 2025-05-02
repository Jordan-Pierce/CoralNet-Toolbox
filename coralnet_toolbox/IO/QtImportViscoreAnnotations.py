import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import uuid
import random
import pandas as pd

from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog,
                             QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit,
                             QSpinBox, QGroupBox, QFormLayout, QStyle)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ImportViscoreAnnotations(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Import Viscore Annotations")
        self.resize(400, 200)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Setup UI components
        self.setup_info_layout()
        self.setup_input_layout()
        self.setup_options_layout()
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """Set up the information section layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Import annotations from a Viscore CSV file. The file should contain "
            "columns for Name, Row, Column, Label, Dot, X, Y, Z, ReprojectionError, "
            "ViewIndex, and ViewCount."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_input_layout(self):
        """Set up the input file selection layout."""
        group_box = QGroupBox("Input")
        layout = QFormLayout()

        # CSV file selection
        self.csv_file_edit = QLineEdit()
        self.csv_file_button = QPushButton("Browse...")
        self.csv_file_button.clicked.connect(self.browse_csv_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.csv_file_edit)
        file_layout.addWidget(self.csv_file_button)
        layout.addRow("CSV File:", file_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """Set up the import options layout."""
        group_box = QGroupBox("Options")
        layout = QFormLayout()

        # Views per dot
        self.views_spinbox = QSpinBox()
        self.views_spinbox.setMinimum(-1)
        self.views_spinbox.setMaximum(1000)
        self.views_spinbox.setValue(3)
        self.views_spinbox.setToolTip("Number of best views to keep per dot")
        layout.addRow("Views per Dot:", self.views_spinbox)

        # Annotation size
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(1)
        self.size_spinbox.setMaximum(10000)
        self.size_spinbox.setValue(224)
        self.size_spinbox.setToolTip("Size of annotation patches in pixels")
        layout.addRow("Annotation Size:", self.size_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Set up the bottom buttons layout."""
        buttons_layout = QHBoxLayout()

        self.import_button = QPushButton("Import")
        self.cancel_button = QPushButton("Cancel")

        self.import_button.clicked.connect(self.import_annotations)
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.import_button)
        buttons_layout.addWidget(self.cancel_button)

        self.layout.addLayout(buttons_layout)

    def browse_csv_file(self):
        """Open file dialog to select CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Viscore Annotations",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            self.csv_file_edit.setText(file_path)

    def import_annotations(self):
        """Handle the annotation import process."""
        if not self.annotation_window.active_image:
            QMessageBox.warning(self, 
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        csv_path = self.csv_file_edit.text()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, 
                                "Invalid File",
                                "Please select a valid CSV file.")
            return

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Reading CSV File")
            progress_bar.show()

            # Read and validate CSV
            df = self.read_and_validate_csv(csv_path)
            if df is None:
                return

            # Filter annotations
            df = self.filter_annotations(df, progress_bar)
            if df is None:
                return

            # Import the annotations
            self.process_annotations(df, progress_bar)

        except Exception as e:
            QMessageBox.critical(self, 
                                 "Critical Error",
                                 f"Failed to import annotations: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            self.accept()

    def read_and_validate_csv(self, csv_path):
        """Read and validate the CSV file."""
        try:
            df = pd.read_csv(csv_path, index_col=False)

            if df.empty:
                QMessageBox.warning(self, 
                                    "Empty CSV",
                                    "The CSV file is empty.")
                return None

            required_columns = ['Name', 'Row', 'Column', 'Label', 'Dot', 'X', 'Y', 'Z',
                                'ReprojectionError', 'ViewIndex', 'ViewCount']

            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(self, "Invalid CSV Format",
                                    "The selected CSV file does not match the expected Viscore format.")
                return None

            return df[required_columns]

        except Exception as e:
            QMessageBox.warning(self, "Error Reading CSV",
                                f"Failed to read CSV file: {str(e)}")
            return None

    def filter_annotations(self, df, progress_bar):
        """Filter annotations based on reprojection error and views per dot."""
        try:
            progress_bar.setWindowTitle("Filtering Annotations")
            progress_bar.start_progress(len(df['Dot'].unique()))

            filtered = []
            views = self.views_spinbox.value()

            for dot in df['Dot'].unique():
                subset = df[df['Dot'] == dot]
                reprojection_error = subset['ReprojectionError']

                # Calculate mean and filter by it
                mean = reprojection_error.mean()
                subset = subset[reprojection_error <= mean]

                # Calculate new mean and std
                std = reprojection_error.std()
                mean = reprojection_error.mean()

                # Filter within +/- one standard deviation
                lower_bound = mean - std
                upper_bound = mean + std
                subset = subset[(reprojection_error >= lower_bound) & (reprojection_error <= upper_bound)]

                # Sort and get top N views
                subset = subset.sort_values(['ReprojectionError', 'ViewIndex'], ascending=[True, True])
                views = views if views != -1 else len(subset)
                subset = subset.head(views)
                filtered.append(subset)

                progress_bar.update_progress()

            df = pd.concat(filtered)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            QMessageBox.warning(self, "Error Filtering",
                                f"Failed to filter annotations: {str(e)}")
            return None

    def process_annotations(self, df, progress_bar):
        """Process and import the annotations."""
        try:
            # Map image names to paths
            image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}

            # Pre-create required labels
            progress_bar.setWindowTitle("Creating Labels")
            progress_bar.start_progress(len(df['Label'].unique()))

            for label_code in df['Label'].unique():
                if pd.notna(label_code):
                    short_code = long_code = str(label_code)
                    if not self.label_window.get_label_by_codes(short_code, long_code):
                        label_id = str(uuid.uuid4())
                        color = QColor(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))

                        self.label_window.add_label_if_not_exists(short_code,
                                                                  long_code,
                                                                  color,
                                                                  label_id)
                progress_bar.update_progress()

            # Import annotations
            progress_bar.setWindowTitle("Importing Annotations")
            progress_bar.start_progress(len(df['Name'].unique()))

            annotation_size = self.size_spinbox.value()

            for image_name, group in df.groupby('Name'):
                image_path = image_path_map.get(os.path.basename(image_name))
                if not image_path:
                    progress_bar.update_progress()
                    continue

                for _, row in group.iterrows():
                    row_coord = int(row['Row'])
                    col_coord = int(row['Column'])
                    label_code = str(row['Label'])

                    existing_label = self.label_window.get_label_by_codes(label_code, label_code)

                    annotation = PatchAnnotation(
                        QPointF(col_coord, row_coord),
                        annotation_size,
                        label_code,
                        label_code,
                        existing_label.color,
                        image_path,
                        existing_label.id
                    )

                    annotation.data = {
                        'Dot': row['Dot'],
                        'X': row['X'],
                        'Y': row['Y'],
                        'Z': row['Z'],
                        'ReprojectionError': row['ReprojectionError'],
                        'ViewIndex': row['ViewIndex'],
                        'ViewCount': row['ViewCount']
                    }

                    self.annotation_window.add_annotation_to_dict(annotation)

                self.image_window.update_image_annotations(image_path)
                progress_bar.update_progress()

            # Load annotations for current image
            self.annotation_window.load_annotations()

        except Exception as e:
            QMessageBox.critical(self, 
                                 "Critical Error",
                                 f"Failed to process annotations: {str(e)}")