import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import polars as pl

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog, 
                            QVBoxLayout, QRadioButton, QPushButton, QGroupBox,
                            QLineEdit, QFormLayout, QHBoxLayout, QLabel)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportViscoreAnnotations(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Export Viscore Annotations")
        self.resize(400, 400)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Setup UI components
        self.setup_info_layout()
        self.setup_output_layout()
        self.setup_options_layout()
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """Set up the information section layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Export annotations to a Viscore-compatible CSV file. Choose the voting "
            "type for multi-view annotations and specify the save location."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """Set up the output file selection layout."""
        group_box = QGroupBox("Output")
        layout = QFormLayout()

        # CSV file selection
        self.csv_file_edit = QLineEdit()
        self.csv_file_button = QPushButton("Browse...")
        self.csv_file_button.clicked.connect(self.browse_csv_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.csv_file_edit)
        file_layout.addWidget(self.csv_file_button)
        layout.addRow("Save Location:", file_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """Set up the voting options layout."""
        group_box = QGroupBox("Voting Options")
        layout = QVBoxLayout()

        # Create horizontal layout for radio buttons
        radio_layout = QHBoxLayout()
        
        # Create vertical layouts for each option
        single_layout = QVBoxLayout()
        multi_layout = QVBoxLayout()

        # Create and add info labels
        single_info = QLabel("Export annotations as-is")
        single_info.setWordWrap(True)
        multi_info = QLabel("Calculate consensus label for each dot")
        multi_info.setWordWrap(True)

        # Create radio buttons
        self.single_voting = QRadioButton("SingleVoting")
        self.multi_voting = QRadioButton("MultiVoting")
        self.single_voting.setChecked(True)

        # Add widgets to their respective vertical layouts
        single_layout.addWidget(single_info)
        single_layout.addWidget(self.single_voting)
        multi_layout.addWidget(multi_info)
        multi_layout.addWidget(self.multi_voting)

        # Add vertical layouts to horizontal layout
        radio_layout.addLayout(single_layout)
        radio_layout.addLayout(multi_layout)

        # Add horizontal layout to main layout
        layout.addLayout(radio_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Set up the bottom buttons layout."""
        buttons_layout = QHBoxLayout()

        self.export_button = QPushButton("Export")
        self.cancel_button = QPushButton("Cancel")

        self.export_button.clicked.connect(self.export_annotations)
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.cancel_button)

        self.layout.addLayout(buttons_layout)

    def browse_csv_file(self):
        """Open file dialog to select save location."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Viscore Annotations",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            self.csv_file_edit.setText(file_path)

    def export_annotations(self):
        """Handle the annotation export process."""
        file_path = self.csv_file_edit.text()
        if not file_path:
            QMessageBox.warning(self, 
                                "Invalid File",
                                "Please select a save location.")
            return

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Exporting Viscore Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            # Collect annotations
            df = []
            dots_dict = {}

            for annotation in self.annotation_window.annotations_dict.values():
                if isinstance(annotation, PatchAnnotation):
                    if 'Dot' in annotation.data:
                        # Get the annotation data
                        data = annotation.to_coralnet()
                        data['Name'] = annotation.image_path

                        # Add the dot to the dots_dict dictionary
                        dot = annotation.data['Dot']
                        if dot not in dots_dict:
                            dots_dict[dot] = []
                        dots_dict[dot].append(data)

                        # Add the data to the dataframe
                        df.append(data)

                        # Update the progress bar
                        progress_bar.update_progress()

            # Apply voting if selected
            if self.multi_voting.isChecked():
                df = self.multiview_voting(dots_dict, progress_bar)

            # Save to CSV
            df = pl.DataFrame(df)
            df.write_csv(file_path)

            QMessageBox.information(self, 
                                    "Success",
                                    "Annotations have been successfully exported.")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, 
                                 "Critical Error",
                                 f"Failed to export annotations: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def multiview_voting(self, dots_dict, progress_bar):
        """Calculate the multiview voting (consensus) for each dot."""
        try:
            progress_bar.setWindowTitle("Calculating Consensus")
            progress_bar.start_progress(len(dots_dict))

            consensus_results = []

            # Iterate over the dots
            for dot, annotations in dots_dict.items():
                votes = {}

                # Accumulate weighted votes from each annotation's machine suggestions
                for ann in annotations:
                    for i in range(1, 6):
                        suggestion = ann.get(f"Machine suggestion {i}")
                        confidence = ann.get(f"Machine confidence {i}")
                        if suggestion is None or confidence is None:
                            continue
                        votes[suggestion] = votes.get(suggestion, 0) + float(confidence)

                # Determine consensus by highest accumulated confidence
                consensus_suggestion = max(votes, key=votes.get) if votes else None

                # Update annotations with consensus
                for ann in annotations:
                    ann["Label"] = consensus_suggestion
                    consensus_results.append(ann)

                progress_bar.update_progress()

            return consensus_results

        except Exception as e:
            QMessageBox.warning(self, 
                                "Error Calculating Consensus",
                                f"Failed to calculate consensus: {str(e)}")
            return []
