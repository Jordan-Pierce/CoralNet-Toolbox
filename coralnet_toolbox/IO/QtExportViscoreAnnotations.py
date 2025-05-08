import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import ujson as json

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog,
                             QVBoxLayout, QRadioButton, QPushButton, QGroupBox,
                             QLineEdit, QFormLayout, QHBoxLayout, QLabel, QTabWidget,
                             QWidget)

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
        self.resize(600, 150)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Setup info section
        self.setup_info_layout()

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.setup_export_csv_tab()
        self.setup_export_json_tab()
        self.layout.addWidget(self.tab_widget)

        # Setup buttons at bottom
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """Set up the information section layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Export annotations to a Viscore-compatible CSV or JSON file."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_export_csv_tab(self):
        """Set up the Viscore export CSV tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add output group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        self.csv_file_edit = QLineEdit()
        self.csv_file_button = QPushButton("Browse...")
        self.csv_file_button.clicked.connect(lambda: self.save_file(self.csv_file_edit, "CSV File (*.csv)"))
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.csv_file_edit)
        file_layout.addWidget(self.csv_file_button)
        output_layout.addRow("Export CSV:", file_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Add voting options group
        layout.addWidget(self.create_voting_options())

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Export JSON")

    def create_voting_options(self):
        """Create and return the voting options group."""
        group_box = QGroupBox("Voting Options")
        layout = QVBoxLayout()

        radio_layout = QHBoxLayout()
        single_layout = QVBoxLayout()
        multi_layout = QVBoxLayout()

        single_info = QLabel("Export annotations as-is")
        single_info.setWordWrap(True)
        multi_info = QLabel("Calculate consensus label for each dot")
        multi_info.setWordWrap(True)

        self.single_voting = QRadioButton("SingleVoting")
        self.multi_voting = QRadioButton("MultiVoting")
        self.single_voting.setChecked(True)

        single_layout.addWidget(single_info)
        single_layout.addWidget(self.single_voting)
        multi_layout.addWidget(multi_info)
        multi_layout.addWidget(self.multi_voting)

        radio_layout.addLayout(single_layout)
        radio_layout.addLayout(multi_layout)
        layout.addLayout(radio_layout)

        group_box.setLayout(layout)
        return group_box

    def setup_export_json_tab(self):
        """Set up the advanced options tab."""
        tab = QWidget()
        layout = QFormLayout()

        # Create input group box
        input_group = QGroupBox("Input")
        input_form = QFormLayout()

        # JSON file choosers
        self.label_json_edit = QLineEdit()
        self.label_json_button = QPushButton("Browse...")
        self.label_json_button.clicked.connect(lambda: self.browse_file(self.label_json_edit, "JSON Files (*.json)"))
        label_json_layout = QHBoxLayout()
        label_json_layout.addWidget(self.label_json_edit)
        label_json_layout.addWidget(self.label_json_button)
        input_form.addRow("Labelset File (JSON):", label_json_layout)

        self.user_json_edit = QLineEdit()
        self.user_json_button = QPushButton("Browse...")
        self.user_json_button.clicked.connect(lambda: self.browse_file(self.user_json_edit, "JSON Files (*.json)"))
        user_json_layout = QHBoxLayout()
        user_json_layout.addWidget(self.user_json_edit)
        user_json_layout.addWidget(self.user_json_button)
        input_form.addRow("User File (JSON, Optional):", user_json_layout)

        input_group.setLayout(input_form)
        layout.addWidget(input_group)

        # Create output group box
        output_group = QGroupBox("Output")
        output_form = QFormLayout()

        # Output directory chooser
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_directory)
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.output_dir_button)
        output_form.addRow("Output Directory:", dir_layout)

        # Username field
        self.username_edit = QLineEdit()
        output_form.addRow("User Name:", self.username_edit)

        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Export JSON")

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

    def save_file(self, line_edit, file_filter):
        """Generic file save for file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            file_filter,
            options=options
        )
        if file_path:
            line_edit.setText(file_path)

    def browse_file(self, line_edit, file_filter):
        """Generic file browser for files."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            file_filter,
            options=options
        )
        if file_path:
            line_edit.setText(file_path)

    def browse_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def export_annotations(self):
        """Handle the annotation export process."""
        if self.tab_widget.currentIndex() == 0:
            self.export_csv_annotations()
        else:
            self.export_json_annotations()

    def export_csv_annotations(self):
        """Handle the CSV annotation export process."""
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
                    # Skip annotations for images not in the raster manager
                    if annotation.image_path not in self.image_window.raster_manager.image_paths:
                        continue

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

            if not df or not dots_dict:
                QMessageBox.warning(self,
                                    "No Annotations",
                                    "No annotations found in project with Viscore dot data.")

            # Apply voting if selected
            if self.multi_voting.isChecked():
                df = self.multiview_voting(dots_dict, progress_bar)

            # Save to CSV
            df = pd.DataFrame(df)
            df.to_csv(file_path, index=False)

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

    def export_json_annotations(self):
        """Handle the JSON annotation export process."""
        # Extract the file paths
        labelset_json_path = self.label_json_edit.text()
        user_json_path = self.user_json_edit.text()

        # Check if labelset file is selected and exists
        if not labelset_json_path:
            QMessageBox.warning(self,
                                "Invalid File",
                                "Please select a labelset JSON file.")
            return

        if not os.path.exists(labelset_json_path):
            QMessageBox.warning(self,
                                "Invalid File",
                                "Labelset file does not exist.")
            return

        # Check if user file exists if provided
        if user_json_path and not os.path.exists(user_json_path):
            QMessageBox.warning(self,
                                "Invalid File",
                                "User file does not exist.")
            return

        # Extract the output directory
        output_dir = self.output_dir_edit.text()
        os.makedirs(output_dir, exist_ok=True)

        # Create the output file path
        username = self.username_edit.text()
        output_path = f"{output_dir}/samples.cl.user.{username}.json"

        # Read the labelset file
        with open(labelset_json_path, 'r') as f:
            labelset_file = json.load(f)

        if 'classlist' not in labelset_file:
            QMessageBox.warning(self,
                                "Invalid File",
                                "The labelset file does not contain a classlist.")
            return

        # Extract the classlist, create a DataFrame
        classlist = pd.DataFrame(labelset_file['classlist'], columns=['id', 'short_name', 'long_name'])

        # Initialize output file structure
        output_file = {
            "savefileb": os.path.basename(output_path),
            "savefile": output_path,
        }

        # Read the user file if provided, otherwise we'll fill cl later
        if user_json_path:
            with open(user_json_path, 'r') as f:
                user_file = json.load(f)
            output_file["cl"] = [int(x) for x in user_file['cl']]

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self, title="Reading Annotations")
            progress_bar.show()

            # Start progress bar
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            # Collect annotations
            dots_dict = {}
            missing_labels = set()

            for annotation in self.annotation_window.annotations_dict.values():
                if isinstance(annotation, PatchAnnotation):
                    # Skip annotations for images not in the raster manager
                    if annotation.image_path not in self.image_window.raster_manager.image_paths:
                        continue

                    if 'Dot' in annotation.data:
                        # Get the annotation data
                        data = annotation.to_coralnet()
                        # Get the dot ID for the annotation
                        dot = annotation.data['Dot']
                        if dot not in dots_dict:
                            dots_dict[dot] = []
                        dots_dict[dot].append(data)

                        # Update the progress bar
                        progress_bar.update_progress()

            # If user file wasn't provided, initialize cl with -1 for each dot
            if not user_json_path:
                max_dot_id = max(dots_dict.keys()) if dots_dict else -1
                output_file["cl"] = [-1] * (max_dot_id + 1)

            # Update progress bar
            progress_bar.setWindowTitle("Calculating Consensus")
            progress_bar.start_progress(len(dots_dict))

            # Loop through each dot
            for dot, annotations in dots_dict.items():
                votes = {}

                # Loop through each annotation associated with the dot
                for ann in annotations:
                    for i in range(1, 6):
                        suggestion = ann.get(f"Machine suggestion {i}")
                        confidence = ann.get(f"Machine confidence {i}")

                        if suggestion is None or confidence is None:
                            continue

                        votes[suggestion] = votes.get(suggestion, 0) + float(confidence)

                # Calculate the consensus suggestion
                consensus_suggestion = max(votes, key=votes.get) if votes else None

                # If the index is currently under Review, update
                if output_file['cl'][dot] == -1 and consensus_suggestion:
                    try:
                        # Check if label exists in classlist
                        label_exists = classlist['long_name'].eq(consensus_suggestion).any()
                        if not label_exists:
                            missing_labels.add(consensus_suggestion)
                            label_id = -1
                        else:
                            label_id = int(classlist[classlist['long_name'] == consensus_suggestion]['id'].values[0])
                    except Exception as e:
                        missing_labels.add(consensus_suggestion)
                        label_id = -1

                    output_file['cl'][dot] = label_id

                # Update the progress bar
                progress_bar.update_progress()

            # Ensure all values in cl are regular Python ints before saving
            output_file['cl'] = [int(x) for x in output_file['cl']]

            # Save output json file
            with open(output_path, 'w') as f:
                json.dump(output_file, f, indent=4)

            if missing_labels:
                # Sort the missing labels
                sorted_set = ', '.join(sorted([str(label) for label in missing_labels]))

                QMessageBox.warning(self,
                                    "Warning",
                                    f"The following labels were not found in the classlist:\n{sorted_set}")

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
