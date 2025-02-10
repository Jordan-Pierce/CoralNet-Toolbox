import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog, 
                            QVBoxLayout, QRadioButton, QPushButton, QGroupBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportViscoreAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_annotations(self):
        self.main_window.untoggle_all_tools()

        # Create voting type dialog
        dialog = QDialog(self.annotation_window)
        dialog.setWindowTitle("Export Options")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        # File selection group
        file_group = QGroupBox("Save Location")
        file_layout = QVBoxLayout()
        file_button = QPushButton("Select Save Location...")
        self.file_path = None
        
        def select_file():
            options = QFileDialog.Options()
            path, _ = QFileDialog.getSaveFileName(
                dialog,
                "Export Viscore Annotations",
                "",
                "CSV Files (*.csv);;All Files (*)",
                options=options
            )
            if path:
                self.file_path = path
                file_button.setText(f"Selected: {path}")
        
        file_button.clicked.connect(select_file)
        file_layout.addWidget(file_button)
        file_group.setLayout(file_layout)
        
        # Voting type group
        group_box = QGroupBox("Voting Type")
        group_layout = QVBoxLayout()
        
        single_voting = QRadioButton("SingleVoting")
        multi_voting = QRadioButton("MultiVoting")
        single_voting.setChecked(True)
        
        group_layout.addWidget(single_voting)
        group_layout.addWidget(multi_voting)
        group_box.setLayout(group_layout)
        
        export_button = QPushButton("Export")
        export_button.clicked.connect(lambda: dialog.accept() if self.file_path else None)
        
        layout.addWidget(file_group)
        layout.addWidget(group_box)
        layout.addWidget(export_button)
        dialog.setLayout(layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
            
        voting_type = "MultiVoting" if multi_voting.isChecked() else "SingleVoting"
        file_path = self.file_path
        
        if file_path:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Exporting Viscore Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            try:
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
                
                if voting_type == "MultiVoting":
                    df = self.multiview_voting(dots_dict)

                df = pd.DataFrame(df)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        f"Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            finally:
                # Restore the cursor
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()
    
    def multiview_voting(self, dots_dict):
        """Calculate the multiview voting (consensus) for each dot."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Calculating Consensus")
        progress_bar.show()
        progress_bar.start_progress(len(dots_dict))
        
        try:
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
                
                # Determine the consensus suggestion by highest accumulated confidence
                consensus_suggestion = max(votes, key=votes.get) if votes else None
                
                # Update the label for each annotation with the consensus suggestion
                for ann in annotations:
                    ann["Label"] = consensus_suggestion
                    consensus_results.append(ann)
                
                # Update the progress bar
                progress_bar.update_progress()
        
        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Calculating Consensus",
                                f"An error occurred while calculating consensus: {str(e)}")
            consensus_results = []
            
        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
        
        return consensus_results