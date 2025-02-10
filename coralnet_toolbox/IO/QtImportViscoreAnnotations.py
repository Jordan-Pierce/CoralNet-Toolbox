import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import uuid

import pandas as pd
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog, 
                             QLineEdit, QDialog, QVBoxLayout, QLabel, QHBoxLayout, 
                             QPushButton, QDialogButtonBox, QSpinBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportViscoreAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_annotations(self):
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        def browse_csv_file(file_path_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                       "Import Viscore Annotations",
                                                       "",
                                                       "CSV Files (*.csv);;All Files (*)",
                                                       options=options)
            if file_path:
                file_path_input.setText(file_path)

        dialog = QDialog(self.annotation_window)
        dialog.setWindowTitle("Import Viscore Annotations")
        dialog.resize(300, 100)

        layout = QVBoxLayout(dialog)

        file_path_label = QLabel("CSV File Path:")
        file_path_input = QLineEdit()
        file_path_button = QPushButton("Browse")
        file_path_button.clicked.connect(lambda: browse_csv_file(file_path_input))
        file_path_layout = QHBoxLayout()
        file_path_layout.addWidget(file_path_input)
        file_path_layout.addWidget(file_path_button)
        layout.addWidget(file_path_label)
        layout.addLayout(file_path_layout)

        views_label = QLabel("Number of Views:")
        views_spinbox = QSpinBox()
        views_spinbox.setMinimum(1)
        views_spinbox.setMaximum(100)
        views_spinbox.setValue(3)
        views_spinbox.setToolTip("Number of best views to keep per dot")
        layout.addWidget(views_label)
        layout.addWidget(views_spinbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            file_path = file_path_input.text()

            views = views_spinbox.value()
            if views < 1:
                QMessageBox.warning(self.annotation_window, 
                                    "Invalid Input", 
                                    "Number of views must be a positive integer.")
                return

            try:
                # Make cursor busy
                QApplication.setOverrideCursor(Qt.WaitCursor)
                progress_bar = ProgressBar(self.annotation_window, title="Reading CSV File")
                progress_bar.show()
                df = pd.read_csv(file_path, index_col=False)

                if df.empty:
                    QMessageBox.warning(self.annotation_window, 
                                        "Empty CSV", 
                                        "The CSV file is empty.")
                    return

                required_columns = ['Name',
                                    'Row',
                                    'Column',
                                    'Label',
                                    'Dot',
                                    'X',
                                    'Y',
                                    'Z',
                                    'ReprojectionError',
                                    'ViewIndex',
                                    'ViewCount']

                if not all(col in df.columns for col in required_columns):
                    QMessageBox.warning(self.annotation_window,
                                        "Invalid CSV Format",
                                        "The selected CSV file does not match the expected Viscore format.")
                    return

                df = df[required_columns]
                df = df.dropna(how='any')
                df = df.assign(Row=df['Row'].astype(int))
                df = df.assign(Column=df['Column'].astype(int))

                image_paths = df['Name'].unique()
                image_paths = [path for path in image_paths if os.path.exists(path)]

                if not image_paths and not self.annotation_window.active_image:
                    QMessageBox.warning(self.annotation_window,
                                        "No Images Found",
                                        "Please load an image before importing annotations.")
                    return

                # Perform the filtering
                progress_bar.setWindowTitle("Filtering Annotations")
                progress_bar.start_progress(len(df['Dot'].unique()))
                
                filtered = []
                
                for dot in df['Dot'].unique():
                    subset = df[df['Dot'] == dot]
                    reprojection_error = subset['ReprojectionError']
                    # Calculate the mean of the reprojection error
                    mean = reprojection_error.mean()
                    # Filter the subset to get only rows with reprojection error less than the mean
                    subset = subset[reprojection_error <= mean]
                    # Get the new mean and std
                    std = reprojection_error.std()
                    mean = reprojection_error.mean()
                    # Subset to get only rows within +/- one standard deviation of the mean
                    lower_bound = mean - std
                    upper_bound = mean + std
                    subset = subset[(reprojection_error >= lower_bound) & (reprojection_error <= upper_bound)]
                    # Sort based on reprpjection error and ViewIndex, ascending
                    subset = subset.sort_values(['ReprojectionError', 'ViewIndex'], ascending=[True, True])
                    # Get the first N views
                    subset = subset.head(views)
                    filtered.append(subset)
                    
                    # Update the progress bar
                    progress_bar.update_progress()
                    
                df = pd.concat(filtered)
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df['Row'] = df['Row'].astype(int)
                df['Column'] = df['Column'].astype(int)

                num_images = df['Name'].nunique()
                num_annotations = len(df)

                msg_box = QMessageBox(self.annotation_window)
                msg_box.setWindowTitle("Filtered Data Summary")
                msg_box.setText(f"Number of Images: {num_images}\nNumber of Annotations: {num_annotations}")
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg_box.setDefaultButton(QMessageBox.Ok)

                result = msg_box.exec_()

                if result == QMessageBox.Cancel:
                    return

                annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                          "Annotation Size",
                                                          "Enter the annotation size for all imported annotations:",
                                                          224, 1, 10000, 1)
                if not ok:
                    return

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Importing Annotations",
                                    f"An error occurred while importing annotations: {str(e)}")
                return
            
            finally:
                # Restore the cursor
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()

            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Importing Viscore Annotations")
            progress_bar.show()
    
            try:
                # Map image names to image paths
                image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}

                # Pre-create all required labels
                all_labels = set(df['Label'].unique())
                progress_bar.start_progress(len(all_labels))
                
                text = 'Machine suggestion'
                machine_suggestions = [col.replace(text, '') for col in df.columns if col.startswith(text)]
                all_labels.update(df[machine_suggestions].values.flatten())

                for label_code in all_labels:
                    if pd.notna(label_code):
                        short_label_code = long_label_code = str(label_code)
                        if not self.label_window.get_label_by_codes(short_label_code, long_label_code):
                            
                            label_id = str(uuid.uuid4())
                            color = QColor(random.randint(0, 255),
                                           random.randint(0, 255),
                                           random.randint(0, 255))
                            
                            self.label_window.add_label_if_not_exists(short_label_code,
                                                                      long_label_code,
                                                                      color,
                                                                      label_id)
                    progress_bar.update_progress()

                # Start the progress bar
                progress_bar.start_progress(len(df['Name'].unique()))
                
                for image_name, group in df.groupby('Name'):
                    image_path = image_path_map.get(os.path.basename(image_name))
                    if not image_path:
                        progress_bar.update_progress()
                        continue

                    for index, row in group.iterrows():
                        row_coord = row['Row']
                        col_coord = row['Column']
                        label_code = row['Label']
                        
                        existing_label = self.label_window.get_label_by_codes(label_code, label_code)
                        color = existing_label.color
                        label_id = existing_label.id

                        annotation = PatchAnnotation(QPointF(col_coord, row_coord),
                                                     annotation_size,
                                                     label_code,
                                                     label_code,
                                                     color,
                                                     image_path,
                                                     label_id)

                        machine_confidence = {}
                    
                        # Get all confidence and suggestion columns
                        confidence_cols = [col for col in row.index if col.startswith('Machine confidence')]
                        suggestion_cols = [col for col in row.index if col.startswith('Machine suggestion')]
                        
                        # Create pairs of valid confidence and suggestion values
                        valid_pairs = {
                            (str(row[sug]), float(row[conf]))
                            for conf, sug in zip(confidence_cols, suggestion_cols)
                            if pd.notna(row[conf]) and pd.notna(row[sug])
                        }
                        
                        # Process all valid pairs at once
                        for suggestion, confidence in valid_pairs:
                            suggested_label = self.label_window.get_label_by_short_code(suggestion)
                            if suggested_label:
                                machine_confidence[suggested_label] = confidence

                        # Update the machine confidence
                        annotation.update_machine_confidence(machine_confidence)
                        
                        # Add the rest of the annotation data
                        annotation.data = {
                            'Dot': row['Dot'],
                            'X': row['X'],
                            'Y': row['Y'],
                            'Z': row['Z'],
                            'ReprojectionError': row['ReprojectionError'],
                            'ViewIndex': row['ViewIndex'],
                            'ViewCount': row['ViewCount']
                        }

                        # Add annotation to the dict
                        self.annotation_window.annotations_dict[annotation.id] = annotation
                        
                    # Update the progress bar
                    progress_bar.update_progress()

                    # Update the image window's image dict
                    self.image_window.update_image_annotations(image_path)

                # Load the annotations for current image
                self.annotation_window.load_annotations()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.critical(self.annotation_window,
                                     "Critical Error",
                                     f"Failed to import annotations: {e}")

            finally:
                # Restore the cursor
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()
