import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import uuid

import polars as pl
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportCoralNetAnnotations:
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

        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self.annotation_window,
                                                     "Import CoralNet Annotations",
                                                     "",
                                                     "CSV Files (*.csv);;All Files (*)",
                                                     options=options)

        if not file_paths:
            return

        annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                  "Patch Annotation Size",
                                                  "Enter the default patch annotation size for imported annotations:",
                                                  224, 1, 10000, 1)
        if not ok:
            return
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Reading CSV Files")
        progress_bar.show()

        try:
            all_data = []
            for file_path in file_paths:
                df = pl.read_csv(file_path)
                all_data.append(df)

            # Concatenate all the data
            df = pl.concat(all_data)

            required_columns = ['Name', 'Row', 'Column', 'Label']
            if not all(col in df.columns for col in required_columns):
                raise Exception("The selected CSV files do not match the expected CoralNet format.")

            # Filter out rows with missing values
            image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}
            df = df.with_column(pl.col('Name').apply(lambda x: os.path.basename(x)))
            df = df.filter(pl.col('Name').is_in(list(image_path_map.keys())))
            df = df.drop_nulls(subset=['Row', 'Column', 'Label'])
            df = df.with_column(pl.col('Row').cast(pl.Int32))
            df = df.with_column(pl.col('Column').cast(pl.Int32))

            if df.is_empty():
                raise Exception("No annotations found for loaded images.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")
            return
        
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing CoralNet Labels")
        progress_bar.show()

        try:
            # Pre-create all required labels
            all_labels = set(df['Label'].unique())
            machine_suggestions = [col for col in df.columns if 'Machine suggestion' in col]
            all_labels.update(df.select(machine_suggestions).to_numpy().flatten())
            
            progress_bar.start_progress(len(all_labels))
                
            for label_code in all_labels:
                if label_code is not None:
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
            progress_bar.setWindowTitle("Importing CoralNet Annotations")
            progress_bar.start_progress(len(df['Name'].unique()))
                    
            # Iterate over the rows
            for image_name, group in df.groupby('Name'):
                image_path = image_path_map.get(image_name)
                if not image_path:
                    continue

                for index, row in group.iterrows():
                    # Read from the row
                    row_coord = row['Row']
                    col_coord = row['Column']
                    label_code = row['Label']
                    
                    # Get the label codes
                    short_label_code = label_code
                    long_label_code = row['Long Label'] if 'Long Label' in row else label_code
                    
                    existing_label = self.label_window.get_label_by_codes(label_code, label_code)
                    color = existing_label.color
                    label_id = existing_label.id

                    annotation = PatchAnnotation(QPointF(col_coord, row_coord),
                                                 row['Patch Size'] if "Patch Size" in row else annotation_size,
                                                 short_label_code,
                                                 long_label_code,
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
                        if row[conf] is not None and row[sug] is not None
                    }
                    
                    # Process all valid pairs at once
                    for suggestion, confidence in valid_pairs:
                        suggested_label = self.label_window.get_label_by_short_code(suggestion)
                        
                        if not suggested_label:
                            color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            self.label_window.add_label_if_not_exists(suggestion, suggestion, color)
                            suggested_label = self.label_window.get_label_by_short_code(suggestion)
                            
                        machine_confidence[suggested_label] = confidence

                    # Update the machine confidence
                    annotation.update_machine_confidence(machine_confidence)

                    # Add annotation to the dict
                    self.annotation_window.add_annotation_to_dict(annotation)
                
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
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
