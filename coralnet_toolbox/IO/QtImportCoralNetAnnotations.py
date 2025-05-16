import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import uuid

import pandas as pd
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


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
                df = pd.read_csv(file_path)
                all_data.append(df)

            # Concatenate all the data
            df = pd.concat(all_data, ignore_index=True)
            
            # Check if Label Code is present instead of Label; 
            # in the CoralNet Annotation file, 'Label code' refers 'Shot Code' in Labelset file.
            if 'Label code' in df.columns and 'Label' not in df.columns:
                df = df.rename(columns={'Label code': 'Label'})

            required_columns = ['Name', 'Row', 'Column', 'Label']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise Exception(f"The selected CSV file(s) are missing necessary columns: {missing_columns}")

            # Filter out rows with missing values
            image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}
            df['Name'] = df['Name'].apply(lambda x: os.path.basename(x))
            df = df[df['Name'].isin(image_path_map.keys())]
            df = df.dropna(how='any', subset=['Row', 'Column', 'Label'])
            df = df.assign(Row=df['Row'].astype(int))
            df = df.assign(Column=df['Column'].astype(int))

            if df.empty:
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
        progress_bar = ProgressBar(self.annotation_window, title="Importing CoralNet Annotations")
        progress_bar.start_progress(len(df['Name'].unique()))
        progress_bar.show()
        
        # Import the annotations
        try:
            # Iterate over the rows of annotations by image basename
            for image_name, group in df.groupby('Name'):
                
                # Check that the image exists (using basename)
                image_path = image_path_map.get(image_name)
                if not image_path:
                    continue
                
                # Iterate over the rows of annotations (for this image)
                for index, row in group.iterrows():
                    # Read from the row
                    row_coord = row['Row']
                    col_coord = row['Column']
                    label_code = row['Label']
                    
                    # Get the label codes
                    short_label_code = label_code
                    # If the user previously exported from the Toolbox, the 'Long Label' column will be present
                    long_label_code = row['Long Label'] if 'Long Label' in row and pd.notna(row['Long Label']) else None
                    
                    # Check if the label already exists, create it if not
                    label = self.label_window.add_label_if_not_exists(short_label_code, 
                                                                      long_label_code,
                                                                      color=None,
                                                                      label_id=None)
                    # Get the label color and ID
                    color = label.color
                    label_id = label.id
                            
                    # Create the annotation
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
                        if pd.notna(row[conf]) and pd.notna(row[sug])
                    }
                    
                    # Check if the sum of all valid confidences is greater than 1
                    if sum([conf for sug, conf in valid_pairs]) > 1:
                        valid_pairs = [(sug, conf / 100) for sug, conf in valid_pairs]
                        
                    # Process all valid pairs at once
                    for suggestion, confidence in valid_pairs:
                        # Get the label object using the short code (because that's all that's available)
                        suggested_label = self.label_window.add_label_if_not_exists(short_label_code=suggestion, 
                                                                                    long_label_code=suggestion)
                            
                        machine_confidence[suggested_label] = confidence

                    # Update the machine confidence
                    annotation.update_machine_confidence(machine_confidence, from_import=True)
                    
                    if 'Verified' in row:
                        # If the verified status is True, update the annotation's verified status
                        verified = str(row['Verified']).lower() == 'true' or row['Verified'] == 1 
                        annotation.set_verified(verified)
                            
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
