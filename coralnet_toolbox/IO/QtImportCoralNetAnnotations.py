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
        """Import annotations from CoralNet CSV export files into the current project."""
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

            df = pd.concat(all_data, ignore_index=True)
            
            if 'Label code' in df.columns and 'Label' not in df.columns:
                df = df.rename(columns={'Label code': 'Label'})

            required_columns = ['Name', 'Row', 'Column', 'Label']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise Exception(f"The selected CSV file(s) are missing necessary columns: {missing_columns}")

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
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing CoralNet Annotations")
        progress_bar.start_progress(len(df)) # Progress per annotation row
        progress_bar.show()
        
        try:
            # Pre-process labels to avoid repeated lookups
            unique_labels = set(df['Label'].unique())
            if 'Machine suggestion' in df.columns:
                unique_labels.update(df['Machine suggestion'].dropna().unique())
            
            label_cache = {}
            for label_code in unique_labels:
                label = self.label_window.add_label_if_not_exists(label_code, label_code)
                label_cache[label_code] = label
            
            annotations_to_add = []
            images_to_update = set()
            
            # 1. Create all annotation objects in memory first
            for _, row in df.iterrows():
                image_name = row['Name']
                image_path = image_path_map.get(image_name)
                if not image_path:
                    continue

                label = label_cache[row['Label']]
                
                annotation = PatchAnnotation(
                    QPointF(row['Column'], row['Row']),
                    row.get('Patch Size', annotation_size),
                    label.short_label_code,
                    label.long_label_code,
                    label.color,
                    image_path,
                    label.id
                )
                
                machine_confidence = self._extract_machine_confidence(row, label_cache)
                if machine_confidence:
                    annotation.update_machine_confidence(machine_confidence, from_import=True)
                
                if 'Verified' in row:
                    verified = str(row['Verified']).lower() == 'true' or row['Verified'] == 1
                    annotation.set_verified(verified)
                
                annotations_to_add.append(annotation)
                images_to_update.add(image_path)
                progress_bar.update_progress()
            
            # 2. Add all created annotations in a single, efficient batch operation
            if annotations_to_add:
                self.annotation_window.add_annotations(annotations_to_add)
            
            # 3. Update UI counts for each affected image only ONCE
            for image_path in images_to_update:
                self.image_window.update_image_annotations(image_path)
                        
            # Load the annotations for the currently visible image
            self.annotation_window.load_annotations()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def _extract_machine_confidence(self, row, label_cache):
        """Extract machine confidence data efficiently."""
        confidence_cols = [col for col in row.index if col.startswith('Machine confidence')]
        suggestion_cols = [col for col in row.index if col.startswith('Machine suggestion')]
        
        if not confidence_cols or not suggestion_cols:
            return {}
        
        # Vectorized approach for confidence extraction
        valid_data = [
            (row[sug], row[conf])
            for conf, sug in zip(confidence_cols, suggestion_cols)
            if pd.notna(row[conf]) and pd.notna(row[sug])
        ]
        
        if not valid_data:
            return {}
        
        # Normalize if sum > 1
        suggestions, confidences = zip(*valid_data)
        total_confidence = sum(confidences)
        if total_confidence > 1:
            confidences = [conf / 100 for conf in confidences]
        
        # Build confidence dict using cached labels
        return {
            label_cache[str(suggestion)]: confidence
            for suggestion, confidence in zip(suggestions, confidences)
        }
