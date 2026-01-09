import warnings

import os

import rasterio
from rasterio.transform import from_bounds

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QMessageBox, QGroupBox, QFormLayout, 
                             QLineEdit)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZExportDialog(QDialog):
    """
    Z-Channel Export Dialog.
    
    Allows users to export z-channel data from rasters to compressed TIFF files.
    Preserves z_data_type, z_unit, z_nodata, and georeferencing information in metadata.
    
    Input: list of Raster objects with z_channel data
    Output: Compressed TIFF files with metadata
    """
    export_completed = pyqtSignal(int)  # Number of files exported

    def __init__(self, rasters, parent=None):
        """
        Initialize the Z-Channel Export Dialog.
        
        Args:
            rasters (list): List of Raster objects to export z-channels from
            parent: Parent widget
        """
        super().__init__(parent)
        self.setModal(True)
        
        self.rasters = rasters
        
        # Output path attributes
        self.output_directory = None
        self.output_folder_name = "z_channels"
        self.naming_protocol = "{output_folder}/{file_name}.tif"
        
        # Filter rasters to only those with z_channel data
        self.exportable_rasters = [r for r in rasters if r.z_channel is not None]
        
        if not self.exportable_rasters:
            QMessageBox.warning(
                self,
                "No Z-Channels Available",
                "None of the selected images have z-channel data to export.",
                QMessageBox.Ok
            )
            self.close()
            return
        
        self.setWindowTitle("Export Z-Channels")
        self.setWindowIcon(get_icon("z.png"))
        self.resize(600, 300)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        
        # Setup UI sections
        self.setup_info_layout()
        self.setup_settings_layout()
        self.setup_buttons_layout()
    
    def setup_info_layout(self):
        """Set up the information section."""
        info_group = QGroupBox("Export Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 5, 10, 5)
        
        info_text = (
            "Z-channels will be exported as compressed TIFF files "
            "preserving float32 data type and metadata including:<br>"
            "• Z-channel data type (depth/elevation)<br>"
            "• Z-channel units (m, cm, ft, etc.)<br>"
            "• Nodata values<br>"
            "• Georeferencing information (if available)"
        )
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        info_group.setMaximumHeight(150)
        self.main_layout.addWidget(info_group)
    
    def setup_settings_layout(self):
        """Set up the settings section."""
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout()
        
        # Output directory selection
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.output_dir_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_directory)
        dir_layout.addWidget(browse_btn)
        
        settings_layout.addRow("Output Directory:", dir_layout)
        
        # Output folder name
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Enter folder name for z-channels...")
        self.output_folder_edit.setText("z_channels")
        self.output_folder_edit.setToolTip(
            "Name of the folder to create inside the output directory.\n"
            "All z-channel files will be saved in this folder."
        )
        settings_layout.addRow("Output Folder Name:", self.output_folder_edit)
        
        # Naming protocol
        self.naming_protocol_edit = QLineEdit()
        self.naming_protocol_edit.setText(self.naming_protocol)
        self.naming_protocol_edit.setToolTip(
            "Available variables:\n"
            "{output_folder} - Output folder path (directory + folder name)\n"
            "{file_name} - Original image filename without extension\n"
            "{file_name_full} - Original image filename with extension\n"
            "{ext} - File extension (always .tif for TIFF export)"
        )
        settings_layout.addRow("Naming Protocol:", self.naming_protocol_edit)
        
        settings_group.setLayout(settings_layout)
        self.main_layout.addWidget(settings_group)
    
    def setup_buttons_layout(self):
        """Set up the bottom buttons section."""
        btn_layout = QHBoxLayout()

        # Add status label on the left
        self.status_label = QLabel()
        
        count = len(self.exportable_rasters)
        if count == 0:
            self.status_label.setText("No images highlighted")
        elif count == 1:
            self.status_label.setText("1 image highlighted")
        else:
            self.status_label.setText(f"{count} images highlighted")
        
        btn_layout.addWidget(self.status_label)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(cancel_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.start_export)
        btn_layout.addWidget(export_btn)
        
        self.main_layout.addLayout(btn_layout)
    
    def browse_output_directory(self):
        """Open a dialog to select the output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Z-Channels",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.output_directory = directory
            self.output_dir_edit.setText(directory)
    
    def start_export(self):
        """Start the z-channel export process."""
        # Validate output directory
        if not self.output_directory or not os.path.exists(self.output_directory):
            QMessageBox.warning(
                self,
                "No Output Directory",
                "Please select an output directory before exporting.",
                QMessageBox.Ok
            )
            return
        
        # Get and validate output folder name
        self.output_folder_name = self.output_folder_edit.text().strip()
        if not self.output_folder_name:
            QMessageBox.warning(
                self,
                "No Folder Name",
                "Please provide a folder name for the z-channel exports.",
                QMessageBox.Ok
            )
            return
        
        # Get and validate naming protocol
        self.naming_protocol = self.naming_protocol_edit.text().strip()
        if not self.naming_protocol:
            QMessageBox.warning(
                self,
                "Invalid Naming Protocol",
                "Please provide a valid naming protocol.",
                QMessageBox.Ok
            )
            return
        
        # Build full output path
        full_output_path = os.path.join(self.output_directory, self.output_folder_name)
        
        # Confirm export
        reply = QMessageBox.question(
            self,
            "Confirm Export",
            f"Export {len(self.exportable_rasters)} z-channel file(s) to:\n{full_output_path}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return
        
        # Perform export
        self.export_z_channels()
    
    def export_z_channels(self):
        """Export z-channels for all exportable rasters."""
        # Create output folder
        output_folder_path = os.path.join(self.output_directory, self.output_folder_name)
        try:
            os.makedirs(output_folder_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Creating Folder",
                f"Failed to create output folder:\n{output_folder_path}\n\nError: {str(e)}",
                QMessageBox.Ok
            )
            return
        
        # Create progress bar
        progress_bar = ProgressBar(
            self,
            title="Exporting Z-Channels",
            text_label="Exporting z-channels..."
        )
        progress_bar.show()
        progress_bar.start_progress(len(self.exportable_rasters))
        
        exported_count = 0
        
        try:
            for raster in self.exportable_rasters:
                try:
                    # Generate output filename
                    output_path = self.generate_output_path(raster)
                    
                    # Export the z-channel
                    success = self.export_single_z_channel(raster, output_path)
                    
                    if success:
                        exported_count += 1
                    
                except Exception as e:
                    print(f"Error exporting z-channel for {raster.basename}: {str(e)}")
                
                # Update progress
                progress_bar.update_progress()
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Completed",
                f"Successfully exported {exported_count} z-channel file(s) to:\n{output_folder_path}",
                QMessageBox.Ok
            )
            
        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()
        
        # Emit completion signal and close
        self.export_completed.emit(exported_count)
        self.close()
    
    def generate_output_path(self, raster):
        """
        Generate the output file path based on naming protocol.
        
        Args:
            raster: Raster object
            
        Returns:
            str: Full output file path
        """
        # Get filename components
        file_name = os.path.splitext(raster.basename)[0]
        file_name_full = raster.basename
        
        # Build full output folder path
        output_folder_path = os.path.join(self.output_directory, self.output_folder_name)
        
        # Replace variables in naming protocol
        output_path = self.naming_protocol.replace("{output_folder}", output_folder_path)
        output_path = output_path.replace("{file_name}", file_name)
        output_path = output_path.replace("{file_name_full}", file_name_full)
        output_path = output_path.replace("{ext}", "tif")
        
        # Ensure .tif extension
        if not output_path.lower().endswith('.tif'):
            output_path += '.tif'
        
        return output_path
    
    def export_single_z_channel(self, raster, output_path):
        """
        Export a single z-channel to a compressed TIFF file.
        
        Args:
            raster: Raster object with z_channel data
            output_path (str): Path to output TIFF file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Get z-channel data and apply transform
            z_data = raster.z_channel
            
            if z_data is not None:
                scalar = raster.z_settings.get('scalar', 1.0)
                offset = raster.z_settings.get('offset', 0.0)
                direction = raster.z_settings.get('direction', 1)
                z_data = (z_data * scalar * direction) + offset
            
            if z_data is None:
                print(f"No z-channel data available for {raster.basename}")
                return False
            
            # Prepare metadata for TIFF tags
            metadata = {}
            
            if raster.z_data_type:
                metadata['z_data_type'] = raster.z_data_type
            
            if raster.z_unit:
                metadata['z_unit'] = raster.z_unit
            
            if raster.z_nodata is not None:
                metadata['z_nodata'] = str(raster.z_nodata)
            
            if raster.z_settings:
                metadata['z_settings'] = str(raster.z_settings)
            
            # Get or create transform for georeferencing
            if hasattr(raster, 'rasterio_src') and raster.rasterio_src is not None:
                try:
                    # Use existing transform if available
                    transform = raster.rasterio_src.transform
                    crs = raster.rasterio_src.crs
                except:
                    transform = None
                    crs = None
            else:
                transform = None
                crs = None
            
            # If no transform available, create a simple one
            if transform is None:
                transform = from_bounds(0, 0, 
                                        z_data.shape[1], z_data.shape[0], 
                                        z_data.shape[1], z_data.shape[0])
            
            # Prepare rasterio profile for TIFF export
            profile = {
                'driver': 'GTiff',
                'dtype': z_data.dtype,  # Preserve original dtype (float32 or uint8)
                'width': z_data.shape[1],
                'height': z_data.shape[0],
                'count': 1,  # Single band
                'compress': 'deflate',  # DEFLATE compression
                'tiled': True,  # Use tiled format for better compression
                'blockxsize': 256,
                'blockysize': 256,
                'transform': transform
            }
            
            # Add CRS if available
            if crs is not None:
                profile['crs'] = crs
            
            # Add nodata value if available
            if raster.z_nodata is not None:
                profile['nodata'] = raster.z_nodata
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Write TIFF file with metadata
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Write z-channel data
                dst.write(z_data, 1)
                
                # Write metadata as TIFF tags
                dst.update_tags(**metadata)
                
                # Write description
                description = f"Transformed Z-Channel data"
                if raster.z_data_type:
                    description += f" ({raster.z_data_type})"
                if raster.z_unit:
                    description += f" in {raster.z_unit}"
                
                dst.update_tags(1, DESCRIPTION=description)
            
            print(f"Exported z-channel for {raster.basename} to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting z-channel for {raster.basename}: {str(e)}")
            return False
