import os
from datetime import datetime
from typing import Dict

import cv2
import numpy as np

import pyqtgraph as pg
from pyqtgraph import PlotWidget

from shapely.geometry import Polygon

import supervision as sv

from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QSlider, QFileDialog, 
                             QWidget, QFrame, QComboBox, QSizePolicy,
                             QMessageBox, QApplication, QScrollArea)

from coralnet_toolbox.MachineLearning.VideoInference.QtInference import TRACKING_COLORS
from coralnet_toolbox.MachineLearning.VideoInference.QtInference import InferenceEngine
from coralnet_toolbox.MachineLearning.VideoInference.QtInference import RegionZoneManager


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoDisplayWidget(QWidget):
    """Custom widget for displaying video frames and handling mouse events for region drawing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Paint the video frame and regions on this widget."""
        if not self.parent_widget:
            return
            
        painter = QPainter(self)
        
        if self.parent_widget.current_frame is not None:
            # Convert frame to QImage and QPixmap
            rgb = cv2.cvtColor(self.parent_widget.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale while maintaining aspect ratio to fit this widget
            widget_width = self.width()
            widget_height = self.height()
            scaled = pixmap.scaled(widget_width, widget_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Center the scaled image within this widget
            offset_x = (widget_width - scaled.width()) // 2
            offset_y = (widget_height - scaled.height()) // 2
            
            painter.drawPixmap(offset_x, offset_y, scaled)
            
        # Draw rectangles and polygons with TRACKING_COLORS
        if self.parent_widget.show_regions:
            for i, region in enumerate(self.parent_widget.regions):
                # Get color from TRACKING_COLORS palette, cycling through colors
                color_idx = i % len(TRACKING_COLORS.colors)
                tracking_color = TRACKING_COLORS.colors[color_idx]
                
                # Convert supervision Color to Qt color using hex string
                hex_color = tracking_color.as_hex()
                qt_color = QColor(hex_color)
                pen = QPen(qt_color, 2)
                painter.setPen(pen)
                
                if isinstance(region, dict) and region.get("type") == "polygon":
                    # Draw polygon
                    points = [self.parent_widget._map_widget_to_frame_coords(pt) for pt in region["points"]]
                    if len(points) > 2:
                        polygon_points = [self.parent_widget._map_frame_to_widget_coords(pt) for pt in points]
                        for i in range(len(polygon_points)):
                            start_pt = polygon_points[i]
                            end_pt = polygon_points[(i + 1) % len(polygon_points)]
                            painter.drawLine(start_pt, end_pt)
                else:
                    # Draw rectangle 
                    rect = region["rect"]
                    painter.drawRect(rect)
                    
        # Draw current rectangle being drawn
        if self.parent_widget.drawing and self.parent_widget.current_rect:
            pen = QPen(Qt.white, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.parent_widget.current_rect)
            
        # Draw in-progress polygon
        if self.parent_widget.drawing_polygon and self.parent_widget.current_polygon_points:
            pen = QPen(Qt.white, 2, Qt.DashLine)
            painter.setPen(pen)
            pts = self.parent_widget.current_polygon_points
            for i in range(1, len(pts)):
                painter.drawLine(pts[i - 1], pts[i])
            
    def mousePressEvent(self, event):
        """Forward mouse press events to parent widget with proper coordinates."""
        if self.parent_widget:
            self.parent_widget.mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Forward mouse move events to parent widget with proper coordinates."""
        if self.parent_widget:
            self.parent_widget.mouseMoveEvent(event)
            

class VideoPlotWidget(QWidget):
    """Widget for displaying real-time analytics plots above the video using PyQtGraph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        
        # Plot configuration
        self.max_points = 100  # Maximum points to display
        self.bg_color = "#F3F3F3"
        self.fg_color = "#111E68"
        self.line_width = 2
        
        # Data storage
        self.frame_numbers = []
        self.total_counts = []
        self.region_data = {}  # region_id -> list of counts
        self.class_data = {}   # class_name -> list of counts
        
        # Plot mode: "total", "regions", or "classes"
        self.plot_mode = "total"
        
        # Plot curves storage for efficient updates
        self.plot_curves = {}
        
        # Legend management
        self.legend_items = []  # Store legend items for separate display
        self.top_classes_limit = 20  # Show top 20 classes by default
        
        self.setup_ui()
        self.setup_plot()
        
    def setup_ui(self):
        """Setup the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a GroupBox to contain the plot widgets
        self.plot_group = QGroupBox("Detection Analytics")
        self.plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Layout for the group box contents
        plot_layout = QVBoxLayout(self.plot_group)
        plot_layout.setContentsMargins(5, 5, 5, 5)
        
        # Controls row 1
        controls_layout1 = QHBoxLayout()
        
        # Plot mode controls - moved to the far left
        controls_layout1.addWidget(QLabel("Plot Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Total Detections", "Per Region", "Per Class"])
        self.mode_combo.currentTextChanged.connect(self.change_plot_mode)
        controls_layout1.addWidget(self.mode_combo)
        
        # Add stretch to push enable/disable and clear buttons to the right
        controls_layout1.addStretch()
        
        # Enable/Disable plot buttons
        self.enable_plot_btn = QPushButton("Enable Plotting")
        self.enable_plot_btn.clicked.connect(self.enable_plotting)
        controls_layout1.addWidget(self.enable_plot_btn)
        
        self.disable_plot_btn = QPushButton("Disable Plotting")
        self.disable_plot_btn.clicked.connect(self.disable_plotting)
        self.disable_plot_btn.setEnabled(False)  # Initially disabled
        controls_layout1.addWidget(self.disable_plot_btn)
        
        # Clear button - moved to the far right after disable plotting
        clear_btn = QPushButton("Clear Plot")
        clear_btn.clicked.connect(self.clear_data)
        controls_layout1.addWidget(clear_btn)
        
        plot_layout.addLayout(controls_layout1)
        
        # Create horizontal layout for plot and legend areas
        plot_and_legend_layout = QHBoxLayout()
        
        # Create PyQtGraph PlotWidget (main plot area)
        self.plot_widget = PlotWidget()
        self.plot_widget.setMinimumHeight(150)
        self.plot_widget.setMaximumHeight(200)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create scrollable legend area with proper initialization
        self.setup_legend_scroll_area()
        
        # Add both widgets to horizontal layout
        plot_and_legend_layout.addWidget(self.plot_widget, stretch=1)  # Plot gets most space
        plot_and_legend_layout.addWidget(self.legend_scroll_area, stretch=0)  # Legend fixed width
        
        plot_layout.addLayout(plot_and_legend_layout)
        
        # Add the group box to the main layout
        main_layout.addWidget(self.plot_group)
        
        # Initialize plotting state - disabled by default
        self.plotting_enabled = False
        
    def setup_legend_scroll_area(self):
        """Setup the scrollable legend area with proper configuration."""
        self.legend_scroll_area = QScrollArea()
        
        # Set size policies and dimensions
        self.legend_scroll_area.setMinimumHeight(150)
        self.legend_scroll_area.setMaximumHeight(200)
        self.legend_scroll_area.setMinimumWidth(150)
        self.legend_scroll_area.setMaximumWidth(150)
        self.legend_scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        
        # Configure scroll behavior
        self.legend_scroll_area.setWidgetResizable(True)
        self.legend_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.legend_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create a widget to hold the legend content
        self.legend_content_widget = QWidget()
        self.legend_content_widget.setMinimumWidth(135)  # Slightly less than scroll area to avoid horizontal scrollbar
        
        # Create layout for legend items
        self.legend_content_layout = QVBoxLayout(self.legend_content_widget)
        self.legend_content_layout.setContentsMargins(5, 5, 5, 5)
        self.legend_content_layout.setSpacing(3)
        self.legend_content_layout.setAlignment(Qt.AlignTop)  # Align items to top
        
        # Set the content widget in the scroll area
        self.legend_scroll_area.setWidget(self.legend_content_widget)
        
        # Style the scroll area
        self.legend_scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.bg_color};
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {self.bg_color};
            }}
        """)
        
    def setup_plot(self):
        """Initialize the PyQtGraph plot widget."""
        # Configure main plot appearance
        self.plot_widget.setBackground(self.bg_color)
        self.plot_widget.setLabel('left', 'Detection Count', color=self.fg_color)
        self.plot_widget.setLabel('bottom', 'Frame Number', color=self.fg_color)
        self.plot_widget.setTitle('Real-time Detection Count', color=self.fg_color)
        
        # Enable grid on main plot
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Set axis colors on main plot
        axis_pen = pg.mkPen(color=self.fg_color, width=1)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        
        # Force the plot to use more of the available space
        self.plot_widget.getPlotItem().getViewBox().setDefaultPadding(0.02)
        
        # Set explicit margins to ensure grid extends across full area
        self.plot_widget.getPlotItem().setContentsMargins(10, 10, 10, 10)
        
        # Enable auto-range on main plot
        self.plot_widget.enableAutoRange()
        
        # Initialize legend with empty state
        self.create_legend_in_separate_area()
        
    def create_legend_in_separate_area(self):
        """Create legend items in the scrollable legend area using QWidget labels."""
        # Clear existing legend items more thoroughly
        while self.legend_content_layout.count():
            child = self.legend_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # If no legend items, show a placeholder
        if not self.legend_items:
            placeholder_label = QLabel("No data to display")
            placeholder_label.setStyleSheet(f"""
                color: {self.fg_color}; 
                font-size: 10px; 
                font-style: italic;
                padding: 5px;
            """)
            placeholder_label.setAlignment(Qt.AlignCenter)
            self.legend_content_layout.addWidget(placeholder_label)
            self.legend_content_layout.addStretch()
            return
        
        # Create legend items as QWidget labels with colored indicators
        for i, (name, color) in enumerate(self.legend_items):
            # Create container widget for each legend item
            legend_item_widget = QWidget()
            legend_item_widget.setFixedHeight(20)  # Fixed height for consistency
            
            legend_item_layout = QHBoxLayout(legend_item_widget)
            legend_item_layout.setContentsMargins(2, 2, 2, 2)
            legend_item_layout.setSpacing(8)
            
            # Create colored indicator (small colored square)
            color_indicator = QLabel()
            color_indicator.setFixedSize(12, 12)
            color_indicator.setStyleSheet(f"""
                background-color: {color}; 
                border: 1px solid #333333;
                border-radius: 2px;
            """)
            legend_item_layout.addWidget(color_indicator)
            
            # Create text label with truncation for long names
            display_name = name
            if len(display_name) > 18:  # Truncate long names
                display_name = display_name[:15] + "..."
                
            text_label = QLabel(display_name)
            text_label.setStyleSheet(f"""
                color: {self.fg_color}; 
                font-size: 9px;
                font-weight: normal;
            """)
            text_label.setToolTip(name)  # Show full name on hover
            text_label.setWordWrap(False)
            legend_item_layout.addWidget(text_label, stretch=1)
            
            # Add the legend item to the layout
            self.legend_content_layout.addWidget(legend_item_widget)
        
        # Add stretch at the end to push items to the top
        self.legend_content_layout.addStretch()
        
        # Force layout update
        self.legend_content_widget.updateGeometry()
        self.legend_scroll_area.updateGeometry()
        
    def change_plot_mode(self, mode_text):
        """Change between total, per-region, and per-class plotting modes."""
        if mode_text == "Total Detections":
            self.plot_mode = "total"
        elif mode_text == "Per Region":
            self.plot_mode = "regions"
        elif mode_text == "Per Class":
            self.plot_mode = "classes"
            
        self.update_plot_display()
        
    def enable_plotting(self):
        """Enable plotting functionality and update button states."""
        self.plotting_enabled = True
        self.enable_plot_btn.setEnabled(False)
        self.disable_plot_btn.setEnabled(True)
        
    def disable_plotting(self):
        """Disable plotting functionality and update button states."""
        self.plotting_enabled = False
        self.enable_plot_btn.setEnabled(True)
        self.disable_plot_btn.setEnabled(False)
        
    def get_top_classes_by_recent_activity(self, n_classes=20):
        """Get top N classes based on recent detection activity - increased default."""
        if not self.class_data or not self.frame_numbers:
            return []
            
        # Calculate recent activity (last 20% of frames or minimum 10 frames)
        recent_frames = max(10, len(self.frame_numbers) // 5)
        
        class_scores = {}
        for class_name, counts in self.class_data.items():
            if len(counts) >= recent_frames:
                # Score based on recent activity and total activity
                recent_sum = sum(counts[-recent_frames:])
                total_sum = sum(counts)
                # Weight recent activity more heavily
                class_scores[class_name] = recent_sum * 2 + total_sum
            else:
                class_scores[class_name] = sum(counts)
        
        # Sort by score and return top N
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        return [class_name for class_name, score in sorted_classes[:n_classes]]
        
    def update_data(self, frame_number: int, zone_statistics: Dict[str, Dict], total_detections: int = 0, 
                    class_counts: Dict[str, int] = None):
        """
        Update plot data with new frame information.
        
        Args:
            frame_number: Current frame number
            zone_statistics: Dictionary of zone statistics from RegionZoneManager
            total_detections: Total detections in frame when no regions are defined
            class_counts: Dictionary mapping class names to detection counts
        """
        # Only update plot data if plotting is enabled
        if not self.plotting_enabled:
            return
            
        # Add frame number
        self.frame_numbers.append(frame_number)
        
        # Calculate total count across all zones, or use total detections if no zones
        if zone_statistics:
            total_current = sum(stats.get('current_count', 0) for stats in zone_statistics.values())
        else:
            # When no regions are defined, show total detections across entire frame
            total_current = total_detections
            
        self.total_counts.append(total_current)
        
        # Update per-region data
        for zone_id, stats in zone_statistics.items():
            if zone_id not in self.region_data:
                self.region_data[zone_id] = []
            self.region_data[zone_id].append(stats.get('current_count', 0))
        
        # Update per-class data from class_counts parameter
        if class_counts:
            # Initialize new classes with zeros for all previous frames
            for class_name in class_counts.keys():
                if class_name not in self.class_data:
                    self.class_data[class_name] = [0] * (len(self.frame_numbers) - 1)
            
            # Add current frame data for all known classes
            for class_name in self.class_data.keys():
                current_count = class_counts.get(class_name, 0)
                self.class_data[class_name].append(current_count)
        else:
            # No class counts provided, pad existing classes with zeros
            for class_name in self.class_data.keys():
                self.class_data[class_name].append(0)
        
        # Ensure all region data lists have the same length
        for zone_id in list(self.region_data.keys()):
            while len(self.region_data[zone_id]) < len(self.frame_numbers):
                self.region_data[zone_id].append(0)
        
        # Trim data if it exceeds max_points
        if len(self.frame_numbers) > self.max_points:
            self.frame_numbers = self.frame_numbers[-self.max_points:]
            self.total_counts = self.total_counts[-self.max_points:]
            for zone_id in self.region_data:
                self.region_data[zone_id] = self.region_data[zone_id][-self.max_points:]
            for class_name in self.class_data:
                self.class_data[class_name] = self.class_data[class_name][-self.max_points:]
        
        # Update the plot
        self.update_plot_display()
        
    def update_plot_display(self):
        """Update the plot display with current data using PyQtGraph."""
        # Clear existing plots and legend items
        self.plot_widget.clear()
        self.plot_curves.clear()
        self.legend_items.clear()
        
        if not self.frame_numbers:
            # Empty plot - just set title and update legend
            self.plot_widget.setTitle("Real-time Detection Count", color=self.fg_color)
            self.create_legend_in_separate_area()
            return
            
        x_data = np.array(self.frame_numbers)
        
        if self.plot_mode == "total":
            # Plot total detections
            y_data = np.array(self.total_counts)
            
            # Create pen for total detections line
            pen = pg.mkPen(color="#7b0068", width=self.line_width)
            
            # Plot the line with markers
            curve = self.plot_widget.plot(
                x_data, y_data, 
                pen=pen,
                symbol='o', 
                symbolSize=4,
                symbolBrush="#7b0068"
            )
            
            self.plot_curves['total'] = curve
            self.legend_items.append(("Total Detections", "#7b0068"))
            self.plot_widget.setTitle("Total Detections Over Time", color=self.fg_color)
            
        elif self.plot_mode == "regions":
            # Plot per-region data, or total data if no regions exist
            if self.region_data:
                # We have region data - plot per region
                for i, (zone_id, counts) in enumerate(self.region_data.items()):
                    if len(counts) > 0:
                        # Use TRACKING_COLORS for consistency
                        color_idx = i % len(TRACKING_COLORS.colors)
                        color_hex = TRACKING_COLORS.colors[color_idx].as_hex()
                        
                        y_data = np.array(counts)
                        
                        # Create pen for this region
                        pen = pg.mkPen(color=color_hex, width=self.line_width)
                        
                        # Plot the line with markers
                        curve = self.plot_widget.plot(
                            x_data, y_data,
                            pen=pen,
                            symbol='o',
                            symbolSize=4,
                            symbolBrush=color_hex
                        )
                        
                        self.plot_curves[zone_id] = curve
                        self.legend_items.append((f"Region {i+1}", color_hex))
                        
                self.plot_widget.setTitle("Detections Per Region Over Time", color=self.fg_color)
            else:
                # No regions defined - show total detections like in total mode
                y_data = np.array(self.total_counts)
                
                # Create pen for total detections line
                pen = pg.mkPen(color="#7b0068", width=self.line_width)
                
                # Plot the line with markers
                curve = self.plot_widget.plot(
                    x_data, y_data, 
                    pen=pen,
                    symbol='o', 
                    symbolSize=4,
                    symbolBrush="#7b0068"
                )
                
                self.plot_curves['total'] = curve
                self.legend_items.append(("Total Detections (No Regions)", "#7b0068"))
                self.plot_widget.setTitle("Total Detections Over Time (No Regions Defined)", color=self.fg_color)
        
        elif self.plot_mode == "classes":
            # Plot per-class data
            if self.class_data:
                # Get top classes to display
                top_classes = self.get_top_classes_by_recent_activity(self.top_classes_limit)
                
                if top_classes:
                    # Plot only the top classes
                    for i, class_name in enumerate(top_classes):
                        if class_name in self.class_data and len(self.class_data[class_name]) > 0:
                            # Cycle colors for each class
                            color_idx = i % len(TRACKING_COLORS.colors)
                            color_hex = TRACKING_COLORS.colors[color_idx].as_hex()
                            
                            y_data = np.array(self.class_data[class_name])
                            
                            # Create pen for this class
                            pen = pg.mkPen(color=color_hex, width=self.line_width)
                            
                            # Plot the line with markers
                            curve = self.plot_widget.plot(
                                x_data, y_data,
                                pen=pen,
                                symbol='o',
                                symbolSize=4,
                                symbolBrush=color_hex
                            )
                            
                            self.plot_curves[class_name] = curve
                            self.legend_items.append((class_name, color_hex))
                    
                    total_classes = len(self.class_data)
                    shown_classes = len(top_classes)
                    title = f"Top {shown_classes} Classes Over Time ({total_classes} total)"
                    self.plot_widget.setTitle(title, color=self.fg_color)
                else:
                    self.plot_widget.setTitle("No Class Data Available", color=self.fg_color)
            else:
                # No class data - fallback to total detections
                y_data = np.array(self.total_counts)
                
                # Create pen for total detections line
                pen = pg.mkPen(color="#7b0068", width=self.line_width)
                
                # Plot the line with markers
                curve = self.plot_widget.plot(
                    x_data, y_data, 
                    pen=pen,
                    symbol='o', 
                    symbolSize=4,
                    symbolBrush="#7b0068"
                )
                
                self.plot_curves['total'] = curve
                self.legend_items.append(("Total Detections (No Classes)", "#7b0068"))
                self.plot_widget.setTitle("Total Detections Over Time (No Classes Defined)", color=self.fg_color)
        
        # Auto-range to fit all data
        self.plot_widget.autoRange()
        
        # Update the separate legend area
        self.create_legend_in_separate_area()
        
    def clear_data(self):
        """Clear all plot data."""
        self.frame_numbers.clear()
        self.total_counts.clear()
        self.region_data.clear()
        self.class_data.clear()
        self.plot_curves.clear()
        self.legend_items.clear()
        self.update_plot_display()
        
    def reset_for_new_video(self):
        """Reset plot data for a new video."""
        self.clear_data()
        

class VideoRegionWidget(QWidget):
    """Widget for displaying video, playback controls, and drawing/editing rectangular regions only."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent
        
        # Polygon drawing state (must be set before any UI or event setup)
        self.drawing_polygon = False
        self.current_polygon_points = []
        self.region_polygons = []
        
        # Add VideoPlotWidget above the video player
        self.plot_widget = VideoPlotWidget(self)
        
        # Inference
        self.inference_engine = InferenceEngine(self)
        self.inference_enabled = False

        # Video frame and display
        self.frame = None
        self.pixmap = None
        self.regions = []  # List of dicts: {"type": "rect"/"polygon", ...}
        self.drawing = False
        self.rect_start = None  # QPoint
        self.rect_end = None    # QPoint
        self.current_rect = None  # QRect
        self.selected_region = None
        self.setMinimumSize(640, 480)

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Region visibility
        self.show_regions = True

        # Video playback attributes
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.current_frame = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_speed = 1.0

        # Video output handling - simplified and fixed
        self.video_sink = None
        self.video_path = None
        self.output_dir = None
        self.output_path = None
        self.should_write_video = False

        self._setup_ui()
        self.disable_video_region()
        self.setFocusPolicy(Qt.StrongFocus)  # Needed for key events

    def _setup_ui(self):
        """Setup the user interface components."""
        # Main layout for the widget
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # Add plot widget above the video player
        self.layout.addWidget(self.plot_widget)
        
        # Video Player GroupBox - takes up most of the space
        self.video_group = QGroupBox("Video Player")
        self.video_layout = QVBoxLayout(self.video_group)
        self.video_layout.setContentsMargins(5, 5, 5, 5)
        
        # Video display area using custom widget
        self.video_display = VideoDisplayWidget(self)
        self.video_layout.addWidget(self.video_display)
        
        # Add video group to main layout with stretch factor
        self.layout.addWidget(self.video_group, stretch=1)
    
        # Controls container - horizontal layout for side-by-side groupboxes
        controls_container = QHBoxLayout()
        controls_container.setSpacing(10)
    
        # Media Controls GroupBox - left side
        self.controls_group = QGroupBox("Media Controls")
        self.controls_group.setMaximumHeight(150)
        self.controls_group.setMinimumHeight(150)
        controls_main_layout = QVBoxLayout(self.controls_group)
        controls_main_layout.setContentsMargins(10, 10, 10, 10)
        controls_main_layout.setSpacing(5)
    
        # Frame label and speed control row - moved to top
        info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self.frame_label)
        info_layout.addStretch()
        self.speed_dropdown = QComboBox()
        self.speed_dropdown.addItems(["0.5x", "1x", "2x"])
        self.speed_dropdown.setCurrentIndex(1)
        self.speed_dropdown.currentIndexChanged.connect(self.change_speed)
        self.speed_dropdown.setMaximumWidth(80)
        info_layout.addWidget(self.speed_dropdown)
        controls_main_layout.addLayout(info_layout)
    
        # Seek slider row - center with increased width
        seek_layout = QHBoxLayout()
        seek_layout.addSpacing(5)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek)
        self.seek_slider.setMinimumWidth(400)  # Increased width
        seek_layout.addWidget(self.seek_slider)
        seek_layout.addSpacing(5)
        controls_main_layout.addLayout(seek_layout)
    
        # Main controls row - playback buttons on left, recording buttons on right
        controls = QHBoxLayout()
        
        # Playback controls group - left side
        playback_controls = QHBoxLayout()
        playback_controls.setSpacing(2)  # Tight spacing for grouped buttons
        
        self.step_back_btn = QPushButton()
        self.step_back_btn.setIcon(self.style().standardIcon(self.style().SP_MediaSeekBackward))
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setFocusPolicy(Qt.NoFocus)
        playback_controls.addWidget(self.step_back_btn)
    
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.play_btn.setToolTip("Play")
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setFocusPolicy(Qt.NoFocus)
        playback_controls.addWidget(self.play_btn)
    
        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPause))
        self.pause_btn.setToolTip("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setFocusPolicy(Qt.NoFocus)
        playback_controls.addWidget(self.pause_btn)
    
        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(self.style().standardIcon(self.style().SP_MediaSeekForward))
        self.step_fwd_btn.setToolTip("Step Forward")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        self.step_fwd_btn.setFocusPolicy(Qt.NoFocus)
        playback_controls.addWidget(self.step_fwd_btn)
    
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_btn.setToolTip("Stop & Reset")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setFocusPolicy(Qt.NoFocus)
        playback_controls.addWidget(self.stop_btn)
    
        # Add playback controls to main controls layout
        controls.addLayout(playback_controls)
        
        # Add stretch to push recording buttons to the right
        controls.addStretch()
    
        # Recording controls group - right side
        recording_controls = QHBoxLayout()
        recording_controls.setSpacing(2)  # Tight spacing for grouped buttons
        
        self.record_play_btn = QPushButton()
        self.record_play_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.record_play_btn.setToolTip("Start Recording")
        self.record_play_btn.setFocusPolicy(Qt.NoFocus)
        self.record_play_btn.clicked.connect(self.start_recording)
        self.record_play_btn.setEnabled(False)  # Only enabled if output_dir is set
        recording_controls.addWidget(self.record_play_btn)
        
        self.record_stop_btn = QPushButton()
        self.record_stop_btn.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.record_stop_btn.setToolTip("Stop Recording")
        self.record_stop_btn.setFocusPolicy(Qt.NoFocus)
        self.record_stop_btn.clicked.connect(self.stop_recording)
        self.record_stop_btn.setEnabled(False)
        recording_controls.addWidget(self.record_stop_btn)
    
        # Add recording controls to main controls layout
        controls.addLayout(recording_controls)
    
        # Add to media controls layout
        controls_main_layout.addLayout(controls)
    
        # Set button sizes
        max_btn_size = 32
        for btn in [self.step_back_btn, 
                    self.play_btn,
                    self.pause_btn, 
                    self.step_fwd_btn, 
                    self.stop_btn, 
                    self.record_play_btn, 
                    self.record_stop_btn]:
            btn.setMaximumSize(max_btn_size, max_btn_size)
    
        # Region Controls GroupBox - right side, minimized width
        self.region_controls_group = QGroupBox("Region Controls")
        self.region_controls_group.setMaximumHeight(150)
        self.region_controls_group.setMinimumHeight(150)
        self.region_controls_group.setMaximumWidth(220)  # Reduced width
        self.region_controls_group.setMinimumWidth(220)
        region_main_layout = QVBoxLayout(self.region_controls_group)
        region_main_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        region_main_layout.setSpacing(3)  # Reduced spacing
        
        # Count criteria row - compact
        criteria_layout = QHBoxLayout()
        criteria_layout.addWidget(QLabel("Count:"))
        self.count_criteria_combo = QComboBox()
        self.count_criteria_combo.addItems(["Centroid", "Bounding Box"])
        self.count_criteria_combo.setCurrentIndex(0)
        self.count_criteria_combo.currentIndexChanged.connect(self.update_region_parameters)
        criteria_layout.addWidget(self.count_criteria_combo)
        region_main_layout.addLayout(criteria_layout)
    
        # Display outside detections row - compact
        outside_layout = QHBoxLayout()
        outside_layout.addWidget(QLabel("Show Outside:"))
        self.display_outside_combo = QComboBox()
        self.display_outside_combo.addItems(["True", "False"])
        self.display_outside_combo.setCurrentIndex(0)
        self.display_outside_combo.currentIndexChanged.connect(self.update_region_parameters)
        outside_layout.addWidget(self.display_outside_combo)
        region_main_layout.addLayout(outside_layout)
        
        # Add vertical stretch to push clear button to bottom
        region_main_layout.addStretch()
        
        # Clear regions button - moved to bottom and full width
        self.clear_regions_btn = QPushButton("Clear Regions")
        self.clear_regions_btn.setFocusPolicy(Qt.NoFocus)
        self.clear_regions_btn.clicked.connect(self.clear_regions)
        self.clear_regions_btn.setMinimumHeight(30)  # Slightly taller
        region_main_layout.addWidget(self.clear_regions_btn)
    
        # Add both group boxes to the controls container
        controls_container.addWidget(self.controls_group, stretch=1)  # Media controls get more space
        controls_container.addWidget(self.region_controls_group, stretch=0)  # Region controls fixed width
    
        # Add controls container to main layout with no stretch
        self.layout.addLayout(controls_container)
        
    def closeEvent(self, event):
        """Handle widget close event."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
        
    def browse_output(self):
        """Open directory dialog to select output directory."""
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_edit.setText(dir_name)  # AttributeError - no output_edit attribute
            self.output_dir = dir_name          # Sets output_dir on wrong object
            # If video already loaded, update output dir for widget
            if self.video_path:
                self.video_region_widget.load_video(self.video_path, dir_name)
            else:
                self.update_record_buttons()
        else:
            self.update_record_buttons()
        
    def enable_video_region(self):
        """Enable all controls in the video region widget."""
        self.setEnabled(True)
        for widget in [self.step_back_btn, 
                       self.play_btn, 
                       self.pause_btn, 
                       self.step_fwd_btn, 
                       self.stop_btn, 
                       self.seek_slider, 
                       self.speed_dropdown, 
                       self.frame_label]:
            widget.setEnabled(True)
            
        for widget in [self.record_play_btn, self.record_stop_btn]:
            # Enable record buttons only if output directory is set
            widget.setEnabled(bool(self.output_dir and os.path.exists(self.output_dir) and self.video_path))

    def disable_video_region(self):
        """Disable all controls in the video region widget."""
        self.setEnabled(False)
        for widget in [self.step_back_btn, 
                       self.play_btn, 
                       self.pause_btn, 
                       self.record_play_btn,
                       self.record_stop_btn,
                       self.step_fwd_btn, 
                       self.stop_btn, 
                       self.seek_slider, 
                       self.speed_dropdown, 
                       self.frame_label]:
            widget.setEnabled(False)
            
    def play_video(self):
        """Play the video from the current position."""
        if not self.is_playing and self.cap:
            self.is_playing = True
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)

    def pause_video(self):
        """Pause the video playback."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)

    def stop_video(self):
        """Stop the video playback and finalize output."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            
        # Finalize current video output when stopped
        self.stop_recording()
        print("Video recording stopped - ready for new recording on next play")
        
        # Reset to first frame
        if self.cap:
            self.seek(0)
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.seek_slider.setValue(0)
            self.update_frame_label()
        
        # Clear regions and reset state
        self.clear_regions()
        
        # Disable Inference from parent widget
        self.parent.disable_inference()
        
    def start_recording(self):
        """Start recording the video to output file. Also start playback if not already playing."""
        if not self.should_write_video and self.output_dir and os.path.exists(self.output_dir) and self.video_path:
            self._setup_video_output(self.video_path, self.output_dir)
        if not self.is_playing:
            self.play_video()
        self.update_record_buttons()

    def stop_recording(self):
        """Stop recording the video and finalize output."""
        self._cleanup_video_sink()
        self.update_record_buttons()
        
    def update_record_buttons(self):
        """Update the enabled state of record buttons based on output directory and recording status."""
        # Record Play button should be enabled if an output directory is set,
        # a video is loaded, and we are not currently recording.
        can_start_recording = bool(self.output_dir and os.path.exists(self.output_dir) and self.video_path)
        
        self.record_play_btn.setEnabled(can_start_recording and not self.should_write_video)
        
        # Record Stop button should be enabled only if we are currently recording.
        self.record_stop_btn.setEnabled(self.should_write_video)

    def seek(self, frame_number):
        """Seek to a specific frame in the video."""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            # Process the frame for inference if enabled
            processed_frame, total_detections, class_counts = self.process_frame_for_inference(frame.copy())
            
            self.current_frame = processed_frame
            self.current_frame_number = frame_number
            self.video_display.update()  # Update the video display widget
            self.update_frame_label()
            # Update plot with current statistics
            stats = self.get_zone_statistics()
            self.plot_widget.update_data(self.current_frame_number, stats, total_detections, class_counts)
        else:
            QMessageBox.critical(self.parent, 
                                 "Error", 
                                 f"Failed to seek to frame {frame_number}")

    def step_forward(self):
        """Step forward one frame in the video."""
        if self.cap:
            next_frame = min(self.current_frame_number + 1, self.total_frames - 1)
            self.seek(next_frame)

    def step_backward(self):
        """Step backward one frame in the video."""
        if self.cap:
            prev_frame = max(self.current_frame_number - 1, 0)
            self.seek(prev_frame)

    def change_speed(self, idx):
        """Change the playback speed based on the selected index."""
        speeds = [0.5, 1.0, 2.0]
        self.playback_speed = speeds[idx]
        if self.is_playing:
            # Restart timer with new speed
            self.timer.stop()
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def update_frame_label(self):
        """Update the frame label with current frame number and total frames."""
        self.frame_label.setText(f"Frame: {self.current_frame_number + 1} / {self.total_frames}")
        
    def enable_inference(self, enable: bool):
        """Enable or disable inference in the video region."""
        self.inference_enabled = enable

    def set_region_visibility(self, visible: bool):
        """Set the visibility of regions in the video."""
        self.show_regions = visible
        self.video_display.update()
        
    def clear_regions(self):
        """Clear all regions and reset the region polygons."""
        self.regions.clear()
        self.region_polygons.clear()  
        
        # Reset the zone manager completely and create a fresh instance
        if self.inference_engine:
            self.inference_engine.zone_manager = RegionZoneManager(
                count_criteria=self.inference_engine.count_criteria,
                display_outside=self.inference_engine.display_outside
            )
            
            # Also, reset the tracker to ensure tracking data doesn't persist
            self.inference_engine.reset_tracker()
        
        # Update the display and refresh the current frame
        self.video_display.update()
        self.seek(self.current_frame_number)
        
        # Also clear the plot data
        self.plot_widget.clear_data()
        
        # Note: Don't call self.regions_widget.clear_regions() to avoid circular reference

    def get_zone_statistics(self):
        """Get zone statistics from the inference engine."""
        if self.inference_engine:
            return self.inference_engine.get_zone_statistics()
        return {}
    
    def update_region_polygons(self):
        """Update region polygons with zone IDs for better tracking."""
        self.region_polygons = []
        
        if self.current_frame is None or not self.regions:
            return
            
        # Get video frame dimensions
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Get video display area dimensions
        widget_w = self.video_display.width()
        widget_h = self.video_display.height()
        
        # Calculate scaling and positioning
        scale = min(widget_w / frame_w, widget_h / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        offset_x = (widget_w - scaled_w) // 2
        offset_y = (widget_h - scaled_h) // 2
        
        # Create zone IDs for tracking
        zone_ids = []
        
        for i, region in enumerate(self.regions):
            zone_id = f"region_{i}"
            
            if isinstance(region, dict) and region.get("type") == "rect":
                rect = region["rect"]
                # Map QRect from video display widget coordinates to video frame coordinates
                left_scaled = rect.left() - offset_x
                top_scaled = rect.top() - offset_y
                right_scaled = rect.right() - offset_x
                bottom_scaled = rect.bottom() - offset_y
                
                # Scale back to original video frame coordinates
                left_frame = left_scaled / scale
                top_frame = top_scaled / scale
                right_frame = right_scaled / scale
                bottom_frame = bottom_scaled / scale
                
                # Clamp to frame bounds and ensure valid rectangle
                left_frame = max(0, min(left_frame, frame_w))
                right_frame = max(0, min(right_frame, frame_w))
                top_frame = max(0, min(top_frame, frame_h))
                bottom_frame = max(0, min(bottom_frame, frame_h))
                
                # Ensure we have a valid rectangle after clamping
                if left_frame < right_frame and top_frame < bottom_frame:
                    poly = Polygon([
                        (left_frame, top_frame),
                        (right_frame, top_frame),
                        (right_frame, bottom_frame),
                        (left_frame, bottom_frame)
                    ])
                    self.region_polygons.append(poly)
                    zone_ids.append(zone_id)
                    
            elif isinstance(region, dict) and region.get("type") == "polygon":
                pts = []
                for pt in region["points"]:
                    x_scaled = (pt.x() - offset_x) / scale
                    y_scaled = (pt.y() - offset_y) / scale
                    x_scaled = max(0, min(x_scaled, frame_w))
                    y_scaled = max(0, min(y_scaled, frame_h))
                    pts.append((x_scaled, y_scaled))
                if len(pts) > 2:
                    self.region_polygons.append(Polygon(pts))
                    zone_ids.append(zone_id)
        
        # Update the inference engine with zone IDs if it has the zone manager
        if hasattr(self.inference_engine, 'zone_manager'):
            self.inference_engine.zone_manager.update_regions(self.region_polygons, zone_ids)
        
    def mousePressEvent(self, event):
        """Handle mouse press events for drawing regions."""
        # Pause video playback if currently playing
        if self.is_playing:
            self.pause_video()
            
        if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
            # Polygon drawing mode
            pos = event.pos()
            if not self.drawing_polygon:
                self.drawing_polygon = True
                self.current_polygon_points = [pos, pos]  # Add first point and a preview point
            else:
                self.current_polygon_points.insert(-1, pos)  # Insert before the preview point
            # Update the video display to show the polygon in progress
            self.video_display.update()
            
        elif event.button() == Qt.LeftButton and not (event.modifiers() & Qt.ControlModifier):
            # Rectangle drawing as before
            # Get the current mouse position
            pos = event.pos()
            # Check if we're not currently drawing a rectangle
            if not self.drawing:
                # Start drawing a new rectangle
                self.drawing = True
                # Set the starting point of the rectangle
                self.rect_start = pos
                # Set the ending point to the same position initially
                self.rect_end = pos
                # Clear any current rectangle being drawn
                self.current_rect = None
            else:
                # Finish drawing the rectangle
                self.drawing = False
                # Check if we have valid start and end points that are different
                if self.rect_start and self.rect_end and self.rect_start != self.rect_end:
                    # Create a QRect from the start and end points
                    rect = self._make_rect(self.rect_start, self.rect_end)
                    # Save current state to undo stack before adding new region
                    self.undo_stack.append(list(self.regions))
                    # Clear redo stack since we're adding a new action
                    self.redo_stack.clear()
                    # Add the new rectangle region to the regions list
                    self.regions.append({"type": "rect", "rect": rect})
                    # Update the region polygons for inference calculations
                    self.update_region_polygons()
                # Reset rectangle drawing state
                self.rect_start = None
                self.rect_end = None
                self.current_rect = None
            # Update the video display to reflect changes
            self.video_display.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for updating region shape."""
        if self.drawing and self.rect_start:
            # Use raw mouse coordinates from the video display widget
            pos = event.pos()
            self.rect_end = pos
            self.current_rect = self._make_rect(self.rect_start, self.rect_end)
            self.video_display.update()
        elif self.drawing_polygon and self.current_polygon_points:
            # Update the last point of the polygon in progress
            pos = event.pos()
            self.current_polygon_points[-1] = pos
            self.video_display.update()
        
    def keyReleaseEvent(self, event):
        """Finish polygon drawing when Ctrl is released."""
        if event.key() == Qt.Key_Control and self.drawing_polygon:
            if len(self.current_polygon_points) > 2:
                self.regions.append({
                    "type": "polygon",
                    "points": list(self.current_polygon_points)
                })
                self.update_region_polygons()
            self.drawing_polygon = False
            self.current_polygon_points = []
            self.video_display.update()

    def _make_rect(self, p1, p2):
        """Return a QRect from two points."""
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        return QRect(left, top, right - left, bottom - top)
    
    def _map_widget_to_frame_coords(self, point):
        """
        Convert coordinates from widget space to frame space.
        
        Args:
            point: QPoint in widget coordinates
            
        Returns:
            QPoint in frame coordinates
        """
        if self.current_frame is None:
            return point
            
        # Get frame dimensions
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Get widget dimensions
        widget_w = self.video_display.width()
        widget_h = self.video_display.height()
        
        # Calculate scale factor
        scale = min(widget_w / frame_w, widget_h / frame_h)
        
        # Calculate offset for centered image
        offset_x = (widget_w - (frame_w * scale)) // 2
        offset_y = (widget_h - (frame_h * scale)) // 2
        
        # Convert to frame coordinates
        frame_x = (point.x() - offset_x) / scale
        frame_y = (point.y() - offset_y) / scale
        
        # Clamp to frame boundaries
        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))
        
        from PyQt5.QtCore import QPoint
        return QPoint(int(frame_x), int(frame_y))

    def _map_frame_to_widget_coords(self, point):
        """
        Convert coordinates from frame space to widget space.
        
        Args:
            point: QPoint in frame coordinates
            
        Returns:
            QPoint in widget coordinates
        """
        if self.current_frame is None:
            return point
            
        # Get frame dimensions
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Get widget dimensions
        widget_w = self.video_display.width()
        widget_h = self.video_display.height()
        
        # Calculate scale factor
        scale = min(widget_w / frame_w, widget_h / frame_h)
        
        # Calculate offset for centered image
        offset_x = (widget_w - (frame_w * scale)) // 2
        offset_y = (widget_h - (frame_h * scale)) // 2
        
        # Convert to widget coordinates
        widget_x = (point.x() * scale) + offset_x
        widget_y = (point.y() * scale) + offset_y
        
        return QPoint(int(widget_x), int(widget_y))
        
    def _get_video_offset(self):
        """Calculate the offset and scale for centering the video in the widget."""
        if self.current_frame is not None:
            frame_h, frame_w = self.current_frame.shape[:2]
            widget_width = self.video_display.width()
            widget_height = self.video_display.height()
            
            # Calculate scale to fit video while maintaining aspect ratio
            scale = min(widget_width / frame_w, widget_height / frame_h)
            scaled_w = int(frame_w * scale)
            scaled_h = int(frame_h * scale)
            
            # Calculate offset to center the scaled video
            offset_x = (widget_width - scaled_w) // 2
            offset_y = (widget_height - scaled_h) // 2
            
            return offset_x, offset_y
        return 0, 0

    def _setup_video_output(self, video_path, output_dir):
        """Setup video output with timestamp in filename."""
        try:
            video_info = sv.VideoInfo.from_video_path(video_path=video_path)
            
            # Create timestamped filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{timestamp}.mp4"
            self.output_path = os.path.join(output_dir, output_filename)
            
            # Initialize VideoSink
            self.video_sink = sv.VideoSink(target_path=self.output_path, video_info=video_info)
            self.video_sink.__enter__()
            self.should_write_video = True
            
        except Exception as e:
            print(f"Failed to setup video output: {e}")
            self.should_write_video = False
            self.video_sink = None

    def _prepare_new_video_output(self):
        """Prepare a new video output with fresh timestamp if output directory exists."""
        if self.output_dir and os.path.exists(self.output_dir) and self.video_path:
            self._setup_video_output(self.video_path, self.output_dir)

    def _write_frame_to_sink(self, frame):
        """Write a frame to the video sink if enabled."""
        if not self.should_write_video or self.video_sink is None:
            return
        
        try:
            # Ensure frame matches expected output size
            expected_shape = (self.video_sink.video_info.height, self.video_sink.video_info.width)
            if frame.shape[:2] != expected_shape:
                frame = cv2.resize(frame, (expected_shape[1], expected_shape[0]))
            self.video_sink.write_frame(frame)
        except Exception as e:
            print(f"Error writing frame to video sink: {e}")
            
    def _cleanup_video_sink(self):
        """Properly cleanup the video sink and prepare for potential new recording."""
        if self.video_sink is not None:
            try:
                self.video_sink.__exit__(None, None, None)
                print(f"Video saved to: {self.output_path}")
            except Exception as e:
                print(f"Error closing video sink: {e}")
            finally:
                self.video_sink = None
                self.should_write_video = False
                self.output_path = None
    
    def _handle_video_end(self):
        """Handle when video reaches the end."""
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
        # Finalize current video output
        self._cleanup_video_sink()
        print("Video recording ended - ready for new recording on next play")
        
    def load_video(self, video_path, output_dir=None):
        """Load a video file and prepare for playback and region drawing."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Clean up existing video capture
            if self.cap:
                self.cap.release()
                
            # Clean up existing video sink
            self._cleanup_video_sink()
            
            # Load new video
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self.parent, 
                                     "Error", 
                                     f"Failed to open video file: {video_path}")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.seek_slider.setMaximum(self.total_frames - 1)
            self.current_frame_number = 0
            self.is_playing = False
            self.video_path = video_path
            self.output_dir = output_dir
            
            # Clear regions and region polygons when loading a new video
            self.regions.clear()
            self.region_polygons.clear()
            self.update()
            
            # Do NOT setup output video here; only do so when recording is started
            self.should_write_video = False
            self.update_record_buttons()
            
            # Load first frame
            self.seek(0)
            self.update()
            self.update_frame_label()
            self.enable_video_region()
            
            # Reset for new video
            self.inference_engine.reset_tracker()
            # Reset plot for new video
            self.plot_widget.reset_for_new_video()
        except Exception as e:
            QMessageBox.critical(self.parent, 
                                 "Error", 
                                 f"Failed to load video: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        
    def next_frame(self):
        """Advance to the next frame in the video and update the display."""
        if not self.cap or not self.is_playing:
            return
            
        # Use playback speed
        step = max(1, int(self.playback_speed))
        
        # Read frames according to playback speed
        frame = None
        for _ in range(step):
            ret, frame = self.cap.read()
            if not ret:
                self._handle_video_end()
                return
                
        if frame is not None:
            processed_frame, total_detections, class_counts = self.process_frame_for_inference(frame.copy())
            # Only write if recording
            if self.should_write_video:
                self._write_frame_to_sink(processed_frame)
            
            # Update display
            self.current_frame = processed_frame
            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.update()
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_number)
            self.seek_slider.blockSignals(False)
            self.update_frame_label()
            
            # Update plot with current statistics, total detection count, and class counts
            stats = self.get_zone_statistics()
            self.plot_widget.update_data(self.current_frame_number, stats, total_detections, class_counts)
            
    def process_frame_for_inference(self, frame):
        """Process frame for inference if enabled."""
        total_detections = 0
        class_counts = {}
        
        if not self.inference_enabled or not self.inference_engine:
            return frame, total_detections, class_counts
        
        try:
            # Run inference on the current frame
            detections = self.inference_engine.infer(frame)
            
            # Get total detection count and class counts
            if detections is not None:
                total_detections = len(detections)
                
                # Extract class information from detections
                if hasattr(detections, 'data') and 'class_name' in detections.data:
                    class_names = detections.data['class_name']
                    # Count occurrences of each class
                    for class_name in class_names:
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
                elif hasattr(detections, 'class_id') and detections.class_id is not None:
                    # Fallback to class IDs if class names not available
                    for class_id in detections.class_id:
                        class_name = f"Class_{class_id}"
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
            
            # Skip region counting if no regions are defined
            if not self.region_polygons:
                # Just draw detections without region processing
                if self.inference_engine.zone_manager and detections is not None and len(detections) > 0:
                    processed_frame = self.inference_engine.zone_manager.draw_detection_annotations_only(
                        frame, detections, self.parent.get_selected_annotators()
                    )
                    return processed_frame, total_detections, class_counts
                return frame, total_detections, class_counts
            
            # Count objects in defined regions
            detections, region_counts = self.inference_engine.count_objects_in_regions(detections, self.region_polygons)
            # Draw detections on the frame
            frame = self.draw_inference_results(frame, detections, region_counts)
            
        except Exception as e:
            print(f"Inference processing failed: {e}")
            
        return frame, total_detections, class_counts

    def draw_inference_results(self, frame, detections, region_counts):
        """Draw inference results on the video frame using supervision annotators."""
        if self.inference_engine.zone_manager:
            return self.inference_engine.zone_manager.annotate_detections_and_zones(
                frame, detections, self.parent.get_selected_annotators()
            )
        return frame
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()

    def update_region_parameters(self):
        """Update region parameters based on the selected count criteria and display outside detections."""
        if not self.inference_engine:
            return
            
        count_criteria = self.count_criteria_combo.currentText()
        display_outside = self.display_outside_combo.currentText() == "True"
        
        # Update the inference engine with the selected criteria
        if (hasattr(self.inference_engine, 'set_count_criteria') and 
            hasattr(self.inference_engine, 'set_display_outside_detections')):
            
            self.inference_engine.set_count_criteria(count_criteria)
            self.inference_engine.set_display_outside_detections(display_outside)
            
            # Refresh the current frame to apply new parameters immediately
            if hasattr(self, 'current_frame_number'):
                self.seek(self.current_frame_number)