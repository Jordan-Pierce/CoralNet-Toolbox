import gc
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

from shapely.geometry import Polygon

import torch

import supervision as sv
from ultralytics import YOLO

from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, 
                             QWidget, QListWidget, QListWidgetItem, QFrame,
                             QAbstractItemView, QFormLayout, QComboBox, QSizePolicy,
                             QMessageBox, QApplication)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Add color palette for consistent tracking colors
TRACKING_COLORS = sv.ColorPalette.from_hex(["#E6194B", 
                                            "#3CB44B", 
                                            "#FFE119", 
                                            "#3C76D1", 
                                            "#FF6347",
                                            "#4169E1", 
                                            "#32CD32", 
                                            "#FFD700"])


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CustomPolygonZone(sv.PolygonZone):
    """Custom polygon zone that handles different counting criteria and provides extensibility."""
    
    def __init__(self, polygon: np.ndarray, count_criteria: str = "Centroid", zone_id: Optional[str] = None):
        """
        Initialize custom polygon zone.
        
        Args:
            polygon: Numpy array of polygon coordinates
            count_criteria: How to count objects ("Centroid", "Bounding Box", "Bottom Center")
            zone_id: Optional identifier for this zone
        """
        # Map count criteria to supervision anchors
        anchor_mapping = {
            "Centroid": [sv.Position.CENTER],
            "Bounding Box": [sv.Position.TOP_LEFT, 
                             sv.Position.TOP_RIGHT, 
                             sv.Position.BOTTOM_LEFT, 
                             sv.Position.BOTTOM_RIGHT],
            "Bottom Center": [sv.Position.BOTTOM_CENTER]
        }
        
        triggering_anchors = anchor_mapping.get(count_criteria, [sv.Position.CENTER])
        super().__init__(polygon=polygon, triggering_anchors=triggering_anchors)
        
        self.count_criteria = count_criteria
        self.zone_id = zone_id or f"zone_{id(self)}"
        self.detection_history = []  # For future tracking features
        self.cumulative_count = 0  # Track total detections that have passed through
        self.seen_tracker_ids = set()  # Track unique tracker IDs for cumulative counting
        
        # NEW: Entry/Exit tracking
        self.object_states = {}  # tracker_id -> bool (inside state)
        self.entry_count = 0
        self.exit_count = 0
        self.net_flow = 0  # entry_count - exit_count
        
    def trigger_with_metadata(self, detections: sv.Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced trigger that returns both boolean mask and metadata.
        Also tracks entry/exit events.
        
        Returns:
            tuple: (boolean_mask, detection_indices_in_zone)
        """
        is_inside = self.trigger(detections)
        detection_indices = np.where(is_inside)[0] if len(is_inside) > 0 else np.array([])
        
        # Track entry/exit events if we have tracker IDs
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            self._update_entry_exit_tracking(detections, is_inside)
            
            # Update cumulative count based on unique tracker IDs (existing logic)
            inside_tracker_ids = detections.tracker_id[is_inside]
            valid_tracker_ids = [tid for tid in inside_tracker_ids if tid is not None]
            
            for tid in valid_tracker_ids:
                if tid not in self.seen_tracker_ids:
                    self.seen_tracker_ids.add(tid)
                    self.cumulative_count += 1
    
        # Store detection history for future features
        self.detection_history.append({
            'count': self.current_count,  # Current detections in zone right now
            'cumulative_count': self.cumulative_count,  # Total unique detections seen over time
            'entry_count': self.entry_count,  # NEW: Total entries
            'exit_count': self.exit_count,   # NEW: Total exits
            'net_flow': self.net_flow,       # NEW: Net flow (entries - exits)
            'detection_indices': detection_indices.tolist()
        })
        
        # Keep only last 250 entries to prevent memory issues
        if len(self.detection_history) > 250:
            self.detection_history = self.detection_history[-250:]
        
        return is_inside, detection_indices
    
    def _update_entry_exit_tracking(self, detections: sv.Detections, is_inside: np.ndarray):
        """
        Update entry/exit tracking based on current detections and their zone status.
        
        Args:
            detections: Current detections
            is_inside: Boolean array indicating which detections are inside the zone
        """
        current_frame_tracker_ids = set()
        
        # Process each detection with a valid tracker ID
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
                
            current_frame_tracker_ids.add(tracker_id)
            currently_inside = bool(is_inside[i]) if i < len(is_inside) else False
            
            if tracker_id in self.object_states:
                # Object was seen before - check for state change
                was_inside = self.object_states[tracker_id]
                
                # Detect entry: was outside, now inside
                if not was_inside and currently_inside:
                    self.entry_count += 1
                
                # Detect exit: was inside, now outside
                elif was_inside and not currently_inside:
                    self.exit_count += 1
                
                # Update state
                self.object_states[tracker_id] = currently_inside
            else:
                # First time seeing this object
                self.object_states[tracker_id] = currently_inside
                
                # If object first appears inside the zone, count as entry
                # (This handles cases where tracking starts mid-scene)
                if currently_inside:
                    self.entry_count += 1
        
        # Clean up object states for tracker IDs not seen in current frame
        # Only remove if they haven't been seen for multiple frames to avoid flicker
        # For now, we'll keep all states to maintain tracking continuity
        
        # Update net flow
        self.net_flow = self.entry_count - self.exit_count
    
    def get_entry_exit_stats(self) -> Dict:
        """Get entry/exit statistics for this zone."""
        return {
            "zone_id": self.zone_id,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "net_flow": self.net_flow,
            "current_count": self.current_count,
            "cumulative_count": self.cumulative_count,
            "active_objects": len(self.object_states),
            "objects_inside": sum(1 for inside in self.object_states.values() if inside)
        }
    
    def reset_entry_exit_counters(self):
        """Reset entry/exit counters while preserving current object states."""
        self.entry_count = 0
        self.exit_count = 0
        self.net_flow = 0
        # Note: We keep object_states to maintain tracking continuity


class CustomPolygonZoneAnnotator(sv.PolygonZoneAnnotator):
    """Custom annotator that adds zone ID and enhanced labeling to PolygonZoneAnnotator."""
    
    def __init__(
        self,
        zone: CustomPolygonZone,
        color: sv.Color = sv.Color.RED,
        thickness: int = 2,
        text_color: sv.Color = sv.Color.BLACK,
        text_scale: float = 0.7,
        text_thickness: int = 2,
        text_padding: int = 10,
        show_current_count: bool = True,
        opacity: float = 0,
        show_zone_id: bool = True,
        show_cumulative_count: bool = True,
        show_entry_exit: bool = True,  # NEW: Show entry/exit counts
        text_position: str = "top_left"
    ):
        """
        Initialize custom polygon zone annotator.
        
        Args:
            zone: The CustomPolygonZone to annotate
            color: Zone border color
            thickness: Zone border thickness
            text_color: Text color
            text_scale: Text scale
            text_thickness: Text thickness
            text_padding: Text padding
            show_current_count: Whether to show the current count
            opacity: Zone fill opacity
            show_zone_id: Whether to show zone ID
            show_cumulative_count: Whether to show cumulative count of all detections
            show_entry_exit: Whether to show entry/exit counts
            text_position: Position for text ("top_left", "center", "top_center")
        """
        super().__init__(
            zone=zone,
            color=color,
            thickness=thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            opacity=opacity
        )
        self.custom_zone = zone
        self.show_zone_id = show_zone_id
        self.show_current_count = show_current_count
        self.show_cumulative_count = show_cumulative_count
        self.show_entry_exit = show_entry_exit  # NEW
        self.text_position = text_position
    
    def annotate(self, scene: np.ndarray, zone_metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Annotate the scene with enhanced zone information using multi-line compact display.
        
        Args:
            scene: The image to annotate
            zone_metadata: Optional metadata about the zone
            
        Returns:
            Annotated scene
        """        
        # First draw the zone using parent method (without label)
        annotated_scene = super().annotate(scene=scene, label=None)
        
        # Build multi-line compact label (without zone ID)
        label_lines = []

        # Line 1: Current and Cumulative counts
        line1_parts = []
        if self.show_current_count:
            line1_parts.append(f"Cur:{self.custom_zone.current_count}")
        if self.show_cumulative_count:
            line1_parts.append(f"Cum:{self.custom_zone.cumulative_count}")
        if line1_parts:
            label_lines.append(" | ".join(line1_parts))

        # Line 2: Entry/Exit/Net flow
        if self.show_entry_exit:
            net_sign = "+" if self.custom_zone.net_flow >= 0 else ""
            label_lines.append(f"In:{self.custom_zone.entry_count} | Out:{self.custom_zone.exit_count} | Net:{net_sign}{self.custom_zone.net_flow}")

        if not label_lines:
            return annotated_scene
        
        # Get polygon bounds for text positioning
        polygon_points = self.custom_zone.polygon
        min_x = int(np.min(polygon_points[:, 0]))
        min_y = int(np.min(polygon_points[:, 1]))
        max_x = int(np.max(polygon_points[:, 0]))
        max_y = int(np.max(polygon_points[:, 1]))
        
        # Calculate starting position based on text_position
        if self.text_position == "top_left":
            start_x = min_x + self.text_padding
            start_y = min_y + 25
        elif self.text_position == "top_center":
            start_x = (min_x + max_x) // 2
            start_y = min_y + 25
        else:  # center
            start_x = (min_x + max_x) // 2
            start_y = (min_y + max_y) // 2
        
        # Calculate dimensions for multi-line text background
        line_heights = []
        line_widths = []
        max_width = 0
        
        for line in label_lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            line_widths.append(text_width)
            line_heights.append(text_height + baseline)
            max_width = max(max_width, text_width)
        
        total_height = sum(line_heights) + (len(label_lines) - 1) * 5  # 5px spacing between lines
        
        # Calculate background rectangle bounds
        if self.text_position == "top_left":
            rect_x1 = start_x - self.text_padding
            rect_y1 = start_y - line_heights[0] - self.text_padding
            rect_x2 = start_x + max_width + self.text_padding
            rect_y2 = start_y + total_height - line_heights[0] + self.text_padding
        elif self.text_position == "top_center":
            rect_x1 = start_x - max_width // 2 - self.text_padding
            rect_y1 = start_y - line_heights[0] - self.text_padding
            rect_x2 = start_x + max_width // 2 + self.text_padding
            rect_y2 = start_y + total_height - line_heights[0] + self.text_padding
        else:  # center
            rect_x1 = start_x - max_width // 2 - self.text_padding
            rect_y1 = start_y - total_height // 2 - self.text_padding
            rect_x2 = start_x + max_width // 2 + self.text_padding
            rect_y2 = start_y + total_height // 2 + self.text_padding
        
        # Draw semi-transparent background using the zone's color
        overlay = annotated_scene.copy()
        # Use the zone's color for the background with slight brightening to improve readability
        bg_color = self.color.as_bgr()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        # Apply transparency (0.7 = 70% original image, 0.3 = 30% overlay)
        annotated_scene = cv2.addWeighted(annotated_scene, 0.7, overlay, 0.3, 0)
        
        # Draw each line of text
        current_y = start_y
        if self.text_position == "center":
            current_y = start_y - total_height // 2 + line_heights[0]
        
        for i, line in enumerate(label_lines):
            # Calculate x position for each line based on text_position
            if self.text_position == "top_center":
                line_x = start_x - line_widths[i] // 2
            elif self.text_position == "center":
                line_x = start_x - line_widths[i] // 2
            else:  # top_left
                line_x = start_x
            
            cv2.putText(
                annotated_scene,
                line,
                (line_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness
            )
            
            # Move to next line
            if i < len(label_lines) - 1:
                current_y += line_heights[i] + 5  # 5px spacing between lines
        
        return annotated_scene


class RegionZoneManager:
    """Manages multiple custom polygon zones for region counting with advanced features."""
    
    def __init__(self, count_criteria: str = "Centroid", display_outside: bool = True):
        self.zones: List[CustomPolygonZone] = []
        self.zone_annotators: List[CustomPolygonZoneAnnotator] = []
        self.count_criteria = count_criteria
        self.display_outside = display_outside
        self.zone_metadata: Dict[str, Dict] = {}  # Store additional zone information
                
        # Tracking settings - will be updated based on annotator selection
        self.show_tracking_traces = False
        self.show_tracking_boxes = False
        self.show_tracking_labels = False
        self.color_lookup = sv.ColorLookup.CLASS  # Default color lookup
        
        # Initialize trace annotator as instance variable to maintain state
        self.trace_annotator = sv.TraceAnnotator(
            color_lookup=self.color_lookup,
            thickness=2,
            trace_length=30,
            position=sv.Position.CENTER
        )

    def update_tracking_settings(self, selected_annotators: List[str]):
        """
        Update tracking settings based on selected annotators.
        
        Args:
            selected_annotators: List of selected annotator names
        """
        has_tracker = "TrackerAnnotator" in selected_annotators
        self.show_tracking_traces = has_tracker
        self.show_tracking_boxes = has_tracker
        self.show_tracking_labels = has_tracker
        # Always use CLASS color lookup for consistent coloring
        self.color_lookup = sv.ColorLookup.CLASS

    def update_regions(self, region_polygons: List[Polygon], zone_ids: Optional[List[str]] = None):
        """
        Update zones based on new region polygons.
        
        Args:
            region_polygons: List of shapely Polygon objects
            zone_ids: Optional list of zone identifiers
        """
        # Build a lookup for existing zones by zone_id
        existing_zones = {zone.zone_id: zone for zone in self.zones}

        self.zones = []
        self.zone_annotators = []
        self.zone_metadata = {}
        zone_ids = zone_ids or [f"region_{i}" for i in range(len(region_polygons))]
        
        for i, poly in enumerate(region_polygons):
            coords = np.array(poly.exterior.coords[:-1], dtype=np.int32)
            zid = zone_ids[i]

            # Try to reuse an existing zone if zone_id matches
            if zid in existing_zones:
                zone = existing_zones[zid]
                # Optionally update polygon if it changed
                zone.polygon = coords
                zone.count_criteria = self.count_criteria
            else:
                zone = CustomPolygonZone(
                    polygon=coords,
                    count_criteria=self.count_criteria,
                    zone_id=zid
                )

            color_idx = i % len(TRACKING_COLORS.colors)
            zone_color = TRACKING_COLORS.colors[color_idx]

            annotator = CustomPolygonZoneAnnotator(
                zone=zone,
                color=zone_color,
                thickness=1,
                text_scale=0.4,
                opacity=0.1,
                text_position="top_left"
            )

            self.zones.append(zone)
            self.zone_annotators.append(annotator)
            self.zone_metadata[zid] = {}

    def count_detections(self, detections: sv.Detections) -> Tuple[sv.Detections, List[int], Dict[str, List[int]]]:
        """
        Count detections in each zone and return filtered detections.
        
        Args:
            detections: Input detections to count
            
        Returns:
            tuple: (filtered_detections, region_counts, zone_detection_map)
        """
        if len(self.zones) == 0:
            return detections, [], {}
        
        region_counts = []
        zone_detection_map = {}
        filtered_detection_indices = set()
        
        for i, zone in enumerate(self.zones):
            is_inside, detection_indices = zone.trigger_with_metadata(detections)
            count = np.sum(is_inside)
            region_counts.append(count)
            
            # Update zone metadata
            zone_id = zone.zone_id
            # Remove max_simultaneous update
            
            # Store detection mapping for this zone
            zone_detection_map[zone_id] = detection_indices.tolist()
            
            # Add indices to filtered set based on display_outside setting
            if self.display_outside:
                # If showing detections outside zones, add all detections
                filtered_detection_indices.update(range(len(detections)))
            else:
                filtered_detection_indices.update(detection_indices)
        
        # Filter detections based on display_outside setting
        if self.display_outside:
            # Show all detections
            filtered_detections = detections
        else:
            # Show only detections inside zones
            if filtered_detection_indices:
                mask = np.zeros(len(detections), dtype=bool)
                mask[list(filtered_detection_indices)] = True
                filtered_detections = detections[mask]
            else:
                # No detections in any zone
                filtered_detections = detections[np.array([], dtype=bool)]
        
        return filtered_detections, region_counts, zone_detection_map
    
    def annotate_zones(self, scene: np.ndarray, detections: Optional[sv.Detections] = None) -> np.ndarray:
        """
        Annotate all zones on the scene using their respective annotators.
        
        Args:
            scene: The image to annotate
            detections: Optional detections for tracking visualization
            
        Returns:
            Annotated scene with all zones drawn
        """
        annotated_scene = scene.copy()
        
        # Draw zone annotations first
        for i, annotator in enumerate(self.zone_annotators):
            zone_id = self.zones[i].zone_id
            metadata = self.zone_metadata.get(zone_id, {})
            annotated_scene = annotator.annotate(annotated_scene, metadata)
        
        return annotated_scene
    
    def annotate_detections_and_zones(self, scene: np.ndarray, detections: Optional[sv.Detections] = None, 
                                      selected_annotators: List[str] = None) -> np.ndarray:
        """
        Annotate both detections and zones on the scene.
        
        Args:
            scene: The image to annotate
            detections: Optional detections to annotate
            selected_annotators: List of annotator names to use for detections
            
        Returns:
            Annotated scene with detections and zones
        """
        annotated_scene = scene.copy()
        
        # Update tracking settings based on selected annotators
        if selected_annotators:
            self.update_tracking_settings(selected_annotators)
        
        # Draw detection annotations if provided
        if detections is not None and len(detections) > 0 and selected_annotators:
            annotated_scene = self._draw_detection_annotations(annotated_scene, detections, selected_annotators)
        
        # Draw zones
        annotated_scene = self.annotate_zones(annotated_scene, detections)
        
        return annotated_scene
    
    def draw_detection_annotations_only(self, frame: np.ndarray, detections: sv.Detections, 
                                       selected_annotators: List[str]) -> np.ndarray:
        """
        Draw only detection annotations without zones.
        
        Args:
            frame: The image to annotate
            detections: Detections to annotate
            selected_annotators: List of annotator names to use
            
        Returns:
            Annotated frame with only detections
        """
        if detections is None or len(detections) == 0:
            return frame
        
        # Update tracking settings based on selected annotators
        self.update_tracking_settings(selected_annotators)
            
        return self._draw_detection_annotations(frame, detections, selected_annotators)
    
    def _draw_detection_annotations(self, frame: np.ndarray, detections: sv.Detections, 
                                    selected_annotators: List[str]) -> np.ndarray:
        """Draw detection annotations using selected annotators."""
        # Update tracking settings
        self.update_tracking_settings(selected_annotators)

        # Validate detections have required attributes
        if detections is None or len(detections) == 0:
            return frame

        tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') else None
        # --- REGION-BASED COLORING LOGIC ---
        # If tracking/tracing is enabled and regions are present, color by region
        if self.show_tracking_traces and self.zones and tracker_ids is not None and len(tracker_ids) > 0:
            # Assign each detection to a region index based on which zone it is in
            region_indices = np.full(len(detections), -1, dtype=int)
            for region_idx, zone in enumerate(self.zones):
                is_inside, _ = zone.trigger_with_metadata(detections)
                region_indices[is_inside] = region_idx
            detections.class_id = region_indices
        # Otherwise, keep class_id as is (by class)

        # Get the class names from detections - handle missing class_name gracefully
        class_names = detections.data.get('class_name', ['object'] * len(detections))
        confidences = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') else None

        # Create labels with or without tracking IDs based on tracking settings
        if tracker_ids is not None and self.show_tracking_labels and len(tracker_ids) > 0:
            labels = []
            for i, (name, conf) in enumerate(zip(class_names, confidences)):
                tid = tracker_ids[i] if i < len(tracker_ids) else None
                if tid is not None:
                    labels.append(f"#{int(tid)} {name}: {conf:.2f}")
                else:
                    labels.append(f"{name}: {conf:.2f}")
        else:
            labels = [f"{name}: {conf:.2f}" for name, conf in zip(class_names, confidences)]

        # Create annotators based on selection (excluding TrackerAnnotator from actual annotation)
        actual_annotators = [ann for ann in selected_annotators if ann != "TrackerAnnotator"]
        annotators = self._create_annotators(actual_annotators)

        # Apply tracking traces if enabled and we have valid tracking data
        if (
            self.show_tracking_traces
            and tracker_ids is not None
            and len(tracker_ids) > 0
        ):
            valid_tracker_mask = np.array([tid is not None for tid in tracker_ids])
            if np.any(valid_tracker_mask):
                valid_detections = detections[valid_tracker_mask]
                frame = self.trace_annotator.annotate(frame, valid_detections)

        for annotator in annotators:
            try:
                if isinstance(annotator, sv.LabelAnnotator):
                    frame = annotator.annotate(scene=frame, detections=detections, labels=labels)
                else:
                    frame = annotator.annotate(scene=frame, detections=detections)
            except Exception as e:
                print(f"Warning: Annotator {type(annotator).__name__} failed: {e}")
                continue

        return frame
    
    def _create_annotators(self, selected_annotators: List[str]) -> List:
        """Create supervision annotators based on selection."""
        annotators = []
        for key in selected_annotators:
            try:
                if key == "BoxAnnotator":
                    annotators.append(sv.BoxAnnotator(color_lookup=self.color_lookup))
                elif key == "RoundBoxAnnotator":
                    annotators.append(sv.RoundBoxAnnotator(color_lookup=self.color_lookup))
                elif key == "BoxCornerAnnotator":
                    annotators.append(sv.BoxCornerAnnotator(color_lookup=self.color_lookup))
                elif key == "ColorAnnotator":
                    annotators.append(sv.ColorAnnotator(color_lookup=self.color_lookup))
                elif key == "CircleAnnotator":
                    annotators.append(sv.CircleAnnotator(color_lookup=self.color_lookup))
                elif key == "DotAnnotator":
                    annotators.append(sv.DotAnnotator(color_lookup=self.color_lookup))
                elif key == "TriangleAnnotator":
                    annotators.append(sv.TriangleAnnotator(color_lookup=self.color_lookup))
                elif key == "EllipseAnnotator":
                    annotators.append(sv.EllipseAnnotator(color_lookup=self.color_lookup))
                elif key == "HaloAnnotator":
                    annotators.append(sv.HaloAnnotator(color_lookup=self.color_lookup))
                elif key == "PercentageBarAnnotator":
                    annotators.append(sv.PercentageBarAnnotator())
                elif key == "MaskAnnotator":
                    annotators.append(sv.MaskAnnotator())
                elif key == "PolygonAnnotator":
                    annotators.append(sv.PolygonAnnotator(color_lookup=self.color_lookup))
                elif key == "BlurAnnotator":
                    annotators.append(sv.BlurAnnotator())
                elif key == "PixelateAnnotator":
                    annotators.append(sv.PixelateAnnotator())
                elif key == "LabelAnnotator":
                    annotators.append(sv.LabelAnnotator(
                        text_position=sv.Position.BOTTOM_CENTER,
                        color_lookup=self.color_lookup
                    ))
            except Exception as e:
                print(f"Warning: Could not create annotator {key}: {e}")
                continue
        return annotators
    
    def get_zone_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive zone statistics."""
        stats = {}
        for zone in self.zones:
            zone_id = zone.zone_id
            stats[zone_id] = {
                'current_count': zone.current_count,
                'cumulative_count': zone.cumulative_count,
                'count_criteria': zone.count_criteria,
                'detection_history_length': len(zone.detection_history)
            }
        return stats
    
    def reset_zones(self):
        """Reset all zone counters and history."""
        for zone in self.zones:
            zone.current_count = 0
            zone.cumulative_count = 0
            zone.detection_history = []
            zone.seen_tracker_ids = set()
 
        # Reset trace annotator to clear traces
        self.trace_annotator = sv.TraceAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
            thickness=2,
            trace_length=30,
            position=sv.Position.CENTER
        )


class InferenceEngine:
    """Handles model loading, inference, and class filtering."""
    def __init__(self, parent=None):
        self.parent = parent
        
        # Set the device (video_region.base.main_window)
        self.device = "cpu"
        
        # Initialize model and task
        self.model = None
        self.task = None
        
        # Default inference parameters
        self.conf = 0.3
        self.iou = 0.2
        self.area_min = 0.0
        self.area_max = 0.4

        self.class_names = []
        self.selected_classes = []
        
        # Initialize ByteTrack tracker - enabled by default
        self.tracker = sv.ByteTrack()
        self.tracking_enabled = True
        
        self.count_criteria = "Centroid"  # Criteria for counting objects in regions
        self.display_outside = True
        
        # Initialize ZoneManager
        self.zone_manager = RegionZoneManager(
            count_criteria=self.count_criteria,
            display_outside=self.display_outside
        )

    def load_model(self, model_path, task):
        """Load the YOLO model for inference."""
        # Make cursor busy while loading
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Set the task
            self.task = task
            # Load the model using YOLO from ultralytics
            self.model = YOLO(model_path, task=self.task)
            # Store class names from the model
            self.class_names = list(self.model.names.values())
            
            # Run a dummy inference to ensure the model is loaded correctly
            self.model(np.zeros((640, 640, 3), dtype=np.uint8))
            
            # Set the device for inference
            self.set_device()
            
            QMessageBox.information(self.parent,
                                    "Model Loaded",
                                    "Model loaded successfully.")
                        
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self.parent, 
                                 "Model Load Error",
                                 "Failed to load model (see console for details)")
            
        finally:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            
    def set_device(self):
        """Set the device for inference."""
        self.device = self.parent.parent.main_window.device

    def set_selected_classes(self, class_indices):
        """Set the selected classes for inference."""
        self.selected_classes = class_indices
        
    def set_inference_params(self, conf, iou, area_min, area_max):
        """Set inference parameters for the video region."""
        self.conf = conf
        self.iou = iou
        self.area_min = area_min
        self.area_max = area_max

    def set_count_criteria(self, count_criteria):
        """Set the criteria for counting objects in regions."""
        self.count_criteria = count_criteria
        self.zone_manager.count_criteria = count_criteria
        
    def set_display_outside_detections(self, display_outside):
        """Set whether to display detections outside the regions."""
        self.display_outside = display_outside
        self.zone_manager.display_outside = display_outside

    def infer(self, frame):
        """Run inference on a single frame with the current model."""
        if self.model is None:
            return sv.Detections.empty()
        
        # Detect, and filter results based on confidence and IoU
        results = self.model(frame, 
                             conf=self.conf, 
                             iou=self.iou, 
                             classes=self.selected_classes,
                             half=True,
                             device=self.device)[0]
        
        # Convert results to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Apply tracking if enabled and detections exist
        if self.tracking_enabled and len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        
        # Apply area filter to detections
        detections = self.apply_area_filter(frame, detections)
            
        return detections
    
    def update_tracker(self, detections):
        """Update the tracker with the current detections."""
        tracker_detections = self.tracker.update_with_detections(detections)
        if tracker_detections:
            return tracker_detections
        return detections
    
    def apply_area_filter(self, frame, detections):
        """Filter detections based on area thresholds."""
        if detections is None or len(detections) == 0:
            return detections
        
        # Calculate the area of the frame
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Filter detections based on relative area
        detections = detections[(detections.area / frame_area) <= self.area_max]
        detections = detections[(detections.area / frame_area) >= self.area_min]
        
        return detections
            
    def count_objects_in_regions(self, detections, region_polygons):
        """Count objects in each region using the new zone manager."""
        if not region_polygons or detections is None or len(detections) == 0:
            region_counts = [0 for _ in region_polygons]
            return detections, region_counts

        # Update zones with current regions and criteria
        self.zone_manager.update_regions(region_polygons)
        
        # Count detections using the zone manager
        filtered_detections, region_counts, zone_detection_map = self.zone_manager.count_detections(detections)
        
        # Store zone detection mapping for potential future use
        self._last_zone_detection_map = zone_detection_map
        
        return filtered_detections, region_counts
    
    def get_zone_statistics(self):
        """Get comprehensive zone statistics."""
        return self.zone_manager.get_zone_statistics()
    
    def reset_tracker(self):
        """Reset the ByteTrack tracker and zone manager."""
        if self.tracker:
            self.tracker = sv.ByteTrack()

    def reset_zone_manager(self):
        """Reset the zone manager."""
        if self.zone_manager:
            self.zone_manager.reset_zones()
            
    def cleanup(self):
        """Clean up resources and reset states."""
        self.reset_tracker()
        self.reset_zone_manager()
        # Reset model and task
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()
            