from typing import List, Optional, Dict, Tuple

import numpy as np
import supervision as sv
from shapely.geometry import Polygon


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
        
    def trigger_with_metadata(self, detections: sv.Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced trigger that returns both boolean mask and metadata.
        
        Returns:
            tuple: (boolean_mask, detection_indices_in_zone)
        """
        is_inside = self.trigger(detections)
        detection_indices = np.where(is_inside)[0] if len(is_inside) > 0 else np.array([])
        
        # Update cumulative count based on unique tracker IDs
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            # Use tracker IDs for more accurate cumulative counting
            inside_tracker_ids = detections.tracker_id[is_inside]
            # Filter out None values
            valid_tracker_ids = [tid for tid in inside_tracker_ids if tid is not None]
            
            for tid in valid_tracker_ids:
                if tid not in self.seen_tracker_ids:
                    self.seen_tracker_ids.add(tid)
                    self.cumulative_count += 1
    
        # Store detection history for future features - use self.current_count which is updated by parent class
        self.detection_history.append({
            'count': self.current_count,  # Current detections in zone right now
            'cumulative_count': self.cumulative_count,  # Total unique detections seen over time
            'detection_indices': detection_indices.tolist()
        })
        
        # Keep only last 100 entries to prevent memory issues
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
        
        return is_inside, detection_indices


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
        show_cumulative_count: bool = True,  # Changed from show_max_count
        text_position: str = "top_left"  # New parameter for text positioning
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
        self.show_current_count = show_current_count  # Store as instance variable instead
        self.show_cumulative_count = show_cumulative_count  # Changed from show_max_count
        self.text_position = text_position
    
    def annotate(self, scene: np.ndarray, zone_metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Annotate the scene with enhanced zone information.
        
        Args:
            scene: The image to annotate
            zone_metadata: Optional metadata about the zone
            
        Returns:
            Annotated scene
        """
        import cv2
        
        # First draw the zone using parent method (without label)
        annotated_scene = super().annotate(scene=scene, label=None)
        
        # Build custom label with enhanced information
        label_parts = []
        
        if self.show_zone_id:
            label_parts.append(f"ID: {self.custom_zone.zone_id}")
        
        if self.show_current_count:
            label_parts.append(f"Current: {self.custom_zone.current_count}")
            
        if self.show_cumulative_count:
            label_parts.append(f"Cumulative: {self.custom_zone.cumulative_count}")
        
        # Join all label parts
        custom_label = " | ".join(label_parts) if label_parts else None
        
        if custom_label:
            # Get polygon bounds for text positioning
            polygon_points = self.custom_zone.polygon
            min_x = int(np.min(polygon_points[:, 0]))
            min_y = int(np.min(polygon_points[:, 1]))
            max_x = int(np.max(polygon_points[:, 0]))
            max_y = int(np.max(polygon_points[:, 1]))
            
            # Calculate text position based on preference
            if self.text_position == "top_left":
                text_x = min_x + self.text_padding
                text_y = min_y + 25  # Offset from top edge
            elif self.text_position == "top_center":
                text_x = (min_x + max_x) // 2
                text_y = min_y + 25
            else:  # center (default behavior)
                text_x = (min_x + max_x) // 2
                text_y = (min_y + max_y) // 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                custom_label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            
            # Draw text background rectangle
            if self.text_position == "top_left":
                rect_x1 = text_x - self.text_padding
                rect_y1 = text_y - text_height - self.text_padding
                rect_x2 = text_x + text_width + self.text_padding
                rect_y2 = text_y + baseline + self.text_padding
            elif self.text_position == "top_center":
                rect_x1 = text_x - text_width // 2 - self.text_padding
                rect_y1 = text_y - text_height - self.text_padding
                rect_x2 = text_x + text_width // 2 + self.text_padding
                rect_y2 = text_y + baseline + self.text_padding
            else:  # center
                rect_x1 = text_x - text_width // 2 - self.text_padding
                rect_y1 = text_y - text_height // 2 - self.text_padding
                rect_x2 = text_x + text_width // 2 + self.text_padding
                rect_y2 = text_y + text_height // 2 + self.text_padding
            
            # Draw semi-transparent background
            overlay = annotated_scene.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
            annotated_scene = cv2.addWeighted(annotated_scene, 0.7, overlay, 0.3, 0)
            
            # Draw text
            if self.text_position in ["top_left", "top_center"]:
                text_y_final = text_y
            else:  # center
                text_y_final = text_y + text_height // 2
                
            if self.text_position == "top_center":
                text_x_final = text_x - text_width // 2
            else:
                text_x_final = text_x
                
            cv2.putText(
                annotated_scene,
                custom_label,
                (text_x_final, text_y_final),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness
            )
        
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
                thickness=2,
                text_scale=0.6,
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