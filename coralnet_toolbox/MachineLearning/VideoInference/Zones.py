from typing import List, Optional, Union, Dict, Tuple

import numpy as np

from shapely.geometry import Polygon

import supervision as sv

# Add color palette for consistent tracking colors
TRACKING_COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1", "#FF6347", "#4169E1", "#32CD32", "#FFD700"])


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
            "Bounding Box": [sv.Position.TOP_LEFT, sv.Position.TOP_RIGHT, 
                           sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT],
            "Bottom Center": [sv.Position.BOTTOM_CENTER]
        }
        
        triggering_anchors = anchor_mapping.get(count_criteria, [sv.Position.CENTER])
        super().__init__(polygon=polygon, triggering_anchors=triggering_anchors)
        
        self.count_criteria = count_criteria
        self.zone_id = zone_id or f"zone_{id(self)}"
        self.detection_history = []  # For future tracking features
        
    def trigger_with_metadata(self, detections: sv.Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced trigger that returns both boolean mask and metadata.
        
        Returns:
            tuple: (boolean_mask, detection_indices_in_zone)
        """
        is_inside = self.trigger(detections)
        detection_indices = np.where(is_inside)[0] if len(is_inside) > 0 else np.array([])
        
        # Store detection history for future features
        self.detection_history.append({
            'count': self.current_count,
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
        display_in_zone_count: bool = True,
        opacity: float = 0,
        show_zone_id: bool = True,
        show_max_count: bool = True,
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
            display_in_zone_count: Whether to show the count
            opacity: Zone fill opacity
            show_zone_id: Whether to show zone ID
            show_max_count: Whether to show max simultaneous count
        """
        super().__init__(
            zone=zone,
            color=color,
            thickness=thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            display_in_zone_count=display_in_zone_count,
            opacity=opacity
        )
        self.custom_zone = zone
        self.show_zone_id = show_zone_id
        self.show_max_count = show_max_count
    
    def annotate(self, scene: np.ndarray, zone_metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Annotate the scene with enhanced zone information.
        
        Args:
            scene: The image to annotate
            zone_metadata: Optional metadata about the zone
            
        Returns:
            Annotated scene
        """
        # Build custom label with enhanced information
        label_parts = []
        
        if self.display_in_zone_count:
            label_parts.append(f"Count: {self.custom_zone.current_count}")
        
        if self.show_zone_id:
            label_parts.append(f"ID: {self.custom_zone.zone_id}")
            
        if self.show_max_count and zone_metadata:
            max_count = zone_metadata.get('max_simultaneous', 0)
            label_parts.append(f"Max: {max_count}")
        
        # Join all label parts
        custom_label = " | ".join(label_parts) if label_parts else None
        
        # Use the parent annotator with our custom label
        return super().annotate(scene=scene, label=custom_label)


class RegionZoneManager:
    """Manages multiple custom polygon zones for region counting with advanced features."""
    
    def __init__(self, count_criteria: str = "Centroid", display_outside: bool = True):
        self.zones: List[CustomPolygonZone] = []
        self.zone_annotators: List[CustomPolygonZoneAnnotator] = []
        self.count_criteria = count_criteria
        self.display_outside = display_outside
        self.zone_metadata: Dict[str, Dict] = {}  # Store additional zone information
        
        # Add tracking annotators - enabled by default
        self.box_annotator = sv.BoxAnnotator(color=TRACKING_COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=TRACKING_COLORS, 
            text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=TRACKING_COLORS, 
            position=sv.Position.CENTER, 
            trace_length=100, 
            thickness=2
        )
        
        # Tracking settings - enabled by default
        self.show_tracking_traces = True
        self.show_tracking_boxes = False  # Keep existing box drawing
        self.show_tracking_labels = True

    def update_regions(self, region_polygons: List[Polygon], zone_ids: Optional[List[str]] = None):
        """
        Update zones based on new region polygons.
        
        Args:
            region_polygons: List of shapely Polygon objects
            zone_ids: Optional list of zone identifiers
        """
        self.zones = []
        self.zone_annotators = []
        self.zone_metadata = {}
        zone_ids = zone_ids or [f"region_{i}" for i in range(len(region_polygons))]
        
        for i, poly in enumerate(region_polygons):
            # Convert shapely polygon to numpy array
            coords = np.array(poly.exterior.coords[:-1], dtype=np.int32)
            
            # Create custom zone
            zone = CustomPolygonZone(
                polygon=coords,
                count_criteria=self.count_criteria,
                zone_id=zone_ids[i]
            )
            
            # Create zone annotator with cycling colors
            color_idx = i % len(TRACKING_COLORS.colors)
            zone_color = TRACKING_COLORS.colors[color_idx]
            
            annotator = CustomPolygonZoneAnnotator(
                zone=zone,
                color=zone_color,
                thickness=2,
                text_scale=0.6,
                opacity=0.1
            )
            
            self.zones.append(zone)
            self.zone_annotators.append(annotator)
            self.zone_metadata[zone_ids[i]] = {'max_simultaneous': 0}

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
            current_max = self.zone_metadata[zone_id].get('max_simultaneous', 0)
            self.zone_metadata[zone_id]['max_simultaneous'] = max(current_max, count)
            
            # Store which detections are in this zone
            zone_detection_map[zone_id] = detection_indices.tolist()
            
            # Collect detection indices for filtering
            if self.display_outside or len(detection_indices) > 0:
                filtered_detection_indices.update(detection_indices)
        
        # Filter detections based on display_outside setting
        if self.display_outside:
            filtered_detections = detections
        else:
            # Only show detections that are inside at least one zone
            if filtered_detection_indices:
                mask = np.zeros(len(detections), dtype=bool)
                mask[list(filtered_detection_indices)] = True
                filtered_detections = detections[mask]
            else:
                filtered_detections = sv.Detections.empty()
        
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
        
        # Draw zone annotations
        for i, annotator in enumerate(self.zone_annotators):
            if i < len(self.zones):
                zone_id = self.zones[i].zone_id
                metadata = self.zone_metadata.get(zone_id, {})
                annotated_scene = annotator.annotate(annotated_scene, zone_metadata=metadata)
        
        # Add tracking visualizations if enabled and detections provided
        if detections is not None and len(detections) > 0:
            if self.show_tracking_traces:
                annotated_scene = self.trace_annotator.annotate(annotated_scene, detections)
            
            if self.show_tracking_boxes:
                annotated_scene = self.box_annotator.annotate(annotated_scene, detections)
            
            if self.show_tracking_labels and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
                annotated_scene = self.label_annotator.annotate(annotated_scene, detections, labels)
        
        return annotated_scene
    
    def get_zone_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics for all zones."""
        stats = {}
        for zone in self.zones:
            stats[zone.zone_id] = {
                'current_count': zone.current_count,
                'count_criteria': zone.count_criteria,
                'history_length': len(zone.detection_history),
                'metadata': self.zone_metadata.get(zone.zone_id, {})
            }
        return stats
    
    def reset_zones(self):
        """Reset all zone counters and history."""
        for zone in self.zones:
            zone.detection_history = []
            # Reset current count if the zone has this attribute
            if hasattr(zone, 'current_count'):
                zone.current_count = 0
        
        # Reset metadata
        for zone_id in self.zone_metadata:
            self.zone_metadata[zone_id]['max_simultaneous'] = 0