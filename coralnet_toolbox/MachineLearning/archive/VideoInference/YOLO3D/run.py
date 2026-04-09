import os
import sys
import time
import cv2
import numpy as np
import torch
import argparse

try:
    import filterpy
except ImportError:
    print("Error: filterpy is required. Please install it with 'pip install filterpy>=1.4.5' before using this script.")
    sys.exit(1)

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def detect_device(preferred_device=None):
    """
    Detect and return the best available device for PyTorch operations.
    
    Args:
        preferred_device (str): User's preferred device ('cuda', 'mps', 'cpu', or None for auto)
        
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu')
    """
    print("Detecting available devices...")
    
    # Check what's available
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Print device availability information
    if cuda_available:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device(s))")
    else:
        print("✗ CUDA not available")
        
    if mps_available:
        print("✓ Apple Silicon MPS available")
    else:
        print("✗ Apple Silicon MPS not available")
        
    print("✓ CPU always available")
    
    # Determine the device to use
    if preferred_device is None:
        # Auto-detect best device
        if cuda_available:
            device = 'cuda'
            print(f"🚀 Auto-selected device: {device} (CUDA GPU acceleration)")
        elif mps_available:
            device = 'mps'
            print(f"🚀 Auto-selected device: {device} (Apple Silicon acceleration)")
        else:
            device = 'cpu'
            print(f"🚀 Auto-selected device: {device} (CPU processing)")
    else:
        # Use user's preferred device if available
        if preferred_device == 'cuda':
            if cuda_available:
                device = 'cuda'
                print(f"🚀 Using user-specified device: {device}")
            else:
                print("⚠️  CUDA requested but not available, falling back to CPU")
                device = 'cpu'
        elif preferred_device == 'mps':
            if mps_available:
                device = 'mps'
                print(f"🚀 Using user-specified device: {device}")
            else:
                print("⚠️  MPS requested but not available, falling back to CPU")
                device = 'cpu'
        elif preferred_device == 'cpu':
            device = 'cpu'
            print(f"🚀 Using user-specified device: {device}")
        else:
            print(f"⚠️  Unknown device '{preferred_device}' requested, falling back to auto-detection")
            return detect_device(None)  # Recursive call for auto-detection
    
    return device


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="3D Object Detection with YOLO and Depth Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,        
        epilog="""
Examples:

  # Basic usage with webcam and default settings
  python run.py

  # Use a video file with custom YOLO model
  python run.py --input video.mp4 --yolo-path custom_model.pt --output result.mp4
  # Process at smaller resolution for faster processing (longest edge = 640px)
  python run.py --input video.mp4 --size 640 --output result_small.mp4

  # Use larger models with GPU
  python run.py --yolo-size large --depth-size base --device cuda

  # Headless processing (no display windows) with low resolution (longest edge = 480px)
  python run.py --input video.mp4 --output result.mp4 --no-display --size 480

  # Filter for specific classes (person=0, car=2)
  python run.py --classes 0 2 --conf-threshold 0.5
""",
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='0',
        help='Input video source (video file path, webcam index like "0", or "1")'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.mp4',
        help='Output video file path'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=None,
        help='Target size for the longest edge in pixels. If not specified, original resolution is used.'
    )
    
    # YOLO model arguments
    yolo_group = parser.add_mutually_exclusive_group()
    yolo_group.add_argument(
        '--yolo-path',
        type=str,
        help='Path to custom YOLO model file (.pt)'
    )
    
    yolo_group.add_argument(
        '--yolo-size',
        type=str,
        choices=['nano', 'small', 'medium', 'large', 'extra'],
        default='nano',
        help='YOLO model size (ignored if --yolo-path is specified)'
    )
    
    # Depth model arguments
    parser.add_argument(
        '--depth-size',
        type=str,
        choices=['small', 'base', 'large', 'apple'],
        default='small',
        help='Depth estimation model size'
    )
    
    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for object detection'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
    )
    
    parser.add_argument(
        '--classes',
        type=int,
        nargs='*',
        help='Filter by specific class IDs (e.g., --classes 0 1 2 for persons, bicycles, cars)'
    )
    
    # Frame range parameters
    parser.add_argument(
        '--start_at',
        type=int,
        default=0,
        help='Start processing at this frame number (0-based)'
    )
    parser.add_argument(
        '--end_at',
        type=int,
        default=None,
        help='End processing at this frame number (inclusive, 0-based). If not provided, '
             'process until the end of the video.'
    )
    
    # Device settings
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps', 'auto'],
        default='auto',
        help='Device to run inference on'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help='Disable object tracking'
    )
    
    parser.add_argument(
        '--no-bev',
        action='store_true',
        help='Disable Bird\'s Eye View visualization'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display windows (useful for headless processing)'
    )
    # Add --yes-display argument
    parser.add_argument(
        '--yes-display',
        action='store_true',
        help='Show all visualization windows (result, depth, detection). By default, only the result frame is shown.'
    )

    # Add --no-depth-viz argument
    parser.add_argument(
        '--no-depth-viz',
        action='store_true',
        help='Disable depth map visualization into the output frame (depth is still used for processing).'
    )
    
    # Object dimensions argument
    parser.add_argument(
        '--object-dimensions',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='Default dimensions for detected objects [length, width, height] in meters'
    )
    
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration from arguments
    # ===============================================
    
    # Input/Output
    source = args.input
    # Convert webcam string numbers to integers
    try:
        if source.isdigit():
            source = int(source)
    except (ValueError, AttributeError):
        pass  # Keep as string for video files
    
    output_path = args.output
    
    # Frame sizing - calculate scale factor from target size
    target_size = args.size
    
    # Model settings
    yolo_model_size = args.yolo_size
    yolo_model_path = args.yolo_path
    depth_model_size = args.depth_size
    
    # Device settings - centralized device detection
    requested_device = args.device if args.device != 'auto' else None
    device = detect_device(requested_device)
    
    # Detection settings
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    classes = args.classes
    
    # Feature toggles
    enable_tracking = not args.no_tracking
    enable_bev = not args.no_bev
    enable_depth = not args.no_depth_viz
    enable_display = not args.no_display
    show_all_frames = args.yes_display and not args.no_depth_viz
    
    # Frame range parameters
    start_frame = args.start_at
    end_frame = args.end_at
    
    # Object dimensions
    object_dimensions = args.object_dimensions
    
    # Camera parameters - simplified approach
    camera_params_file = None  # Path to camera parameters file (None to use default parameters)
    # ===============================================
    print("\nConfiguration:")
    print(f"Input source: {source}")
    print(f"Output path: {output_path}")
    if target_size is not None:
        print(f"Target size: {target_size}px (longest edge)")
    else:
        print("Using original resolution (no scaling)")
    print(f"YOLO model: {'Custom path: ' + yolo_model_path if yolo_model_path else 'Size: ' + yolo_model_size}")
    print(f"Depth model size: {depth_model_size}")
    print(f"Device: {device}")
    print(f"Frame range: {start_frame} to {end_frame if end_frame is not None else 'end'}")
    print(f"Tracking: {'enabled' if enable_tracking else 'disabled'}")
    print(f"Bird's Eye View: {'enabled' if enable_bev else 'disabled'}")
    print(f"Display: {'enabled' if enable_display else 'disabled'}")
    print(f"Object dimensions: {object_dimensions} (L, W, H in meters)")
    
    # Initialize models with the detected device
    print("\nInitializing models...")
    try:
        print("Loading YOLO object detector...")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device,
            path=yolo_model_path
        )
        print("✓ YOLO object detector loaded successfully")
        
    except Exception as e:
        print(f"✗ Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu',
            path=yolo_model_path
        )
    
    try:
        print("Loading depth estimation model...")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
        print("✓ Depth estimation model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # Convert string number to integer for webcam
            
    except ValueError:
        pass  # Keep as string (for video file)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Sometimes happens with webcams
        fps = 30
    
    # Calculate scale factor and target dimensions
    if target_size is not None:
        # Calculate scale factor based on longest edge
        longest_edge = max(original_width, original_height)
        scale_factor = target_size / longest_edge
        
        # Calculate new dimensions preserving aspect ratio
        width = int(original_width * scale_factor)
        height = int(original_height * scale_factor)
    else:
        # Use original dimensions
        scale_factor = 1.0
        width = original_width
        height = original_height    
        
    print(f"Original resolution: {original_width}x{original_height}")
    if target_size is not None:
        print(f"Target resolution: {width}x{height} (longest edge: {max(width, height)}px)")
    else:
        print(f"Using original resolution: {width}x{height}")
        
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator(default_dimensions=object_dimensions)
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(image_shape=(width, height), scale=100)  # Increased scale to spread objects out
    
    # Initialize video writer - use a more reliable codec and check output path
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # On Windows, try different codec options
    if sys.platform == 'win32':
        try:
            # First try H264 codec
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Check if writer was successfully initialized
            if not out.isOpened():
                # Fallback to XVID codec
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Last resort, try MJPG
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        print(f"Warning: Could not create output video with standard codecs. Trying AVI format.")
                        # Try changing extension to .avi
                        output_path = os.path.splitext(output_path)[0] + '.avi'
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            # Last fallback to MP4V
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        # For other platforms, use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Check if writer was successfully initialized
    if not out.isOpened():
        print(f"Error: Could not create output video file at {output_path}")
        print("Continuing without saving output video.")
        out = None
    else:
        print(f"Successfully opened output video file: {output_path}")
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    
    # Variables for cumulative volume calculation
    cumulative_volume_m3 = 0.0
    previous_tracked_ids = set()
    track_data = {}  # Stores the last known data (e.g., volume) for each track ID
    
    # Main loop
    current_frame = 0
    while True:
        # Check for key press at the beginning of each loop
        if enable_display:
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
        try:            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames before start_frame
            if current_frame < start_frame:
                current_frame += 1
                continue
                
            # Stop if we've processed the end_frame
            if end_frame is not None and current_frame > end_frame:
                print(f"Reached end frame {end_frame}, stopping processing.")
                break
                
            # Apply resizing if needed
            if target_size is not None:
                frame = cv2.resize(frame, (width, height))
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
                
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
                
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            if enable_tracking:
                # Get the set of currently active track IDs from this frame's detections
                current_tracked_ids = {det[3] for det in detections if det[3] is not None}

                # Determine which IDs have disappeared since the last frame
                disappeared_ids = previous_tracked_ids - current_tracked_ids

                # For each disappeared track, add its last known volume to the cumulative total
                if disappeared_ids:
                    for track_id in disappeared_ids:
                        if track_id in track_data:
                            cumulative_volume_m3 += track_data[track_id]['volume']
                            del track_data[track_id]  # Clean up the old track data

                # Update the set of previous IDs for the next frame's comparison
                previous_tracked_ids = current_tracked_ids
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                    depth_method = 'median'
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    
                    # Calculate 3D bounding box 
                    estimated_3d_box = bbox3d_estimator.estimate_3d_box(bbox, depth_value)
                    
                    # Add estimated 3D box data
                    box_3d.update(estimated_3d_box)
                    boxes_3d.append(box_3d)
                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        track_data[obj_id] = {'volume': box_3d['volume']}
                        active_ids.append(obj_id)
                        
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)
                
            # Step 4: Visualization
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:                    
                    # Use a default color for all objects
                    color = (255, 255, 255)  # Green as default
                    
                    # Draw box with depth information
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Draw Bird's Eye View if enabled
            if enable_bev:
                try:
                    # Reset BEV and draw objects
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    
                    # Get resized BEV image directly from the BEV object
                    bev_resized = bev.get_resized_image(width, height)
                    
                    # Get dimensions for positioning
                    bev_height, bev_width = bev_resized.shape[:2]
                    
                    # Create a region of interest in the result frame
                    roi = result_frame[height - bev_height:height, 0:bev_width]
                    
                    # Simple overlay - just copy the BEV image to the ROI
                    result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                    
                    # Add a border around the BEV visualization
                    cv2.rectangle(result_frame, 
                                  (0, height - bev_height), 
                                  (bev_width, height), 
                                  (255, 255, 255), 1)
                    
                    # Add a title to the BEV visualization
                    cv2.putText(result_frame, "Bird's Eye View", 
                                (10, height - bev_height + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"

            # Add FPS, device info, frame count, and tracks info to the result frame (top-right corner)
            text_main = f"{fps_display} | Device: {device} | Frame: {current_frame}"
            text_tracks = ""
            if enable_tracking:
                current_tracks = len(detector.tracking_trajectories)
                # Use a set to keep track of all unique tracks seen so far
                detector.all_tracks_seen.update(detector.tracking_trajectories.keys())
                total_tracks = len(detector.all_tracks_seen)
                text_tracks = f"Tracks Current: {current_tracks} | Total: {total_tracks}"
                
            # Calculate volume statistics
            text_volume = ""
            if boxes_3d:
                # Only count volumes for currently tracked objects (those with object_id)
                text_volume = f"Cumulative Vol: {(cumulative_volume_m3):.1f} m3"

            # Calculate text size for right alignment
            (text_width_main, text_height_main), _ = cv2.getTextSize(text_main, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (text_width_tracks, text_height_tracks), _ = cv2.getTextSize(text_tracks, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (text_width_volume, text_height_volume), _ = cv2.getTextSize(text_volume, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            box_width = max(text_width_main, text_width_tracks, text_width_volume) + 24  # padding
            box_height = text_height_main + text_height_tracks + text_height_volume + 40  # padding and gaps

            # Calculate position dynamically based on frame size - top-right corner
            margin_x = int(width * 0.00)  # 0% from right edge
            margin_y = int(height * 0.00)  # 0% from top edge
            box_x1 = width - box_width - margin_x
            box_y1 = margin_y
            x_offset = box_x1 + 12  # 12px left padding inside box

            # Calculate vertical centering for text
            total_text_height = text_height_main + text_height_tracks + text_height_volume + 16  # gaps
            center_y = box_y1 + box_height / 2
            y_offset_main = int(center_y - total_text_height / 2 + text_height_main)
            y_offset_tracks = int(y_offset_main + text_height_main + 8)  # 8px gap between lines
            y_offset_volume = int(y_offset_tracks + text_height_tracks + 8)  # 8px gap between lines

            # Draw semi-transparent white box behind the text
            overlay = result_frame.copy()
            box_x2 = box_x1 + box_width
            box_y2 = box_y1 + box_height
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

            # Place main text on the result frame (top-left corner)
            cv2.putText(result_frame, text_main, (x_offset, y_offset_main),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Place tracks info below main text
            if enable_tracking:
                cv2.putText(result_frame, text_tracks, (x_offset, y_offset_tracks),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Place volume info below tracks text
            if text_volume:
                cv2.putText(result_frame, text_volume, (x_offset, y_offset_volume),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add depth map to the corner of the result frame if enabled
            if enable_depth:
                try:
                    # Calculate depth map dimensions based on frame size
                    depth_size_factor = 0.25  # Percentage of frame height
                    depth_height = int(height * depth_size_factor)
                    
                    # Maintain original aspect ratio
                    depth_aspect_ratio = width / height
                    depth_width = int(depth_height * depth_aspect_ratio)
                    
                    # Make sure depth visualization doesn't exceed 1/3 of the frame width
                    if depth_width > width // 3:
                        depth_width = width // 3
                        depth_height = int(depth_width / depth_aspect_ratio)
                    
                    # Resize depth map
                    depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                    
                    # Overlay on the top-left corner
                    result_frame[0:depth_height, 0:depth_width] = depth_resized
                    
                    # Add a border around the depth visualization
                    cv2.rectangle(result_frame, 
                                  (0, 0), 
                                  (depth_width, depth_height), 
                                  (255, 255, 255), 1)
                    
                except Exception as e:
                    print(f"Error adding depth map to result: {e}")
            
            # Write frame to output video (only if writer is valid)
            if out is not None and out.isOpened():
                out.write(result_frame)
            
            # Display frames only if display is enabled
            if enable_display:
                cv2.imshow("3D Object Detection", result_frame)
                if show_all_frames:
                    # Only show depth map if not disabled
                    if enable_depth:
                        cv2.imshow("Depth Map", depth_colored)
                    cv2.imshow("Object Detection", detection_frame)
                    
            # Check for key press again at the end of the loop
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
                
            # Increment frame counter
            current_frame += 1
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            if enable_display:
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                    print("Exiting program...")
                    break
            
            # Still increment frame counter even if there was an error
            current_frame += 1
            continue
    
    # Clean up
    print("Cleaning up resources...")
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows()