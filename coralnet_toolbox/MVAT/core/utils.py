import os
import struct

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions for COLMAP parsing
# ----------------------------------------------------------------------------------------------------------------------


def parse_colmap_cameras(cameras_file):
    """
    Parse COLMAP cameras file (.txt or .bin format).

    Returns:
        dict: camera_id -> camera_params dict
    """
    if cameras_file.endswith('.bin'):
        return parse_colmap_cameras_bin(cameras_file)
    else:
        return parse_colmap_cameras_txt(cameras_file)


def parse_colmap_images(images_file):
    """
    Parse COLMAP images file (.txt or .bin format).

    Returns:
        dict: image_name -> image_data dict
    """
    if images_file.endswith('.bin'):
        return parse_colmap_images_bin(images_file)
    else:
        return parse_colmap_images_txt(images_file)
    

def parse_colmap_points3D(points3D_file):
    """
    Parse COLMAP points3D file (.txt or .bin format).

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    if points3D_file.endswith('.bin'):
        return parse_colmap_points3D_bin(points3D_file)
    else:
        return parse_colmap_points3D_txt(points3D_file)


def parse_colmap_cameras_txt(cameras_txt_path):
    """
    Parse COLMAP cameras.txt file.

    Returns:
        dict: camera_id -> camera_params dict
    """
    cameras = {}
    with open(cameras_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip comments and header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        if line.startswith('Number of cameras:'):
            i += 1
            continue

        # Parse camera line: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        parts = line.split()
        if len(parts) < 4:
            i += 1
            continue

        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])

        # Parse camera parameters based on model
        if model == 'PINHOLE':
            # PARAMS: fx, fy, cx, cy
            fx, fy, cx, cy = map(float, parts[4:8])
            params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        elif model == 'SIMPLE_PINHOLE':
            # PARAMS: f, cx, cy
            f, cx, cy = map(float, parts[4:7])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy}
        elif model == 'SIMPLE_RADIAL':
            # PARAMS: f, cx, cy, k
            f, cx, cy, k = map(float, parts[4:8])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k}
        elif model == 'RADIAL':
            # PARAMS: f, cx, cy, k1, k2
            f, cx, cy, k1, k2 = map(float, parts[4:9])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2}
        else:
            # Default to pinhole with identity parameters
            params = {'fx': width * 0.8, 'fy': height * 0.8, 'cx': width / 2, 'cy': height / 2}

        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
        i += 1

    return cameras


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt file.

    Returns:
        dict: image_name -> image_data dict
    """
    images = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip comments and header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        if line.startswith('Number of images:'):
            i += 1
            continue

        # Parse image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = ' '.join(parts[9:])

        # Skip the points2d line for now
        i += 2

        images[name] = {
            'image_id': image_id,
            'camera_id': camera_id,
            'quaternion': np.array([qw, qx, qy, qz]),
            'translation': np.array([tx, ty, tz])
        }

    return images


def parse_colmap_cameras_bin(cameras_bin_path):
    """
    Parse COLMAP cameras.bin file.

    Returns:
        dict: camera_id -> camera_params dict
    """
    cameras = {}
    
    try:
        with open(cameras_bin_path, 'rb') as f:
            # Read number of cameras
            num_cameras_data = f.read(8)
            if len(num_cameras_data) < 8:
                print(f"Warning: Could not read number of cameras from {cameras_bin_path}")
                return cameras
            num_cameras = struct.unpack('<Q', num_cameras_data)[0]
            print(f"Reading {num_cameras} cameras from binary file")
            
            for i in range(num_cameras):
                try:
                    # Read camera ID
                    camera_id_data = f.read(4)
                    if len(camera_id_data) < 4:
                        print(f"Warning: Could not read camera ID for camera {i}")
                        break
                    camera_id = struct.unpack('<I', camera_id_data)[0]
                    
                    # Read model (enum)
                    model_data = f.read(4)
                    if len(model_data) < 4:
                        print(f"Warning: Could not read model for camera {camera_id}")
                        break
                    model_id = struct.unpack('<I', model_data)[0]
                    
                    # Map model ID to string
                    model_map = {
                        0: 'SIMPLE_PINHOLE',
                        1: 'PINHOLE',
                        2: 'SIMPLE_RADIAL',
                        3: 'RADIAL',
                        4: 'OPENCV',
                        5: 'OPENCV_FISHEYE',
                        6: 'FULL_OPENCV',
                        7: 'FOV',
                        8: 'SIMPLE_RADIAL_FISHEYE',
                        9: 'RADIAL_FISHEYE',
                        10: 'THIN_PRISM_FISHEYE'
                    }
                    model = model_map.get(model_id, 'PINHOLE')
                    
                    # Read width and height
                    width_data = f.read(4)
                    height_data = f.read(4)
                    if len(width_data) < 4 or len(height_data) < 4:
                        print(f"Warning: Could not read dimensions for camera {camera_id}")
                        break
                    width = struct.unpack('<I', width_data)[0]
                    height = struct.unpack('<I', height_data)[0]
                    
                    # Read number of parameters
                    num_params_data = f.read(4)
                    if len(num_params_data) < 4:
                        print(f"Warning: Could not read num_params for camera {camera_id}")
                        break
                    num_params = struct.unpack('<I', num_params_data)[0]
                    
                    # Read parameters
                    params_data = []
                    for j in range(num_params):
                        param_data = f.read(8)
                        if len(param_data) < 8:
                            print(f"Warning: Could not read parameter {j} for camera {camera_id}")
                            break
                        params_data.append(struct.unpack('<d', param_data)[0])
                    
                    if len(params_data) < num_params:
                        print(f"Warning: Incomplete parameters for camera {camera_id}")
                        continue
                    
                    # Parse parameters based on model
                    if model == 'PINHOLE' and len(params_data) >= 4:
                        fx, fy, cx, cy = params_data[:4]
                        params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                    elif model == 'SIMPLE_PINHOLE' and len(params_data) >= 3:
                        f, cx, cy = params_data[:3]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy}
                    elif model == 'SIMPLE_RADIAL' and len(params_data) >= 4:
                        f, cx, cy, k = params_data[:4]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k}
                    elif model == 'RADIAL' and len(params_data) >= 5:
                        f, cx, cy, k1, k2 = params_data[:5]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2}
                    else:
                        # Default to pinhole with reasonable parameters
                        print(f"Warning: Unknown or incomplete camera model {model} "
                              f"for camera {camera_id}, using defaults")
                        params = {'fx': width * 0.8, 'fy': height * 0.8, 'cx': width / 2, 'cy': height / 2}
                    
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }
                    
                except Exception as e:
                    print(f"Error parsing camera {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary cameras file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = cameras_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_cameras_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return cameras


def parse_colmap_images_bin(images_bin_path):
    """
    Parse COLMAP images.bin file.

    Returns:
        dict: image_name -> image_data dict
    """
    images = {}
    
    try:
        with open(images_bin_path, 'rb') as f:
            # Read number of images
            num_images_data = f.read(8)
            if len(num_images_data) < 8:
                print(f"Warning: Could not read number of images from {images_bin_path}")
                return images
            num_images = struct.unpack('<Q', num_images_data)[0]
            print(f"Reading {num_images} images from binary file")
            
            for i in range(num_images):
                try:
                    # Read image ID
                    image_id_data = f.read(4)
                    if len(image_id_data) < 4:
                        print(f"Warning: Could not read image ID for image {i}")
                        break
                    image_id = struct.unpack('<I', image_id_data)[0]
                    
                    # Read quaternion (qw, qx, qy, qz)
                    quat_data = f.read(32)  # 4 doubles = 32 bytes
                    if len(quat_data) < 32:
                        print(f"Warning: Could not read quaternion for image {image_id}")
                        break
                    qw, qx, qy, qz = struct.unpack('<dddd', quat_data)
                    
                    # Read translation (tx, ty, tz)
                    trans_data = f.read(24)  # 3 doubles = 24 bytes
                    if len(trans_data) < 24:
                        print(f"Warning: Could not read translation for image {image_id}")
                        break
                    tx, ty, tz = struct.unpack('<ddd', trans_data)
                    
                    # Read camera ID
                    camera_id_data = f.read(4)
                    if len(camera_id_data) < 4:
                        print(f"Warning: Could not read camera ID for image {image_id}")
                        break
                    camera_id = struct.unpack('<I', camera_id_data)[0]
                    
                    # Read image name (null-terminated string)
                    name_bytes = b''
                    while True:
                        byte = f.read(1)
                        if not byte:
                            print(f"Warning: Unexpected end of file reading name for image {image_id}")
                            break
                        if byte == b'\x00':
                            break
                        name_bytes += byte
                    
                    if not name_bytes and not byte == b'\x00':
                        print(f"Warning: Could not read name for image {image_id}")
                        break
                        
                    try:
                        name = name_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        print(f"Warning: Could not decode name for image {image_id}")
                        name = f"image_{image_id}"
                    
                    # Read number of 2D points
                    num_points_data = f.read(4)
                    if len(num_points_data) < 4:
                        print(f"Warning: Could not read num_points for image {image_id}")
                        break
                    num_points = struct.unpack('<I', num_points_data)[0]
                    
                    # Skip 2D points data (each point: x, y, point3d_id)
                    points_data_size = num_points * 24  # 3 doubles per point
                    skipped_data = f.read(points_data_size)
                    if len(skipped_data) < points_data_size:
                        print(f"Warning: Could not skip 2D points data for image {image_id}")
                        break
                    
                    images[name] = {
                        'image_id': image_id,
                        'camera_id': camera_id,
                        'quaternion': np.array([qw, qx, qy, qz]),
                        'translation': np.array([tx, ty, tz])
                    }
                    
                except Exception as e:
                    print(f"Error parsing image {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary images file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = images_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_images_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return images


def parse_colmap_points3D_txt(points3D_txt_path):
    """
    Parse COLMAP points3D.txt file.

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    points = []
    colors = []
    
    try:
        with open(points3D_txt_path, 'r') as f:
            lines = f.readlines()
        
        # Skip comments and header
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                i += 1
                continue
            
            if line.startswith('Number of points:'):
                i += 1
                continue
            
            # Parse point line: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            parts = line.split()
            if len(parts) < 8:
                i += 1
                continue
            
            # Extract X, Y, Z, R, G, B
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            
            points.append([x, y, z])
            colors.append([r, g, b])
            
            i += 1
    
    except Exception as e:
        print(f"Error parsing points3D.txt: {e}")
        return np.array([]), np.array([])
    
    return np.array(points), np.array(colors, dtype=np.uint8)


def parse_colmap_points3D_bin(points3D_bin_path):
    """
    Parse COLMAP points3D.bin file.

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    points = []
    colors = []
    
    try:
        with open(points3D_bin_path, 'rb') as f:
            # Read number of points
            num_points_data = f.read(8)
            if len(num_points_data) < 8:
                print(f"Warning: Could not read number of points from {points3D_bin_path}")
                return np.array([]), np.array([])
            num_points = struct.unpack('<Q', num_points_data)[0]
            print(f"Reading {num_points} points from binary file")
            
            for i in range(num_points):
                try:
                    # Read point ID (u64)
                    point_id_data = f.read(8)
                    if len(point_id_data) < 8:
                        print(f"Warning: Could not read point ID for point {i}")
                        break
                    # point_id = struct.unpack('<Q', point_id_data)[0]  # We don't need the ID
                    
                    # Read X, Y, Z (3 doubles)
                    xyz_data = f.read(24)
                    if len(xyz_data) < 24:
                        print(f"Warning: Could not read XYZ for point {i}")
                        break
                    x, y, z = struct.unpack('<ddd', xyz_data)
                    
                    # Read R, G, B (3 uint8)
                    rgb_data = f.read(3)
                    if len(rgb_data) < 3:
                        print(f"Warning: Could not read RGB for point {i}")
                        break
                    r, g, b = struct.unpack('<BBB', rgb_data)
                    
                    # Read error (double)
                    error_data = f.read(8)
                    if len(error_data) < 8:
                        print(f"Warning: Could not read error for point {i}")
                        break
                    # error = struct.unpack('<d', error_data)[0]  # We don't need the error
                    
                    # Read number of track elements
                    num_track_data = f.read(8)
                    if len(num_track_data) < 8:
                        print(f"Warning: Could not read num_track for point {i}")
                        break
                    num_track = struct.unpack('<Q', num_track_data)[0]
                    
                    # Skip track data (each track: image_id u32, point2d_idx u32)
                    track_data_size = num_track * 8  # 2 u32 per track
                    skipped_data = f.read(track_data_size)
                    if len(skipped_data) < track_data_size:
                        print(f"Warning: Could not skip track data for point {i}")
                        break
                    
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    
                except Exception as e:
                    print(f"Error parsing point {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary points3D file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = points3D_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_points3D_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return np.array(points), np.array(colors, dtype=np.uint8)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])


