import glob
import os
import json
import numpy as np
import toml


# TODO: Move to Loader.py file
def load_json_files(cam_dirs):
    """
    Load all JSON files from the given directories.

    Args:
        cam_dirs (list): List of directories containing JSON files for cameras.

    Returns:
        list: A list containing data loaded from JSON files for each camera.
    """
    print("Loading JSON files...")
    all_cam_data = []
    for cam_dir in cam_dirs:
        cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        cam_data = []
        for cam_file in cam_files:
            with open(cam_file, 'r') as file:
                data = json.load(file)
                cam_data.append(data)
        all_cam_data.append(cam_data)
    return all_cam_data

# TODO: Move to Loader.py file
def unpack_keypoints(paired_keypoints_list):
    """
    Unpacks the paired keypoints from a list of paired keypoints.

    Args:
        paired_keypoints_list (list): List of paired keypoints.

    Returns:
        tuple: A tuple containing two lists, where each list contains the x and y coordinates of the keypoints.
    """
    points1, points2 = [], []
    
    for frame in paired_keypoints_list:
        for point in frame:
            if len(point) == 2:
                u1, v1 = point[0]
                u2, v2 = point[1]
                points1.append((u1, v1))  
                points2.append((u2, v2))
    
    print(f"Shape of points1: {np.array(points1).shape}")
    print(f"Shape of points2: {np.array(points2).shape}")

    return points1, points2

# TODO: Move to Loader.py file
def extract_individual_camera_keypoints(paired_keypoints_list):
    """
    Extracts individual camera keypoints from a list of paired keypoints.

    Args:
        paired_keypoints_list (list): A list of paired keypoints.

    Returns:
        dict: A dictionary containing the keypoints for each camera,
              where the keys are the camera indices and the values 
              are lists of keypoints for each frame.
    """
    other_cameras_keypoints = {}
    for i, camera_pair in enumerate(paired_keypoints_list):
        other_camera_index = i  # camera index 
        other_cameras_keypoints[other_camera_index] = []

        for frame in camera_pair:
            frame_keypoints_other_camera = []
            for keypoints_pair in frame:
                if len(keypoints_pair) == 2:
                    frame_keypoints_other_camera.append(keypoints_pair[1])
                    
            other_cameras_keypoints[other_camera_index].append(frame_keypoints_other_camera)
            
    return other_cameras_keypoints



# TODO: Move to Loader.py file
def load_intrinsics_from_toml(toml_file):
    """
    Load intrinsic camera parameters from a TOML file.
    
    Args:
        toml_file (str): Path to the TOML file containing camera intrinsic parameters.
        
    Returns:
        tuple: A tuple containing:
            - list: List of intrinsic matrices (K) for each camera.
            - list: List of image dimensions [width, height].
    """
    print(f"Loading intrinsic parameters from {toml_file}")
    try:
        config = toml.load(toml_file)
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        return None, None
    
    # Initialize lists to store intrinsic matrices and image dimensions
    Ks = []
    image_size = None
    
    # Extract all camera sections
    camera_sections = []
    for section_name, section_data in config.items():
        if section_name != "metadata" and isinstance(section_data, dict):
            if "matrix" in section_data:
                camera_sections.append(section_data)
    
    # Sort camera sections if necessary
    # You might want to sort them based on camera name or some other criteria
    
    # Extract intrinsic matrices from each camera section
    for camera_section in camera_sections:
        matrix = np.array(camera_section["matrix"])
        Ks.append(matrix)
        
        # If we haven't set the image size yet, use the first camera's size
        if image_size is None and "size" in camera_section:
            image_size = camera_section["size"]
    
    if not Ks:
        print("No camera intrinsic matrices found in the TOML file")
        return None, None
    
    if image_size is None:
        print("No image size found in the TOML file")
        return None, None
    
    print(f"Loaded {len(Ks)} intrinsic matrices")
    return Ks, image_size


def get_latest_trc_file(trc_file_dir):
    """Get the most recently modified TRC file in the directory."""
    trc_files = glob.glob(os.path.join(trc_file_dir, "*.trc"))
    if not trc_files:
        return None
    # Sort by modification time (newest last)
    latest_file = max(trc_files, key=os.path.getmtime)
    return latest_file
