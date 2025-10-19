import numpy as np
import toml
from utilities.parse_trc_file import trc_file_to_structured_points_3d


# TODO: TO EDIT TO TRC 21 KEYPOINTS
def get_keypoint_3d(frame_keypoints, keypoint_name):
    """
    Retrieve the 3D coordinate for a given keypoint name from a TRC file frame.
    
    Args:
        frame_keypoints (np.ndarray): Array of keypoints for a single frame.
        keypoint_name (str): Name of the keypoint (e.g., "RHeel", "RKnee", etc.)
        
    Returns:
        np.ndarray or None: 3D coordinate as an array, or None if the keypoint is not available.
    """
    keypoint_indices_map = {
        "CHip": 0,
        "RHip": 1,
        "RKnee": 2,
        "RAnkle": 3,
        "RBigToe": 4,
        "RSmallToe": 5,
        "RHeel": 6,
        "LHip": 7,
        "LKnee": 8,
        "LAnkle": 9,
        "LBigToe": 10,
        "LSmallToe": 11,
        "LHeel": 12,
        "Neck": 13,
        "Nose": 14,
        "RShoulder": 15,
        "RElbow": 16,
        "RWrist": 17,
        "LShoulder": 18,
        "LElbow": 19,
        "LWrist": 20
    }
    
    idx = keypoint_indices_map.get(keypoint_name, None)
    if idx is not None and idx < len(frame_keypoints):
        keypoint = frame_keypoints[idx, 0]
        
        # Check if any coordinate is 0.0 (indicating missing data)
        if keypoint[0, 0] == 0.0 and keypoint[1, 0] == 0.0 and keypoint[2, 0] == 0.0:
            return None
        
        return keypoint
    else:
        return None

def compute_segment_length(structured_points_3d, kp1_name, kp2_name):
    """
    Compute the average Euclidean distance between two keypoints across all frames.
    
    Args:
        structured_points_3d (list): List of frames, where each frame contains keypoint data.
        kp1_name (str): Name of the first keypoint.
        kp2_name (str): Name of the second keypoint.
        
    Returns:
        float or None: Average Euclidean distance between the two keypoints, or None if no valid measurements.
    """
    distances = []
    
    for frame_keypoints in structured_points_3d:
        pt1 = get_keypoint_3d(frame_keypoints, kp1_name)
        pt2 = get_keypoint_3d(frame_keypoints, kp2_name)
        
        if pt1 is not None and pt2 is not None:
            # Convert points to flat arrays for distance calculation
            p1_flat = np.array([pt1[0, 0], pt1[1, 0], pt1[2, 0]])
            p2_flat = np.array([pt2[0, 0], pt2[1, 0], pt2[2, 0]])
            
            distance = np.linalg.norm(p1_flat - p2_flat)
            distances.append(distance)
    
    # Return average distance if we have valid measurements
    if distances:
        avg_distance = sum(distances) / len(distances)
        print(f"Segment {kp1_name}-{kp2_name}: {len(distances)} valid frames, avg distance: {avg_distance:.4f}")
        return avg_distance
    else:
        print(f"Segment {kp1_name}-{kp2_name}: No valid measurements found")
        return None

def compute_scale_factor_from_segments(structured_points_3d, known_segments):
    """
    Compute an absolute scale factor by comparing the measured 3D segment lengths
    to their known real-world lengths across all frames.
    
    Args:
        structured_points_3d (list): List of frames, where each frame contains keypoint data.
        known_segments (list of tuples): Each tuple is (kp1, kp2, known_length) where:
                                       - kp1 and kp2 are keypoint names
                                       - known_length is the real-world length in meters.
    
    Returns:
        float or None: Averaged scale factor to convert 3D reconstruction into metric units,
                     or None if no valid segments were measured.
    """
    scale_factors = []
    
    print(f"Computing scale factors from {len(known_segments)} known segments:")
    
    for kp1, kp2, known_length in known_segments:
        measured_length = compute_segment_length(structured_points_3d, kp1, kp2)
        
        if measured_length is not None and measured_length > 0:
            scale_factor = known_length / measured_length
            scale_factors.append(scale_factor)
            print(f"  {kp1}-{kp2}: known={known_length:.3f}m, measured={measured_length:.3f}, scale={scale_factor:.5f}")
    
    if scale_factors:
        avg_scale = sum(scale_factors) / len(scale_factors)
        print(f"Average scale factor: {avg_scale:.5f} (from {len(scale_factors)} segments)")
        return avg_scale
    else:
        print("No valid scale factors could be computed")
        return None

def load_segments_from_toml(toml_file):
    """
    Load known body segments from a TOML file. Based on Dr. David Pagnon's algorithm.
    
    Args:
        toml_file (str): Path to the TOML file containing segment definitions.
        
    Returns:
        list: List of tuples (kp1, kp2, length) for each defined segment.
    """
    print(f"Loading segment definitions from {toml_file}")
    if toml_file is None:
        print("No segments file provided, using defaults")
        return get_default_segments()
        
    try:
        config = toml.load(toml_file)
        print(f"TOML content: {config}")  # Debug: Print the loaded TOML content
    except Exception as e:
        print(f"Error loading segments TOML file: {e}")
        return get_default_segments()
    
    known_segments = []
    
    # Check for segments section
    if "segments" not in config:
        print("No 'segments' section found in TOML file, using defaults")
        return get_default_segments()
    
    segments_section = config["segments"]
    
    # Handle array of tables format [[segments]]
    if isinstance(segments_section, list):
        print("Detected array of tables format")
        for segment in segments_section:
            if isinstance(segment, dict) and "keypoint1" in segment and "keypoint2" in segment and "length" in segment:
                known_segments.append((
                    segment["keypoint1"],
                    segment["keypoint2"],
                    float(segment["length"])
                ))
    
    # Handle nested dictionary format [segments.segment1]
    elif isinstance(segments_section, dict):
        print("Detected nested dictionary format")
        for segment_name, segment_data in segments_section.items():
            if isinstance(segment_data, dict) and "keypoint1" in segment_data and "keypoint2" in segment_data and "length" in segment_data:
                known_segments.append((
                    segment_data["keypoint1"],
                    segment_data["keypoint2"],
                    float(segment_data["length"])
                ))
    
    if not known_segments:
        print("No valid segments found in the TOML file, using defaults")
        return get_default_segments()
    
    print(f"Successfully loaded {len(known_segments)} segment definitions:")
    for i, (kp1, kp2, length) in enumerate(known_segments):
        print(f"  {i+1}. {kp1} to {kp2}: {length}m")
    
    return known_segments


def get_default_segments():
    """
    Provide default body segment definitions if none are available from config.
    
    Returns:
        list: List of default segment tuples (keypoint1, keypoint2, length).
    """
    default_height = 1.7  # Default human height in meters
    # TODO: Gonna have to adjust these because halpe 26 has a head keypoint
    return [
        ('RHeel', 'Nose', default_height - 0.12), # 0.12 pertains to average distance of nose to the top of head
        ('RElbow', 'RWrist', 0.25),
        ('RHip', 'RKnee', 0.47),
        ('LHeel', 'Nose', default_height - 0.12),
        ('LElbow', 'LWrist', 0.25),
        ('LHip', 'LKnee', 0.47)
    ]


#this doesn't need to be here

def calculate_scale_factor( segments_file, trc_file):
    """
    Apply scale calibration to convert from arbitrary to metric units.
    
    Args:
        all_best_results (dict): Dictionary of camera parameters.
        final_idx_of_ref_cam (int): Index of the reference camera.
        segments_file (str): Path to segments definition file.
        Ks (list): List of intrinsic matrices.
        
    Returns:
        dict: Dictionary of scaled camera parameters.
    """
    # Load known segments from file
    known_segments = load_segments_from_toml(segments_file)
    

    structure_points_3d = trc_file_to_structured_points_3d(trc_file)
    
    # Compute and apply scale
    scale_factor = compute_scale_factor_from_segments(structure_points_3d, known_segments)
    print(f"Computed scale factor: {scale_factor}")
    
    if scale_factor is None or scale_factor <= 0:
        print("Warning: Invalid scale factor. Using default scale of 1.0")
        scale_factor = 1.0
    return scale_factor
    



# def apply_scale_to_results(all_best_results, scale_factor, ref_cam_idx):
#     """
#     Apply absolute scale to camera parameters by scaling only the z-component 
#     of the translation vectors (Tz), leaving Tx and Ty unchanged.
    
#     Args:
#         all_best_results (dict): Dictionary containing camera calibration results.
#         scale_factor (float): Scale factor to convert to metric units.
#         ref_cam_idx (int): Index of reference camera.
        
#     Returns:
#         dict: Updated calibration results with the z-component scaled.
#     """
#     scaled_results = {}
    
#     for pair_key, params in all_best_results.items():
#         scaled_results[pair_key] = params.copy()
        
#         # Scale translation vectors for all cameras except the reference
#         if pair_key.startswith(f"Camera{ref_cam_idx}_"):
#             t = np.array(params['t'])  # ensure it's a numpy array
#             # Only scale the z component (index 2)
#             t[2] = t[2] * scale_factor
#             scaled_results[pair_key]['t'] = t
            
#     return scaled_results


def apply_scale_to_results(all_best_results, scale_factor, ref_cam_idx):
    """
    Apply absolute scale to camera parameters by scaling all components
    of the translation vectors uniformly (tx, ty, tz).
    
    Args:
        all_best_results (dict): Dictionary containing camera calibration results.
        scale_factor (float): Scale factor to convert to metric units.
        ref_cam_idx (int): Index of reference camera.
        
    Returns:
        dict: Updated calibration results with scaled translations.
    """
    scaled_results = {}
    
    for pair_key, params in all_best_results.items():
        scaled_results[pair_key] = params.copy()
        
        # Skip pairs that don't involve the reference camera
        if f"Camera{ref_cam_idx}" not in pair_key:
            continue
            
        # Scale the entire translation vector uniformly
        t = np.array(params['t'])  # ensure it's a numpy array
        t = t * scale_factor
        scaled_results[pair_key]['t'] = t
            
    return scaled_results
