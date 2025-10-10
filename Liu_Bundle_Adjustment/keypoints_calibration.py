
"""

This script is for calibration. Inputs would be the precalculated intrinsics values from a calibration toml file.
The output is a calibration toml file with the calculated extrinsics value.
The original script is based on Mr. Hunminkim's implementation of Liu's paper with some added as suggested by Dr. Pagnon.  
List of things TODO
1. Translate plane into -x
2. Create a callback function that saves the best extrinsics values (there might've been better local minima passed)
3. Integrate an option for CasCalib's estimation of intrinsic values or Find another good estimate of intrinsic values
"""




import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import time
import os
import json
import glob
import argparse
import toml
import tempfile


from pathlib import Path
from scipy.optimize import least_squares
import io
from contextlib import redirect_stdout
from Liu_Bundle_Adjustment.calculate_scale import calculate_scale_factor
from Pose2Sim import Pose2Sim
from utilities.trc_Xup_to_Yup import trc_Xup_to_Yup_func
from utilities.OpenPose_to_AlphaPose import OpenPose_to_AlphaPose_func
from Liu_Bundle_Adjustment.keypoints_confidence_multi import extract_paired_keypoints_with_reference
from utilities.write_to_toml import write_to_toml


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


def create_paired_inlier(inliers1, inliers2):
    """
    Creates a list of paired inliers.

    Args:
        inliers1 (numpy.ndarray): Array of inlier points from camera 1.
        inliers2 (numpy.ndarray): Array of inlier points from camera 2.

    Returns:
        list: List of tuples, where each tuple contains paired points (tuples), 
              where each sub-tuple is a point (x, y) from camera 1 and camera 2 respectively.
    """
    paired_inliers = [((p1[0], p1[1]), (p2[0], p2[1])) for p1, p2 in zip(inliers1, inliers2)]
    return paired_inliers


def compute_fundamental_matrix(paired_keypoints_list):
    """
    Compute the fundamental matrix from paired keypoints and return inlier keypoints.

    This function takes a list of paired keypoints and computes the fundamental matrix using the RANSAC algorithm.
    It also filters out outliers based on the RANSAC result.

    Args:
        paired_keypoints_list (list): A list of tuples, where each tuple contains two arrays of keypoints, 
                                      one for each image.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The computed fundamental matrix.
            - numpy.ndarray: Points from the first image that are considered inliers.
            - numpy.ndarray: Points from the second image that are considered inliers.
    """
    points1, points2 = unpack_keypoints(paired_keypoints_list)
    
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)

    # Compute the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # Filter points based on the mask
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]

    return F, inliers1, inliers2


def compute_essential_matrix(F, K1, K2):
    """
    Compute the essential matrix given the fundamental matrix and camera calibration matrices.

    Args:
        F (numpy.ndarray): The fundamental matrix.
        K1 (numpy.ndarray): The calibration matrix of camera 1.
        K2 (numpy.ndarray): The calibration matrix of other camera.

    Returns:
        numpy.ndarray: The computed essential matrix.
    """
    E = K2.T @ F @ K1
    return E


def recover_pose_from_essential_matrix(E, points1_inliers, points2_inliers, K):
    """
    Recover the camera pose from the Essential matrix using inliers.

    Args:
        E (numpy.ndarray): The Essential matrix.
        points1_inliers (numpy.ndarray): The inlier points from the first image.
        points2_inliers (numpy.ndarray): The inlier points from the second image.
        K (numpy.ndarray): The camera intrinsic matrix (assuming the same for both cameras).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The rotation matrix (R).
            - numpy.ndarray: The translation vector (t).
            - numpy.ndarray: The mask used for recovering the pose.
    """
    # Ensure points are in the correct shape and type
    points1_inliers = points1_inliers.astype(np.float32)
    points2_inliers = points2_inliers.astype(np.float32)

    # Recovering the pose
    _, R, t, mask = cv2.recoverPose(E, points1_inliers, points2_inliers, K)

    return R, t, mask


def cam_create_projection_matrix(K, R, t):
    """
    Creates the camera projection matrix.

    Args:
        K (numpy.ndarray): The camera's intrinsic parameters matrix.
        R (numpy.ndarray): The rotation matrix.
        t (numpy.ndarray): The translation vector.

    Returns:
        numpy.ndarray: The created projection matrix.
    """
    RT = np.hstack([R, t.reshape(-1, 1)])
    return K @ RT


def triangulate_points(paired_keypoints_list, P1, P2):
    """
    Triangulates a list of paired keypoints using the given camera projection matrices.

    Args:
        paired_keypoints_list (list): List of paired keypoints, where each item is a tuple containing 
                                      two sets of coordinates for the same keypoint observed in both cameras.
        P1 (numpy.ndarray): Camera projection matrix for the reference camera.
        P2 (numpy.ndarray): Camera projection matrix for the other camera.

    Returns:
        list: List of 3D points corresponding to the triangulated keypoints.
    """
    points_3d = []

    for keypoint_pair in paired_keypoints_list:
        (x1, y1), (x2, y2) = keypoint_pair

        # Convert coordinates to homogeneous format for triangulation
        point_3d_homogeneous = cv2.triangulatePoints(P1, P2, 
                                                     np.array([[x1], [y1]], dtype=np.float64), 
                                                     np.array([[x2], [y2]], dtype=np.float64))

        # Normalize to convert to non-homogeneous 3D coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        points_3d.append(point_3d)

    return points_3d


def plot_3d_points(points_3d):
    """
    Plots a set of 3D points.

    Args:
        points_3d (list): List of 3D points represented as (x, y, z) coordinates.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point in points_3d:
        ax.scatter(point[0], point[1], point[2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def compute_reprojection_error(precomputed_points_3d, keypoints_detected, P1, P2):
    """
    Computes the reprojection error for a set of paired keypoints using the given projection matrices
    and precomputed 3D points.

    Args:
        precomputed_points_3d (list): List of precomputed 3D points as NumPy arrays.
        keypoints_detected (list): List of paired keypoints, each represented as a tuple 
                                  (2D point in camera 1, 2D point in camera 2).
        P1 (numpy.ndarray): Camera projection matrix for the reference camera.
        P2 (numpy.ndarray): Camera projection matrix for the other camera.

    Returns:
        float: The mean reprojection error over all keypoints.
    """
    total_error = 0
    total_points = 0

    # Ensure the length of 3D points matches the 2D keypoints
    assert len(precomputed_points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    # Process each pair of 3D point and 2D keypoints
    for point_3d, (point1, point2) in zip(precomputed_points_3d, keypoints_detected):
        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d.flatten(), 1)

        # Reproject the 3D point to the 2D image plane for both cameras
        point1_reprojected = P1 @ point_3d_homogeneous
        point1_reprojected /= point1_reprojected[2]

        point2_reprojected = P2 @ point_3d_homogeneous
        point2_reprojected /= point2_reprojected[2]

        # Compute reprojection errors for each camera's reprojected point
        error1 = np.linalg.norm(point1_reprojected[:2] - np.array(point1))
        error2 = np.linalg.norm(point2_reprojected[:2] - np.array(point2))

        total_error += error1 + error2
        total_points += 2

    mean_error = total_error / total_points if total_points > 0 else 0
    return mean_error


def vectorize_params_for_intrinsic_loss(points_3d, keypoints_detected, R, t):
    """
    Vectorizes parameters for the intrinsic loss calculation.
    
    Args:
        points_3d (list): List of 3D points.
        keypoints_detected (list): List of detected 2D keypoints.
        R (numpy.ndarray): Rotation matrix.
        t (numpy.ndarray): Translation vector.
        
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: u coordinates of detected keypoints.
            - numpy.ndarray: v coordinates of detected keypoints.
            - numpy.ndarray: X coordinates in camera space.
            - numpy.ndarray: Y coordinates in camera space.
            - numpy.ndarray: Z coordinates in camera space.
    """
    # Initialize arrays
    u_detected = []
    v_detected = []
    Xc = []
    Yc = []
    Zc = []
    transformation_matrix = np.hstack((R, t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix

    # Make sure the number of 3D points matches the 2D keypoints
    assert len(points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    for point_3d, detected_point in zip(points_3d, keypoints_detected):
        if not isinstance(detected_point, (list, tuple, np.ndarray)) or len(detected_point) != 2:
            continue
        # detected point = (u, v)
        u, v = detected_point
        u_detected.append(u)
        v_detected.append(v)

        # world to camera transformation
        point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates
        point_camera = transformation_matrix.dot(point_3d_homogeneous)
        X, Y, Z = point_camera[:3]
        Xc.append(X)
        Yc.append(Y)
        Zc.append(Z)

    return np.array(u_detected), np.array(v_detected), np.array(Xc), np.array(Yc), np.array(Zc)


def compute_intrinsics_optimization_loss(x, u_detected, v_detected, Xc, Yc, Zc, u0, v0):
    """
    Computes the loss for the intrinsic parameters optimization.

    Args:
        x (numpy.ndarray): Intrinsic parameters to optimize (f_x, f_y).
        u_detected (numpy.ndarray): Detected u coordinates.
        v_detected (numpy.ndarray): Detected v coordinates.
        Xc (numpy.ndarray): X coordinates in camera space.
        Yc (numpy.ndarray): Y coordinates in camera space.
        Zc (numpy.ndarray): Z coordinates in camera space.
        u0 (float): Principal point u-coordinate.
        v0 (float): Principal point v-coordinate.

    Returns:
        float: The mean loss for the intrinsic parameters optimization.
    """
    f_x, f_y = x  # Intrinsic parameters to optimize
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)

    valid_keypoints_count = Xc.shape[0]
    
    loss = np.abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + np.abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
    total_loss = np.sum(loss)
   
    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0
    print(f"Mean loss of intrinsic: {mean_loss}")
    return mean_loss


def optimize_intrinsic_parameters(points_3d, keypoints_detected, K, R, t, tolerance_list, u0, v0):
    """
    Optimizes the intrinsic parameters using the given 3D points and detected keypoints.

    Args:
        points_3d (list): List of 3D points (triangulated human body joints).
        keypoints_detected (list): Original detected 2D keypoints.
        K (numpy.ndarray): Intrinsic parameters matrix.
        R (numpy.ndarray): Rotation matrix.
        t (numpy.ndarray): Translation vector.
        tolerance_list (dict): Dictionary with optimization parameters like ftol, xtol, etc.
        u0 (float): Principal point u-coordinate.
        v0 (float): Principal point v-coordinate.

    Returns:
        numpy.ndarray: The optimized intrinsic parameters matrix.
    """
    # Create the initial guess for the intrinsic parameters
    x0 = np.array([K[0, 0], K[1, 1]])
    u_detected, v_detected, Xc, Yc, Zc = vectorize_params_for_intrinsic_loss(points_3d, keypoints_detected, R, t)
    
    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(
        compute_intrinsics_optimization_loss, 
        x0, 
        args=(u_detected, v_detected, Xc, Yc, Zc, u0, v0), 
        x_scale='jac', 
        verbose=1, 
        method='trf', 
        loss='huber', 
        diff_step=tolerance_list['diff_step'], 
        tr_solver='lsmr', 
        ftol=tolerance_list['ftol'], 
        max_nfev=tolerance_list['max_nfev'], 
        xtol=tolerance_list['xtol'], 
        gtol=tolerance_list['gtol']
    )

    # Create the optimized intrinsic matrix
    K_optimized = np.array([[result.x[0], 0, u0], [0, result.x[1], v0], [0, 0, 1]])

    return K_optimized


def vectorize_params_for_extrinsic_loss(points_3d, points_2d):
    """
    Vectorize parameters for extrinsic loss calculation.
    
    Args:
        points_3d (list): List of 3D points.
        points_2d (list): List of 2D points.
        
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: u coordinates of detected keypoints.
            - numpy.ndarray: v coordinates of detected keypoints.
            - numpy.ndarray: Homogeneous 3D points.
    """
    u_detected_list = []
    v_detected_list = []
    point_3d_homogeneous_list = []

    # Convert to numpy arrays
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Check if points_3d and points_2d are not empty
    if points_3d.size == 0 or points_2d.size == 0:
        print("Warning: Empty points_3d or points_2d input.")
        return np.array(u_detected_list), np.array(v_detected_list), np.array(point_3d_homogeneous_list)

    # Flatten the points_3d array to match the expected shape (N, 3)
    points_3d_flat = points_3d.reshape(-1, 3)

    for point_3d, detected_point in zip(points_3d_flat, points_2d):
        if len(detected_point) != 2:
            continue

        u_detected, v_detected = detected_point

        # World to camera transformation
        point_3d_homogeneous = np.append(point_3d, 1)  # Convert to homogeneous coordinates

        u_detected_list.append(u_detected)
        v_detected_list.append(v_detected)
        point_3d_homogeneous_list.append(point_3d_homogeneous)

    return np.array(u_detected_list), np.array(v_detected_list), np.array(point_3d_homogeneous_list)


def compute_extrinsics_optimization_loss(x, ext_K, u_detected, v_detected, point_3d_homogeneous, isConstrained, h, ext_R):
    """
    Computes the loss for the extrinsic parameters optimization.

    Args:
        x (numpy.ndarray): Translation vector to optimize.
        ext_K (numpy.ndarray): Camera intrinsic matrix.
        u_detected (numpy.ndarray): Detected u coordinates.
        v_detected (numpy.ndarray): Detected v coordinates.
        point_3d_homogeneous (numpy.ndarray): Homogeneous 3D points.
        isConstrained (bool): Whether to apply constraint on translation magnitude.
        h (float): Constraint coefficient.
        ext_R (numpy.ndarray): Rotation matrix.

    Returns:
        float: The mean loss for the extrinsic parameters optimization.
    """
    f_x, f_y, u0, v0 = ext_K[0, 0], ext_K[1, 1], ext_K[0, 2], ext_K[1, 2]
    dx = 1.0  # Pixel scaling factor dx (assumed to be 1 if not known)
    dy = 1.0  # Pixel scaling factor dy (assumed to be 1 if not known)
    obj_t = x
    transformation_matrix = np.hstack((ext_R, obj_t.reshape(-1, 1)))  # transformation matrix
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))  # homogeneous transformation matrix

    # transformation_matrix is a 2D array of shape (4, 4) and 
    # point_3d_homogeneous is a 2D array of shape (N, 4), where N is the number of points, we can do dot product
    point_camera = np.dot(point_3d_homogeneous, transformation_matrix.T)

    # point_camera is a 2D array of shape (N, 4). Xc, Yc, Zc are all 1D np arrays
    Xc, Yc, Zc, _ = point_camera.T
    valid_keypoints_count = Xc.shape[0]
    
    # Compute reprojection loss
    loss = np.abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + np.abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
    total_loss = np.sum(loss)

    # Add constraint term if isConstrained is True (for camera 1)
    if isConstrained:
        translation_magnitude = np.linalg.norm(obj_t)  # |T|
        constraint_term = h * np.abs(translation_magnitude - 1)  # h(|T| - 1)
        total_loss += constraint_term
   
    if valid_keypoints_count > 0:
        mean_loss = total_loss / valid_keypoints_count
    else:
        mean_loss = 0
    print(f"Mean loss of extrinsic: {mean_loss}")
    return mean_loss


def optimize_extrinsic_parameters(points_3d, other_cameras_keypoints, ext_K, ext_R, ext_t, tolerance_list, isConstrained=False, h=1.0):
    """
    Optimizes the extrinsic parameters using the given 3D points and detected keypoints.

    Args:
        points_3d (list): List of 3D points (triangulated human body joints).
        other_cameras_keypoints (list): Original detected 2D keypoints for the other cameras.
        ext_K (numpy.ndarray): Intrinsic parameters matrix.
        ext_R (numpy.ndarray): Rotation matrix.
        ext_t (numpy.ndarray): Initial translation vector.
        tolerance_list (dict): Dictionary with optimization parameters like ftol, xtol, etc.
        isConstrained (bool, optional): Whether to apply constraint on translation magnitude. Defaults to False.
        h (float, optional): Constraint coefficient. Defaults to 1.0.

    Returns:
        numpy.ndarray: The optimized translation vector.
    """
    # Create the initial guess for the extrinsic parameters
    x0 = ext_t.flatten()
    print(f"Initial x0: {x0}")
    u_detected, v_detected, point_3d_homogeneous = vectorize_params_for_extrinsic_loss(points_3d, other_cameras_keypoints)
    
    # Optimize the extrinsic parameters using the least squares method
    result = least_squares(
        compute_extrinsics_optimization_loss, 
        x0, 
        args=(ext_K, u_detected, v_detected, point_3d_homogeneous, isConstrained, h, ext_R), 
        x_scale='jac', 
        verbose=1, 
        method='trf', 
        loss='huber', 
        diff_step=tolerance_list['diff_step'], 
        tr_solver='lsmr', 
        ftol=tolerance_list['ftol'], 
        max_nfev=tolerance_list['max_nfev'], 
        xtol=tolerance_list['xtol'], 
        gtol=tolerance_list['gtol']
    )

    optimized_t = result.x  # optimized t vector
    print(f"Optimized t: {optimized_t}")

    return optimized_t

# TODO: REMOVE ONCE MIGRATED
def get_keypoint_3d(points_3d, keypoint_name):
    """
    Retrieve the 3D coordinate for a given keypoint name from a BODY_25 set.
    
    For "Head", we define it as the average of "Nose", "REye", and "LEye".
    
    Args:
        points_3d (list or np.ndarray): List/array of 3D keypoints.
        keypoint_name (str): Name of the keypoint (e.g., "RHeel", "Head", etc.)
        
    Returns:
        np.ndarray or None: 3D coordinate as an array, or None if the keypoint is not available.
    """
    keypoint_indices_map = {
         "Nose": 0,
         "Neck": 1,
         "RShoulder": 2,
         "RElbow": 3,
         "RWrist": 4,
         "LShoulder": 5,
         "LElbow": 6,
         "LWrist": 7,
         "MidHip": 8,
         "RHip": 9,
         "RKnee": 10,
         "RAnkle": 11,
         "LHip": 12,
         "LKnee": 13,
         "LAnkle": 14,
         "REye": 15,
         "LEye": 16,
         "REar": 17,
         "LEar": 18,
         "LBigToe": 19,
         "LSmallToe": 20,
         "LHeel": 21,
         "RBigToe": 22,
         "RSmallToe": 23,
         "RHeel": 24,
         "Background": 25
    }
    
    if keypoint_name == "Head":
        # Define "Head" as the average of "Nose", "REye", and "LEye"
        indices = [keypoint_indices_map["Nose"], keypoint_indices_map["REye"], keypoint_indices_map["LEye"]]
        head_points = [points_3d[i] for i in indices if i < len(points_3d)]
        if head_points:
            return np.mean(head_points, axis=0)
        else:
            return None
    else:
        idx = keypoint_indices_map.get(keypoint_name, None)
        if idx is not None and idx < len(points_3d):
            return points_3d[idx]
        else:
            return None
# TODO: REMOVE ONCE MIGRATED
def get_keypoint_3d_BODY_25B(points_3d, keypoint_name):
    """
    Retrieve the 3D coordinate for a given keypoint name from a BODY_25B set.
    
    Args:
        points_3d (list or np.ndarray): List/array of 3D keypoints.
        keypoint_name (str): Name of the keypoint (e.g., "RHeel", "Head", etc.)
        
    Returns:
        np.ndarray or None: 3D coordinate as an array, or None if the keypoint is not available.
    """
    keypoint_indices_map = {
        "Nose": 0,
        "LHip": 11,
        "RHip": 12,
        "LKnee": 13,
        "RKnee": 14,
        "LAnkle": 15,
        "RAnkle": 16,
        "Neck": 17,
        "Head": 18,
        "LBigToe": 19,
        "LSmallToe": 20, 
        "LHeel": 21,
        "RBigToe": 22,
        "RSmallToe": 23, 
        "RHeel": 24,
        "LShoulder": 5,
        "RShoulder": 6,
        "LElbow": 7,
        "RElbow": 8,
        "LWrist": 9,
        "RWrist": 10,
        "CHip": None  # Center hip doesn't have a specific index in the points_3d array
    }
    
    idx = keypoint_indices_map.get(keypoint_name, None)
    if idx is not None and idx < len(points_3d):
        return points_3d[idx]
    elif keypoint_name == "CHip" and len(points_3d) > max(keypoint_indices_map["LHip"], keypoint_indices_map["RHip"]):
        # Calculate CHip as the midpoint between left and right hip
        left_hip = points_3d[keypoint_indices_map["LHip"]]
        right_hip = points_3d[keypoint_indices_map["RHip"]]
        return (left_hip + right_hip) / 2
    else:
        return None

# TODO: REMOVE ONCE MIGRATED
def compute_segment_length(points_3d, kp1_name, kp2_name):
    """
    Compute the Euclidean distance between two keypoints.
    
    Args:
        points_3d (list or np.ndarray): List/array of 3D keypoints.
        kp1_name (str): Name of the first keypoint.
        kp2_name (str): Name of the second keypoint.
        
    Returns:
        float or None: Euclidean distance between the two keypoints, or None if either keypoint is missing.
    """
    pt1 = get_keypoint_3d(points_3d, kp1_name)
    pt2 = get_keypoint_3d(points_3d, kp2_name)
    if pt1 is not None and pt2 is not None:
        return np.linalg.norm(pt1 - pt2)
    else:
        return None

#TODO Remove once migrated
def compute_scale_factor_from_segments(points_3d, known_segments):
    """
    Compute an absolute scale factor by comparing the measured 3D segment lengths
    to their known real-world lengths.
    
    Args:
        points_3d (list or np.ndarray): Triangulated 3D keypoints.
        known_segments (list of tuples): Each tuple is (kp1, kp2, known_length) where:
                                         - kp1 and kp2 are keypoint names
                                         - known_length is the real-world length in meters.
    
    Returns:
        float or None: Averaged scale factor to convert 3D reconstruction into metric units,
                       or None if no valid segments were measured.
    """
    scale_factors = []
    for kp1, kp2, known_length in known_segments:
        measured_length = compute_segment_length(points_3d, kp1, kp2)
        if measured_length is not None and measured_length > 0:
            scale_factor = known_length / measured_length
            scale_factors.append(scale_factor)
    if scale_factors:
        return sum(scale_factors) / len(scale_factors)
    else:
        return None

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

# TODO: REMOVE ONCE MIGRATED
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
# TODO: REMOVE ONCE MIGRATED
def get_default_segments():
    """
    Provide default body segment definitions if none are available from config.
    
    Returns:
        list: List of default segment tuples (keypoint1, keypoint2, length).
    """
    default_height = 1.7  # Default human height in meters
    
    return [
        ('RHeel', 'Head', default_height),
        ('RElbow', 'RWrist', 0.25),
        ('RHip', 'RKnee', 0.47),
        ('LHeel', 'Head', default_height),
        ('LElbow', 'LWrist', 0.25),
        ('LHip', 'LKnee', 0.47)
    ]


def select_reference_camera(all_cam_data, camera_dirs, Ks, confidence_threshold):
    """
    Find the best reference camera based on keypoint visibility and reprojection error.
    
    Args:
        all_cam_data (list): List of loaded camera data.
        camera_dirs (list): List of camera directories.
        Ks (list): List of intrinsic matrices for each camera.
        confidence_threshold (float): Confidence threshold for keypoint filtering.
        
    Returns:
        tuple: A tuple containing:
            - int: Index of the best reference camera.
            - int: Index of the constrained camera.
            - list: List of paired keypoints for the best reference camera.
            - list: List of paired inliers for all camera pairs.
            - list: List of inlier points for secondary cameras.
            - dict: Dictionary of camera R and t values.
    """
    camera_Rt_list = [{} for _ in camera_dirs]
    valid_ref_cam_idx = []
    average_scores_for_valid_ref_cam = []
    temp_paired_keypoints_global_list = []
    inliers_pair_list = []
    inlier2_list = []
    
    # Find the minimum number of frames across all cameras
    min_frames = min(len(cam_data) for cam_data in all_cam_data)
    for i in range(len(all_cam_data)):
        all_cam_data[i] = all_cam_data[i][:min_frames]
    
    print("Finding the best reference camera...")
    for i, cam_dir in enumerate(camera_dirs):
        no_R_t_solutions_flag = False
        zero_keypoints_flag = False
        total_score = 0
        print(f"Current assigned reference camera: {i}")
        paired_keypoints_list = extract_paired_keypoints_with_reference(all_cam_data, i, confidence_threshold)
        
        temp_idx = -1
        for j, K in enumerate(Ks):
            if j == i:
                continue
            temp_idx += 1
            paired_keypoints = paired_keypoints_list[temp_idx]
            keypoints_for_current_camera_pairing = sum(len(frame_keypoints) for frame_keypoints in paired_keypoints)
            
            if keypoints_for_current_camera_pairing == 0:
                zero_keypoints_flag = True
                break
            
            F, inliers1, inliers2 = compute_fundamental_matrix(paired_keypoints)
            inliers_count = len(inliers1)
            print(f"Camera {i} paired with Camera {j}: inliers count: {inliers_count}")
            
            E = compute_essential_matrix(F, Ks[i], K)
            R, t, mask = recover_pose_from_essential_matrix(E, inliers1, inliers2, K)
            
            if R is None or t is None:
                no_R_t_solutions_flag = True
                break
                
            P1 = cam_create_projection_matrix(Ks[i], np.eye(3), np.zeros((3, 1)))
            P2 = cam_create_projection_matrix(K, R, t)
            inliers_pair = create_paired_inlier(inliers1, inliers2)
            points_3d_int = triangulate_points(inliers_pair, P1, P2)
            mean_error = compute_reprojection_error(points_3d_int, inliers_pair, P1, P2)
            score = inliers_count / mean_error if mean_error > 0 else inliers_count
            total_score += score
            print(f"Camera {j} relative to Reference Camera {i}: Mean reprojection error: {mean_error}")
            camera_Rt_list[i][j] = [R, t]
        
        if not no_R_t_solutions_flag and not zero_keypoints_flag:
            average_score = total_score/(len(camera_dirs)-1)
            valid_ref_cam_idx.append(i)
            average_scores_for_valid_ref_cam.append(average_score)
            temp_paired_keypoints_global_list.append(paired_keypoints_list)
    
    # Find the best reference camera based on average score
    if not valid_ref_cam_idx:
        raise ValueError("No valid reference camera found!")
        
    index_of_best_ref_cam = average_scores_for_valid_ref_cam.index(max(average_scores_for_valid_ref_cam))
    final_idx_of_ref_cam = valid_ref_cam_idx[index_of_best_ref_cam]
    paired_keypoints_list = temp_paired_keypoints_global_list[index_of_best_ref_cam]
    
    # Identify constrained camera
    available_cameras = [i for i in range(len(Ks)) if i != final_idx_of_ref_cam]
    constrained_camera = available_cameras[0]  # Pick the first available camera
    
    print(f"Best reference camera: {final_idx_of_ref_cam}")
    print(f"Constrained camera: {constrained_camera}")
    
    final_camera_Rt = camera_Rt_list[final_idx_of_ref_cam]
    
    # Create inliers lists for optimization
    temp_idx = -1
    for j in range(len(Ks)):
        if j == final_idx_of_ref_cam:
            continue
        temp_idx += 1
        paired_keypoints = paired_keypoints_list[temp_idx]
        
        _, inliers1, inliers2 = compute_fundamental_matrix(paired_keypoints)
        inliers_pair = create_paired_inlier(inliers1, inliers2)
        inliers_pair_list.append(inliers_pair)
        inlier2_list.append(inliers2)
    
    return final_idx_of_ref_cam, constrained_camera, paired_keypoints_list, inliers_pair_list, inlier2_list, final_camera_Rt


def optimize_camera_parameters(final_idx_of_ref_cam, final_camera_Rt, Ks, inliers_pair_list, inlier2_list, constrained_camera):
    """
    Optimize intrinsic and extrinsic camera parameters.
    
    Args:
        final_idx_of_ref_cam (int): Index of the reference camera.
        final_camera_Rt (dict): Dictionary of camera R and t values.
        Ks (list): List of intrinsic matrices for each camera.
        inliers_pair_list (list): List of paired inliers for all camera pairs.
        inlier2_list (list): List of inlier points for secondary cameras.
        
    Returns:
        dict: Dictionary of optimized camera parameters.
    """
    # Initialize tolerance parameters for optimization
    tolerance_list = {
        'ftol': 1e-3,
        'xtol': 1e-4,
        'gtol': 1e-3,
        'max_nfev': 50,
        'diff_step': 1e-3
    }
    
    all_best_results = {}
    iterations = 1  # Number of intrinsic optimization iterations
    
    # Fixed reference camera intrinsic matrix
    Fix_K = Ks[final_idx_of_ref_cam]
    
    # Principal point coordinates
    u0 = Fix_K[0, 2]  # Principal point u0
    v0 = Fix_K[1, 2]  # Principal point v0
    
    # First phase: Optimize intrinsics for each camera
    print("Phase 1: Intrinsic parameter optimization")
    temp_idx = -1
    for j in range(len(Ks)):
        if j == final_idx_of_ref_cam:
            continue
        temp_idx += 1
        
        paired_keypoints = inliers_pair_list[temp_idx]
        inliers2 = inlier2_list[temp_idx]
        
        camera_pair_key = f"Camera{final_idx_of_ref_cam}_{j}"
        print(f"Optimizing for pair: {camera_pair_key}")
        K_optimized = Ks[j]
        R_optimized, t_optimized = final_camera_Rt[j]
        
        optimization_results = {'K1': [], 'K2': [], 'R': [], 't': [], 'errors': []}
        
        for iteration in range(iterations):
            print(f"---Iteration {iteration + 1} for {camera_pair_key} ---")
            OPT_K = K_optimized
            
            P1 = cam_create_projection_matrix(Fix_K, np.eye(3), np.zeros((3, 1)))
            P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
            
            points_3d_int = triangulate_points(paired_keypoints, P1, P2)
            OPT_K_optimized = optimize_intrinsic_parameters(
                points_3d_int, inliers2, OPT_K, R_optimized, t_optimized, tolerance_list, u0, v0
            )
            OPT_K = OPT_K_optimized
            
            P2 = cam_create_projection_matrix(OPT_K, R_optimized, t_optimized)
            int_error = compute_reprojection_error(points_3d_int, paired_keypoints, P1, P2)
            
            print(f"Camera pair {camera_pair_key} Iteration {iteration + 1}: Mean reprojection error: {int_error}")
            
            optimization_results['K1'].append(Fix_K)
            optimization_results['K2'].append(OPT_K)
            optimization_results['R'].append(R_optimized)
            optimization_results['t'].append(t_optimized)
            optimization_results['errors'].append(int_error)
            
            K_optimized = OPT_K
        
        # Select best parameters from optimization iterations
        if optimization_results['errors']:
            min_error_for_pair = min(optimization_results['errors'])
            index_of_min_error = optimization_results['errors'].index(min_error_for_pair)
            best_K1 = optimization_results['K1'][index_of_min_error]
            best_K2 = optimization_results['K2'][index_of_min_error]
            best_R = optimization_results['R'][index_of_min_error]
            best_t = optimization_results['t'][index_of_min_error]
            
            all_best_results[camera_pair_key] = {
                'K1': best_K1,
                'K2': best_K2,
                'R': best_R,
                't': best_t,
                'error': min_error_for_pair
            }
    
    # Display best intrinsic optimization results
    for pair_key, results in all_best_results.items():
        print(f"Best intrinsic results for {pair_key}:")
        print(f"- Minimum reprojection error: {results['error']}")
    
    # Second phase: Joint optimization of extrinsic parameters
    print("\nPhase 2: Extrinsic parameter joint optimization")
    
    # Set up extrinsic optimization parameters
    tolerance_list = {
        'ftol': 1e-2,
        'xtol': 1e-2,
        'gtol': 1e-2,
        'max_nfev': 100,
        'diff_step': 1e-2
    }
    
    prev_average_error = float('inf')
    min_improvement = 1e-10  # Minimum improvement threshold
    MAX_ITERATIONS = 100
    iteration = 0
    
    while True:
        total_error = 0
        temp_idx = -1
        
        for i, K in enumerate(Ks):
            if i == final_idx_of_ref_cam:  # Skip reference camera
                continue
            temp_idx += 1
            
            inliers_pair = inliers_pair_list[temp_idx]
            other_keypoints_detected = inlier2_list[temp_idx]
            
            print(f"Calibrating camera {i}...")
            
            # Import best results for this camera pair
            ext_K = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['K2']
            ext_R = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['R']
            ext_t = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t']
            
            ref_t = np.zeros((3, 1))  # Reference camera t vector
            P1 = cam_create_projection_matrix(Ks[final_idx_of_ref_cam], np.eye(3), ref_t)
            P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)
            
            # Triangulate points
            points_3d = triangulate_points(inliers_pair, P1, P2)
            before_optimization_error = compute_reprojection_error(points_3d, inliers_pair, P1, P2)
            print(f"Camera {i} before optimization error: {before_optimization_error}")
            
            # Extrinsic parameter optimization
            print(f"Before optimization t vector: {ext_t}")
            
            if i == constrained_camera:
                optimized_t = optimize_extrinsic_parameters(
                    points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, 
                    tolerance_list, isConstrained=True, h=1.0
                )
            else:
                optimized_t = optimize_extrinsic_parameters(
                    points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tolerance_list
                )
            
            ext_t = optimized_t  # Update t vector
            
            N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)
            new_points_3d = triangulate_points(inliers_pair, P1, N_P2)
            ex_reprojection_error = compute_reprojection_error(new_points_3d, inliers_pair, P1, N_P2)
            total_error += ex_reprojection_error
            
            all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t'] = ext_t
        
        # Calculate average error across all cameras
        current_average_error = total_error / (len(Ks) - 1)
        print(f"Iteration {iteration}: Average reprojection error = {current_average_error}")
        
        if (prev_average_error - current_average_error) < min_improvement:
            print("Optimization completed: Error reduction below threshold")
            break
            
        if iteration >= MAX_ITERATIONS:
            print("Optimization completed: Maximum iterations reached")
            break
            
        prev_average_error = current_average_error
        iteration += 1
    
    return all_best_results

# TODO: TO REMOVE ONCE MIGRATED
def apply_scale_calibration(all_best_results, final_idx_of_ref_cam, segments_file, Ks, inliers_pair_list):
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
    
    # Get initial 3D points using unscaled parameters
    P1 = cam_create_projection_matrix(Ks[final_idx_of_ref_cam], np.eye(3), np.zeros((3, 1)))
    first_cam_idx = next(i for i in range(len(Ks)) if i != final_idx_of_ref_cam)
    first_cam_key = f"Camera{final_idx_of_ref_cam}_{first_cam_idx}"
    P2 = cam_create_projection_matrix(
        all_best_results[first_cam_key]['K2'],
        all_best_results[first_cam_key]['R'],
        all_best_results[first_cam_key]['t']
    )
    
    # Use first pair of cameras to compute scale
    # TODO: Why only use first camera pair, I need to use the others as well.
    inliers_pair = inliers_pair_list[0]  # Using first camera pair
    points_3d = triangulate_points(inliers_pair, P1, P2)
    
    # Compute and apply scale
    scale_factor = compute_scale_factor_from_segments(points_3d, known_segments)
    print(f"Computed scale factor: {scale_factor}")
    
    if scale_factor is None or scale_factor <= 0:
        print("Warning: Invalid scale factor. Using default scale of 1.0")
        scale_factor = 1.0
    
    # Apply scale to all camera parameters
    scaled_results = apply_scale_to_results(all_best_results, scale_factor, final_idx_of_ref_cam)
    
    return scaled_results
# Used creation time instead of modification time which is wrong
# def get_latest_trc_file(trc_file_dir):
#     """Get the most recently created TRC file in the directory"""
#     trc_files = glob.glob(os.path.join(trc_file_dir, "*.trc"))
#     if not trc_files:
#         return None
        
    # # Return the most recent file
    # return max(trc_files, key=os.path.getctime)

def get_latest_trc_file(trc_file_dir):
    """Get the most recently modified TRC file in the directory."""
    trc_files = glob.glob(os.path.join(trc_file_dir, "*.trc"))
    if not trc_files:
        return None
    # Sort by modification time (newest last)
    latest_file = max(trc_files, key=os.path.getmtime)
    return latest_file

def run_pose2sim_triangulation(target_dir):


    original_dir = os.getcwd()
    try:
        # Change to the desired directory
        target_directory = target_dir  # Replace with your actual directory path
        os.chdir(target_directory)
        Pose2Sim.triangulation()

    finally:
        # Change back to the original directory
        os.chdir(original_dir)


def calibrate_cameras(openpose_dir, intrinsics_file, segments_file, 
                      confidence_threshold, trc_file_dir, pose2sim_project_dir, output_path, output_filename, img_width, img_height, calc_intrinsics_method='default'):
    """
    Main function to perform hybrid camera calibration.
    
    Args:
        alphapose_dir (str): Path to AlphaPose keypoints directory. Temporarily removed
        openpose_dir (str): Path to OpenPose keypoints directory.
        intrinsics_file (str): Path to TOML file with intrinsic parameters.
        segments_file (str): Path to TOML file with segment definitions.
        confidence_threshold (float): Confidence threshold for keypoint filtering.
        output_path (str): Path to save output files.
        output_filename (str): Name of output TOML file.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.
        calc_intrinsics_method (str): Method for calculating intrinsics ('default' uses Pose2Sim Checkerboard method, 'CasCalib' uses CasCalib method, 'Custom' user will input an initial estimate f). 
    Returns:
        bool: True if calibration succeeded, False otherwise.
    """
    start_time = time.time()
    
    try:
        # Create Path objects
        # ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(alphapose_dir)
        ROOT_PATH_FOR_OPENPOSE_KEYPOINTS = Path(openpose_dir)
        
        # Find all JSON files
        # json_files = list(ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS.glob('*.json'))
        
        # if not json_files:
        #     print(f"No JSON files found in AlphaPose directory: {alphapose_dir}")
        #     return False
        
        # # Extract camera names from filenames
        # camera_names = [file_path.stem for file_path in json_files]
        
        # # Create DATA_PATHS
        # DATA_PATHS = {'detections': json_files}
        
        # Print camera info
        # print("Camera Names:", camera_names)
        
        # Get all OpenPose subdirectories
        OPENPOSE_KEYPOINTS_DIRECTORY = [str(path) for path in ROOT_PATH_FOR_OPENPOSE_KEYPOINTS.glob('*') 
                                       if path.is_dir()]
        print("OpenPose keypoints directories:", OPENPOSE_KEYPOINTS_DIRECTORY)
        
        if not OPENPOSE_KEYPOINTS_DIRECTORY:
            print(f"No subdirectories found in OpenPose directory: {openpose_dir}")
            return False
        
        print("Found directories:", OPENPOSE_KEYPOINTS_DIRECTORY)
        
        # TODO: Make a decision tree for intrinsics calculation method
        if calc_intrinsics_method == 'default':
            print("Using Pose2Sim Checkerboard method for intrinsics calculation")
            
            # Temporary, assumes the current toml file has the intrinsics
            
            # Load camera intrinsics from TOML file
            Ks, _ = load_intrinsics_from_toml(intrinsics_file)
            if Ks is None:
                print("Failed to load intrinsics from TOML file")
                return False
            
        elif calc_intrinsics_method == 'CasCalib':
            print("Using CasCalib method for intrinsics calculation")

            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Temporary output folder for alphapose conversion: {temp_dir}")

                # Loop through all OpenPose keypoint directories
                for i, openpose_dir in enumerate(OPENPOSE_KEYPOINTS_DIRECTORY):
                    output_file = os.path.join(temp_dir, f"alphapose_results_cam{i+1}.json")

                    # Call the converter function directly
                    OpenPose_to_AlphaPose_func(openpose_dir, output_file)

                    print(f"Converted: {openpose_dir}")
                    print(f"Output: {output_file}")
            
            # TODO: Call CasCalib function here to get Ks
            Ks = []  # Placeholder, replace with actual Ks from CasCalib

            ## Call CasCalib function here
        elif calc_intrinsics_method == 'Custom':
            print("Using user-provided initial estimate for intrinsics calculation")


        print(Ks)
        # Calculate principal points
        u0 = img_width / 2
        v0 = img_height / 2

        image_size = [img_width, img_height]

        print(f"Confidence threshold: {confidence_threshold}")
        
        # Load JSON files from OpenPose directories
        all_cam_data = load_json_files(OPENPOSE_KEYPOINTS_DIRECTORY)
        
        # Select the best reference camera
        final_idx_of_ref_cam, constrained_camera, paired_keypoints_list, inliers_pair_list, inlier2_list, final_camera_Rt = select_reference_camera(
            all_cam_data, OPENPOSE_KEYPOINTS_DIRECTORY, Ks, confidence_threshold
        )
        
        # Optimize camera parameters
        all_best_results = optimize_camera_parameters(
            final_idx_of_ref_cam, final_camera_Rt, Ks, inliers_pair_list, inlier2_list, constrained_camera
        )
        
        # Apply scaling to get metric units
        # all_best_results = apply_scale_calibration(
        #     all_best_results, final_idx_of_ref_cam, segments_file, Ks, inliers_pair_list
        # )
        
        # Print final optimized results
        print("\nUnitless optimized camera parameters:")
        for pair_key, results in all_best_results.items():
            print(f"Results for {pair_key}:")
            print(f"- t: {results['t']}")
        
        # Test
        #print(all_best_results[pair_key]['K1'])
        
        # Write results to TOML file
        write_to_toml(
            all_best_results, 
            [0.0, 0.0, 0.0], 
            image_size, 
            output_path=output_path, 
            output_filename=output_filename
        )

        run_pose2sim_triangulation(pose2sim_project_dir)
        latest_trc_file = get_latest_trc_file(trc_file_dir)

        # TODO: ADD THE APPLY SCALE FUNCTION HERE
        scale_factor = calculate_scale_factor(segments_file, latest_trc_file)
        all_best_results = apply_scale_to_results(all_best_results, scale_factor, final_idx_of_ref_cam)
        

        write_to_toml(
        all_best_results, 
        [0.0, 0.0, 0.0], 
        image_size, 
        output_path=output_path, 
        output_filename=output_filename
        )
        # Print final optimized results
        print("\nFinal optimized camera parameters:")
        for pair_key, results in all_best_results.items():
            print(f"Results for {pair_key}:")
            print(f"- t: {results['t']}")

        run_pose2sim_triangulation(pose2sim_project_dir)

        trc_Xup_to_Yup_func(latest_trc_file, None)


        # Calculate and report elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Calibration completed successfully in {elapsed_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error during calibration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Parse command-line arguments and run the calibration.
    """
    parser = argparse.ArgumentParser(description='Hybrid camera calibration from keypoints')
    
    parser.add_argument('--alphapose_dir', type=str, default=".", 
                        help='Path to the Alphapose keypoints directory')
    parser.add_argument('--openpose_dir', type=str, required=True, 
                        help='Path to the Openpose keypoints directory')
    parser.add_argument('--intrinsics_file', type=str, required=True,
                        help='Path to TOML file with camera intrinsic parameters')
    parser.add_argument('--segments_file', type=str, default=None,
                        help='Path to TOML file with body segment definitions')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for keypoints (default: 0.7)')
    parser.add_argument('--output_path', type=str, default=".",
                        help='Path to the output directory (default: current directory)')
    parser.add_argument('--output_filename', type=str, default="calibration.toml",
                        help='Name of the output TOML file (default: calibration.toml)')
    
    args = parser.parse_args()
    
    print("Starting hybrid camera calibration...")
    success = calibrate_cameras(

        args.openpose_dir,
        args.intrinsics_file,
        args.segments_file,
        args.confidence,
        args.output_path,
        args.output_filename
    )
    
    if success:
        print(f"Calibration completed successfully. Results saved to {os.path.join(args.output_path, args.output_filename)}")
    else:
        print("Calibration failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())