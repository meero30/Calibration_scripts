import numpy as np
import cv2
from utilities.Loader import unpack_keypoints




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


def create_heuristic_K_matrix(image_width, image_height):
    """
    Creates a stable, deterministic initial guess for a K matrix.
    """

    # 1. Guess Principal Point (cx, cy) is at the center
    cx = image_width / 2.0
    cy = image_height / 2.0

    # 2. Guess Focal Length (fx, fy) is the largest dimension
    fx = max(image_width, image_height)
    fy = max(image_width, image_height)

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    return K