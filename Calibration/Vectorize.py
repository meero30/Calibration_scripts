import numpy as np


# TODO: Move to Vectorize.py file
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