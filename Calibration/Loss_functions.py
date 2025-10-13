import numpy as np


# TODO: Move to Loss_functions.py file
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