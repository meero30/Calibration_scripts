
import numpy as np
from scipy.optimize import least_squares

from Calibration.Calculations import cam_create_projection_matrix, compute_essential_matrix, compute_fundamental_matrix, compute_reprojection_error, recover_pose_from_essential_matrix, triangulate_points
from Calibration.Data_transformations_helper import create_paired_inlier, extract_paired_keypoints_with_reference
from Calibration.Loss_functions import compute_extrinsics_optimization_loss, compute_intrinsics_optimization_loss
from Calibration.Vectorize import vectorize_params_for_extrinsic_loss, vectorize_params_for_intrinsic_loss


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



def optimize_camera_parameters_test(final_idx_of_ref_cam, final_camera_Rt, Ks, inliers_pair_list, inlier2_list, constrained_camera):
    """
    Jointly optimize intrinsic and extrinsic camera parameters in alternating cycles.
    Each main loop iteration performs one intrinsic phase and one extrinsic phase.
    Stops when the average error improvement falls below a threshold.
    """

    print("\n========== DEBUG INFO FOR BUNDLE ADJUSTMENT ==========")

    print("Number of cameras (Ks):", len(Ks))
    for i, K in enumerate(Ks):
        print(f"Camera {i} intrinsic matrix shape: {np.shape(K)}")
        print(K)

    print("\nExtrinsics in final_camera_Rt:")
    for key, val in final_camera_Rt.items():
        try:
            R, t = val
            print(f"Camera {key}: R shape {np.shape(R)}, t shape {np.shape(t)}")
        except Exception as e:
            print(f"Camera {key} data not unpacked correctly:", e)
    print("\n========== END DEBUG INFO ==========\n")

    # ---- Parameters ----
    tol_intrinsic = {'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5, 'max_nfev': 1000, 'diff_step': 1e-3}
    tol_extrinsic = {'ftol': 1e-2, 'xtol': 1e-2, 'gtol': 1e-2, 'max_nfev': 100, 'diff_step': 1e-2}

    Fix_K = Ks[final_idx_of_ref_cam]
    u0, v0 = Fix_K[0, 2], Fix_K[1, 2]

    # ---- Initialize results ----
    all_best_results = {}
    prev_average_error = float('inf')
    min_improvement = 1e-10
    MAX_MAIN_ITER = 50
    main_iter = 0

    while True:
        print(f"\n========== MAIN JOINT OPTIMIZATION LOOP {main_iter + 1} ==========")
        total_error = 0
        temp_idx = -1

        # ==========================================================
        # Phase 1: Intrinsic optimization
        # ==========================================================
        print("\n>>> Phase 1: Intrinsic parameter optimization <<<")
        for j in range(len(Ks)):
            if j == final_idx_of_ref_cam:
                continue
            temp_idx += 1

            paired_keypoints = inliers_pair_list[temp_idx]
            inliers2 = inlier2_list[temp_idx]
            camera_pair_key = f"Camera{final_idx_of_ref_cam}_{j}"

            K_optimized = Ks[j]
            R_optimized, t_optimized = final_camera_Rt[j]

            P1 = cam_create_projection_matrix(Fix_K, np.eye(3), np.zeros((3, 1)))
            P2 = cam_create_projection_matrix(K_optimized, R_optimized, t_optimized)
            points_3d = triangulate_points(paired_keypoints, P1, P2)

            # Intrinsic refinement
            K_new = optimize_intrinsic_parameters(points_3d, inliers2, K_optimized, R_optimized, t_optimized, tol_intrinsic, u0, v0)
            Ks[j] = K_new  # update Ks list

            # Compute error
            P2_new = cam_create_projection_matrix(K_new, R_optimized, t_optimized)
            int_error = compute_reprojection_error(points_3d, paired_keypoints, P1, P2_new)
            print(f"Camera pair {camera_pair_key}: intrinsic reprojection error = {int_error}")

            all_best_results[camera_pair_key] = {'K1': Fix_K, 'K2': K_new, 'R': R_optimized, 't': t_optimized, 'error': int_error}
            total_error += int_error

        # ==========================================================
        # Phase 2: Extrinsic optimization
        # ==========================================================
        print("\n>>> Phase 2: Extrinsic parameter optimization <<<")
        temp_idx = -1
        for i, K in enumerate(Ks):
            if i == final_idx_of_ref_cam:
                continue
            temp_idx += 1

            inliers_pair = inliers_pair_list[temp_idx]
            other_keypoints_detected = inlier2_list[temp_idx]

            ext_entry = all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]
            ext_K = ext_entry['K2']
            ext_R = ext_entry['R']
            ext_t = ext_entry['t']

            P1 = cam_create_projection_matrix(Fix_K, np.eye(3), np.zeros((3, 1)))
            P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)
            points_3d = triangulate_points(inliers_pair, P1, P2)

            # Extrinsic refinement
            if i == constrained_camera:
                optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tol_extrinsic, isConstrained=True, h=1.0)
            else:
                optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t, tol_extrinsic)

            ext_t = optimized_t
            N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)
            new_points_3d = triangulate_points(inliers_pair, P1, N_P2)
            ex_error = compute_reprojection_error(new_points_3d, inliers_pair, P1, N_P2)

            print(f"Camera {i}: extrinsic reprojection error = {ex_error}")
            total_error += ex_error
            all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['t'] = ext_t
            all_best_results[f"Camera{final_idx_of_ref_cam}_{i}"]['error'] = ex_error

        # ==========================================================
        # Phase 3: Convergence check
        # ==========================================================
        current_average_error = total_error / (len(Ks) - 1)
        print(f"\nMain Loop {main_iter}: Average reprojection error = {current_average_error}")

        if (prev_average_error - current_average_error) < min_improvement:
            print("Converged: Error improvement below threshold.")
            break

        if main_iter >= MAX_MAIN_ITER:
            print("Reached maximum joint iterations.")
            break

        prev_average_error = current_average_error
        main_iter += 1

    print("\nJoint optimization completed.")
    return all_best_results

# Old version kept for reference, trying out new joint optimization above
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

    print("\n========== DEBUG INFO FOR BUNDLE ADJUSTMENT ==========")

    # 1. Intrinsics
    print("Number of cameras (Ks):", len(Ks))
    for i, K in enumerate(Ks):
        print(f"Camera {i} intrinsic matrix shape: {np.shape(K)}")
        print(K)

    # 2. Extrinsics (Rotation & Translation)
    print("\nExtrinsics in final_camera_Rt:")
    for key, val in final_camera_Rt.items():
        try:
            R, t = val
            print(f"Camera {key}: R shape {np.shape(R)}, t shape {np.shape(t)}")
            print("R:\n", R)
            print("t.T:", t.T)
        except Exception as e:
            print(f"Camera {key} data not unpacked correctly:", e)

    # 3. Inlier pairs (used for triangulation)
    print("\nInliers Pair List:")
    print("Length of inliers_pair_list:", len(inliers_pair_list))
    if len(inliers_pair_list) > 0:
        print("Example inlier_pair_list[0] shape:", np.shape(inliers_pair_list[0]))
        print(inliers_pair_list[0][:5])

    # 4. Inlier secondary list (target 2D detections)
    print("\nInlier2 List:")
    print("Length of inlier2_list:", len(inlier2_list))
    if len(inlier2_list) > 0:
        print("Example inlier2_list[0] shape:", np.shape(inlier2_list[0]))
        print(inlier2_list[0][:5])

    # 5. Triangulation sanity check
    try:
        example_pair = inliers_pair_list[0]
        P1 = cam_create_projection_matrix(Ks[0], np.eye(3), np.zeros((3, 1)))
        P2 = cam_create_projection_matrix(Ks[1], *final_camera_Rt[1])
        points_3d_test = triangulate_points(example_pair, P1, P2)
        print("\nTriangulated 3D points shape:", np.shape(points_3d_test))
        print("Example 3D points:\n", points_3d_test[:5])
    except Exception as e:
        print("\nTriangulation test failed:", e)

    print("\n========== END DEBUG INFO ==========\n")


    # Initialize tolerance parameters for optimization
    # tolerance_list = {
    #     'ftol': 1e-3,
    #     'xtol': 1e-4,
    #     'gtol': 1e-3,
    #     'max_nfev': 50,
    #     'diff_step': 1e-3
    # }


    tolerance_list = {
        'ftol': 1e-5,
        'xtol': 1e-5,
        'gtol': 1e-5,
        'max_nfev': 1000,
        'diff_step': 1e-3
    }
    
    all_best_results = {}
    iterations = 20  # Number of intrinsic optimization iterations
    
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
    
    # Doesnt have any history of all calculated extrinsics, only keeps updating the last best result
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