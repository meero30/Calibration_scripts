import time
import cv2
import numpy as np  # <-- We still use numpy for data prep
from Calibration.Calculations import cam_create_projection_matrix, triangulate_points
from scipy.optimize import least_squares

# --- JAX: Imports ---
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import expm


# --- JAX: Re-implementation of cv2.Rodrigues ---
# # We need a pure-JAX function so we can differentiate through it.
# def _R_from_rodrigues_jax(rvec):
#     """rvec (3,) -> R (3x3) using pure JAX"""
#     theta = jnp.linalg.norm(rvec)
    
#     # --- JAX: Use jax.lax.cond for safe branching ---
#     # This avoids a divide-by-zero if theta is 0
#     def true_fn(rvec):
#         # Not zero, compute rotation
#         k = rvec / theta
#         K = jnp.array([
#             [0, -k[2], k[1]],
#             [k[2], 0, -k[0]],
#             [-k[1], k[0], 0]
#         ])
#         cos_t = jnp.cos(theta)
#         sin_t = jnp.sin(theta)
#         return jnp.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)
    
#     def false_fn(rvec):
#         # Zero, return identity
#         return jnp.eye(3)
        
#     # Check if theta is close to zero
#     return jax.lax.cond(theta < 1e-12, false_fn, true_fn, rvec)

# --- JAX: Re-implementation of cv2.Rodrigues ---
# This version uses the matrix exponential, which is
# numerically stable and has a well-defined, smooth
# derivative (Jacobian) even for zero-rotations.
# This fixes the optimizer instability.
# def _R_from_rodrigues_jax(rvec):
#     """rvec (3,) -> R (3x3) using jax.scipy.linalg.expm"""

#     # Create the 3x3 skew-symmetric matrix K
#     K = jnp.array([
#         [0, -rvec[2], rvec[1]],
#         [rvec[2], 0, -rvec[0]],
#         [-rvec[1], rvec[0], 0],
#     ])

#     # Compute the matrix exponential, which is the
#     # stable equivalent of the Rodrigues formula.
#     return expm(K)


# --- JAX: Re-implementation of cv2.Rodrigues ---
# This is the "fast and stable" version.
# It uses the direct algebraic formula but avoids division by zero
# using a Taylor approximation, which is fully differentiable.
def _R_from_rodrigues_jax(rvec):
    """rvec (3,) -> R (3x3) using the stable algebraic formula"""
    
    # Calculate theta_sq and theta
    theta_sq = jnp.dot(rvec, rvec)
    theta = jnp.sqrt(theta_sq)
    
    # Create the skew-symmetric matrix K from rvec
    K = jnp.array([
        [0, -rvec[2], rvec[1]],
        [rvec[2], 0, -rvec[0]],
        [-rvec[1], rvec[0], 0]
    ])
    
    # R = I + A*K + B*K^2
    # We need to compute A = sin(theta)/theta
    # and B = (1 - cos(theta))/theta^2
    
    # Use Taylor series expansion for A and B when theta is near zero
    # This avoids division by zero and is numerically stable.
    # We use jnp.where to create a branchless, JIT-compatible selection.
    
    # Check if theta is very small
    is_near_zero = theta_sq < 1e-8
    
    # Taylor expansion for A = 1 - theta^2/6 + ...
    A_small = 1.0 - theta_sq / 6.0
    # Taylor expansion for B = 1/2 - theta^4/24 + ...
    B_small = 0.5 - theta_sq / 24.0
    
    # Standard formula for A
    A_large = jnp.sin(theta) / theta
    # Standard formula for B
    B_large = (1.0 - jnp.cos(theta)) / theta_sq
    
    # Select A and B based on the value of theta
    A = jnp.where(is_near_zero, A_small, A_large)
    B = jnp.where(is_near_zero, B_small, B_large)
    
    # Compute the final rotation matrix
    return jnp.eye(3) + A * K + B * (K @ K)

# --- JAX: Keep the original for creating x0 ---
# This is fine, as it's outside the optimization loop.
def _rodrigues_from_R(R):
    """R (3x3) -> rvec (3,) using OpenCV"""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.flatten()

# --- JAX: This is our JAX-compatible projection function ---
# def project_point_jax(rvec, tvec, K, pt3d):
#     """Projects one 3D point to 2D using JAX"""
#     R = _R_from_rodrigues_jax(rvec)
#     Xc = R @ pt3d + tvec
    
#     # Add epsilon to prevent division by zero
#     z = Xc[2] + 1e-12 
#     uv_h = K @ Xc
    
#     # Return (u, v)
#     return uv_h[:2] / z

def project_point_jax(rvec, tvec, K, pt3d):
    """
    Projects one 3D point to 2D using JAX,
    correctly replicating the original script's logic.
    """
    R = _R_from_rodrigues_jax(rvec)
    Xc = R @ pt3d + tvec
    
    # 1. Project 3D point into homogeneous coordinates
    #    uv_h = [u*w, v*w, w]
    uv_h = K @ Xc
    
    # 2. Get the 'w' component (the third element)
    w = uv_h[2]
    
    # 3. Add epsilon for a numerically stable (and differentiable)
    #    divide, preventing division by zero.
    #    We use jnp.sign to preserve the sign, as w could be negative
    #    (point behind camera), which is important.
    w_safe = jnp.sign(w) * jnp.maximum(jnp.abs(w), 1e-12)

    # 4. Perform perspective division (de-homogenize)
    #    This is [u*w / w, v*w / w]
    return uv_h[:2] / w_safe

# --- JAX: This is our new parameter unpacking function ---
# It's designed to be JIT-compatible
def unpack_params_jax(
    x,
    static_Ks,
    num_cameras,
    final_idx_of_ref_cam,
    optimize_intrinsics
):
    """Unpacks the flat parameter vector 'x' into JAX arrays"""
    
    cams_rvecs_list = []
    cams_tvecs_list = []
    cams_Ks_list = []
    offset = 0

    params_per_cam = 8 if optimize_intrinsics else 6
    
    # This loop is JAX-compatible because num_cameras is a static value
    for cam in range(num_cameras):
        if cam == final_idx_of_ref_cam:
            cams_rvecs_list.append(jnp.zeros(3))
            cams_tvecs_list.append(jnp.zeros(3))
            cams_Ks_list.append(static_Ks[cam])
        else:
            rvec = x[offset:offset+3]
            tvec = x[offset+3:offset+6]
            cams_rvecs_list.append(rvec)
            cams_tvecs_list.append(tvec)
            
            if optimize_intrinsics:
                fx, fy = x[offset+6], x[offset+7]
                # JAX-safe "update" of an array
                Kcam = static_Ks[cam].at[0,0].set(fx).at[1,1].set(fy)
                cams_Ks_list.append(Kcam)
                offset += 8
            else:
                cams_Ks_list.append(static_Ks[cam])
                offset += 6
                
    # Stack lists into final arrays
    all_rvecs = jnp.stack(cams_rvecs_list)
    all_tvecs = jnp.stack(cams_tvecs_list)
    all_Ks = jnp.stack(cams_Ks_list)
    
    # The rest of the vector 'x' is the 3D points
    all_pts_3d = x[offset:].reshape(-1, 3)
    
    return all_rvecs, all_tvecs, all_Ks, all_pts_3d

# --- JAX: This is the core residual function to be JIT-compiled ---
def residuals_fn_jax(
    x,
    # These are all static arguments that won't change
    static_Ks,
    cam_indices,
    point_indices,
    points_2d_obs,
    num_cameras,
    final_idx_of_ref_cam,
    optimize_intrinsics,
    pair_camera_idxs,
    #constrained_camera,
    constraint_weight
):
    """
    Computes all residuals as a single JAX array.
    This is the function we will differentiate.
    """
    
    # 1. Unpack all parameters
    all_rvecs, all_tvecs, all_Ks, all_pts_3d = unpack_params_jax(
        x,
        static_Ks,
        num_cameras,
        final_idx_of_ref_cam,
        optimize_intrinsics
    )
    
    # 2. Gather parameters for each observation using array indexing
    # This replaces the slow Python 'for obs in observations:' loop
    obs_rvecs = all_rvecs[cam_indices]
    obs_tvecs = all_tvecs[cam_indices]
    obs_Ks = all_Ks[cam_indices]
    obs_pts_3d = all_pts_3d[point_indices]
    
    # 3. Vectorize the projection function
    # jax.vmap maps project_point_jax over all observations at once
    project_vmap = vmap(project_point_jax)
    points_2d_proj = project_vmap(obs_rvecs, obs_tvecs, obs_Ks, obs_pts_3d)
    
    # 4. Compute reprojection errors
    reprojection_errors = (points_2d_proj - points_2d_obs).flatten()
    
    # 5. Compute |t|=1 constraint
    # tvec_constrained = all_tvecs[constrained_camera]
    # t_norm = jnp.linalg.norm(tvec_constrained)
    # constraint_error = jnp.array([constraint_weight * (t_norm - 1.0)])
    non_ref_tvecs = all_tvecs[pair_camera_idxs]
    # Define the |t|=1 constraint for a *single* vector
    def norm_constraint(tvec):
        t_norm = jnp.linalg.norm(tvec)
        return constraint_weight * (t_norm - 1.0)
    
    # Use vmap to apply this function to *all* non-reference t-vectors
    constraint_error = vmap(norm_constraint)(non_ref_tvecs)

    # 6. Combine all errors into one vector
    return jnp.concatenate([reprojection_errors, constraint_error])

def bundle_adjustment_refine(
    Ks,
    final_idx_of_ref_cam,
    final_camera_Rt,
    inliers_pair_list,
    inlier2_list,
    optimize_intrinsics=False,
    constrained_camera=1,
    constraint_weight=1000,
    max_nfev=200,
    verbose=1,
):
    t0 = time.time()
    num_cameras = len(Ks)
    pair_camera_idxs = [j for j in range(num_cameras) if j != final_idx_of_ref_cam]

    # --- Build observation and point lists (Unchanged) ---
    observations_list = [] # <-- Renamed to avoid confusion
    points3d_list = []
    point_base_idx = []
    global_point_counter = 0

    for p_idx, cam_j in enumerate(pair_camera_idxs):
        paired = inliers_pair_list[p_idx]
        K_ref = Ks[final_idx_of_ref_cam]
        R_ref = np.eye(3)
        t_ref = np.zeros((3, 1))
        K_j = Ks[cam_j]
        R_j, t_j = final_camera_Rt[cam_j]

        P1 = cam_create_projection_matrix(K_ref, R_ref, t_ref)
        P2 = cam_create_projection_matrix(K_j, R_j, t_j)
        pts3d_raw = triangulate_points(paired, P1, P2)
        pts3d = np.array([p.flatten() for p in pts3d_raw], dtype=np.float64)

        n_pts = pts3d.shape[0]
        points3d_list.append(pts3d)
        point_base_idx.append(global_point_counter)

        for i_pt in range(n_pts):
            u1, v1 = paired[i_pt][0]
            u2, v2 = paired[i_pt][1]
            observations_list.append((final_idx_of_ref_cam, global_point_counter + i_pt, float(u1), float(v1)))
            observations_list.append((cam_j, global_point_counter + i_pt, float(u2), float(v2)))
        global_point_counter += n_pts

    total_points = global_point_counter
    if verbose:
        print(f"[BA] {len(pair_camera_idxs)} pairs, {total_points} points, {len(observations_list)} observations")

    # --- Build parameter vector 'x0' (Unchanged) ---
    # We use original cv2 and numpy to build the *initial guess*
    cam_var_index = {}
    param_list = []
    for cam in range(num_cameras):
        if cam == final_idx_of_ref_cam:
            cam_var_index[cam] = None
            continue
        R, t = final_camera_Rt[cam]
        rvec = _rodrigues_from_R(R) # <-- Original cv2-based
        tvec = t.flatten()
        if optimize_intrinsics:
            fx, fy = Ks[cam][0, 0], Ks[cam][1, 1]
            cam_block = np.hstack([rvec, tvec, [fx, fy]])
        else:
            cam_block = np.hstack([rvec, tvec])
        cam_var_index[cam] = len(param_list)
        param_list.append(cam_block)
    cam_param_vec = np.hstack(param_list) if param_list else np.array([])
    pts_vec = np.hstack([pts.flatten() for pts in points3d_list]) if total_points > 0 else np.array([])
    x0 = np.hstack([cam_param_vec, pts_vec]).astype(np.float64)

    # --- JAX: Data Preparation ---
    # Convert all static data to JAX arrays once.
    # We send these to the device (e.g., GPU) if available.
    static_Ks_jax = jax.device_put(jnp.array(Ks))
    cam_indices_jax = jax.device_put(jnp.array([o[0] for o in observations_list], dtype=jnp.int32))
    point_indices_jax = jax.device_put(jnp.array([o[1] for o in observations_list], dtype=jnp.int32))
    points_2d_obs_jax = jax.device_put(jnp.array([[o[2], o[3]] for o in observations_list], dtype=jnp.float64))
    pair_camera_idxs_jax = jax.device_put(jnp.array(pair_camera_idxs, dtype=jnp.int32))
    # --- JAX: Setup JIT and Jacobian Functions ---
    # We "close over" the static data by wrapping our residual function
    # in a lambda that only takes 'x'.
    residuals_for_scipy = lambda x: residuals_fn_jax(
        x,
        static_Ks_jax,
        cam_indices_jax,
        point_indices_jax,
        points_2d_obs_jax,
        num_cameras,
        final_idx_of_ref_cam,
        optimize_intrinsics,
        pair_camera_idxs_jax,
        #constrained_camera,
        constraint_weight
    )

    if verbose:
        print("[BA] JIT-compiling residual and Jacobian functions... (may take a moment)")
    
    # # JIT-compile the residual function
    # jit_residuals = jit(residuals_for_scipy)
    
    # # JIT-compile the Jacobian function using jax.jacfwd
    # # jacfwd is (forward-mode) is generally more efficient for "tall"
    # # Jacobians (more residuals than parameters), but jacrev (reverse-mode)
    # # can also be used.
    # jit_jacobian = jit(jax.jacfwd(residuals_for_scipy))
    
    # JIT-compile the core JAX functions (rename with _)
    _jit_residuals = jit(residuals_for_scipy)
    _jit_jacobian = jit(jax.jacfwd(residuals_for_scipy))
    
    if verbose:
        print("[BA] JIT compilation complete.")

    # --- JAX: Create wrappers for SciPy ---
    # These wrappers call the fast JAX code, then make a
    # WRITABLE NUMPY COPY for SciPy to use.
    
    def scipy_residuals_wrapper(x):
        return np.array(_jit_residuals(x))

    def scipy_jacobian_wrapper(x):
        return np.array(_jit_jacobian(x))
    
    # --- Optimization ---
    if verbose:
        print("[BA] Starting optimization with |t|=1 constraint and JAX Jacobian")
    
    # # --- JAX: Call least_squares with the JIT-compiled functions ---
    # lsq_result = least_squares(
    #     jit_residuals,  # <--- JAX residual function
    #     x0,
    #     jac=jit_jacobian, # <--- JAX Jacobian function
    #     method="trf",
    #     verbose=1,
    #     max_nfev=max_nfev,
    #     ftol=1e-5,
    #     xtol=1e-5,
    #     gtol=1e-5,
    #     loss="huber",
    # )

    # --- JAX: Call least_squares with the NEW wrappers ---
    # lsq_result = least_squares(
    #     scipy_residuals_wrapper,  # <--- NEW
    #     x0,
    #     jac=scipy_jacobian_wrapper, # <--- NEW
    #     method="trf",
    #     verbose=1,
    #     max_nfev=max_nfev,
    #     ftol=1e-5,
    #     xtol=1e-5,
    #     gtol=1e-5,
    #     loss="huber",
    # )

    lsq_result = least_squares(
        scipy_residuals_wrapper, 
        x0,
        jac=scipy_jacobian_wrapper, 
        method="trf",
        verbose=1,
        max_nfev=max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-5,
        loss="linear"
    )

    # --- JAX: Unpack final results ---
    # We can use the JAX unpacker on the final result vector
    # (which is a numpy array) by first converting it to a jnp array.
    cams_final_r, cams_final_t, Ks_final, pts_final = unpack_params_jax(
        jnp.array(lsq_result.x),
        static_Ks_jax,
        num_cameras,
        final_idx_of_ref_cam,
        optimize_intrinsics
    )
    
    # Convert results back to standard numpy/dicts
    pts_final = np.array(pts_final)
    refined_Ks = [np.array(K) for K in Ks_final]
    refined_Rs, refined_ts = {}, {}
    for cam in range(num_cameras):
        # We must use the *original* cv2 function to get the
        # R matrix back, as our JAX one is for JIT-compilation.
        # Or, we can just call our JAX function and convert to numpy.
        Rm = np.array(_R_from_rodrigues_jax(cams_final_r[cam]))
        tm = np.array(cams_final_t[cam]).reshape(3, 1)
        refined_Rs[cam] = Rm
        refined_ts[cam] = tm

    # Split points per pair (Unchanged)
    refined_points_per_pair = []
    for idx, base in enumerate(point_base_idx):
        npts = points3d_list[idx].shape[0]
        seg = pts_final[base : base + npts, :]
        refined_points_per_pair.append(seg.copy())

    t1 = time.time()
    
    # --- JAX: Compute final residuals using the JIT function ---
    # Note: lsq_result.fun already contains the final residuals
    final_residuals = lsq_result.fun 
    initial_residuals = _jit_residuals(x0)
    
    if verbose:
        print(f"[BA] Done in {t1 - t0:.1f}s | success={lsq_result.success}")
        print(f"[BA] Initial residual norm: {np.linalg.norm(initial_residuals):.3f}")
        print(f"[BA] Final residual norm:   {np.linalg.norm(final_residuals):.3f}")
        print(f"[BA] Translation constraint camera {constrained_camera} final |t| = {np.linalg.norm(refined_ts[constrained_camera]):.4f}")

    return refined_Ks, refined_Rs, refined_ts, refined_points_per_pair