########## Normal Bundle Adjustment Pipeline ##########


import time
import cv2
import numpy as np
from Calibration.Calculations import cam_create_projection_matrix, triangulate_points
from scipy.optimize import least_squares

def _rodrigues_from_R(R):
    """R (3x3) -> rvec (3,) using OpenCV"""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.flatten()

def _R_from_rodrigues(rvec):
    """rvec (3,) -> R (3x3)"""
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R

def bundle_adjustment_refine(
    Ks,
    final_idx_of_ref_cam,
    final_camera_Rt,
    inliers_pair_list,
    inlier2_list,
    optimize_intrinsics=False,
    constrained_camera=1,   # <-- translation constraint camera index
    constraint_weight=1000, # <-- penalty strength for |t|-1
    max_nfev=200,
    verbose=1,
):
    """
    Bundle Adjustment refinement with Liu-style |t| = 1 constraint.
    - Ks: list of 3x3 intrinsic matrices
    - final_idx_of_ref_cam: reference camera index (fixed)
    - final_camera_Rt: dict {cam_idx: [R(3x3), t(3x1)]}
    - inliers_pair_list: list of (N, 2, 2) arrays of corresponding 2D points
    - inlier2_list: list of (N, 2) arrays of keypoints (used for projection consistency)
    - optimize_intrinsics: refine fx, fy if True
    - constrained_camera: which camera’s translation magnitude is constrained to 1
    - constraint_weight: penalty multiplier for deviation from |t| = 1
    - Returns refined Ks, Rs, ts, and per-pair 3D points
    """

    t0 = time.time()
    num_cameras = len(Ks)
    pair_camera_idxs = [j for j in range(num_cameras) if j != final_idx_of_ref_cam]

    # Build observation and point lists
    observations = []
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

        # Triangulate
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
            observations.append((final_idx_of_ref_cam, global_point_counter + i_pt, float(u1), float(v1)))
            observations.append((cam_j, global_point_counter + i_pt, float(u2), float(v2)))
        global_point_counter += n_pts

    total_points = global_point_counter
    if verbose:
        print(f"[BA] {len(pair_camera_idxs)} pairs, {total_points} points, {len(observations)} observations")

    # --- Build parameter vector ---
    cam_var_index = {}
    param_list = []
    for cam in range(num_cameras):
        if cam == final_idx_of_ref_cam:
            cam_var_index[cam] = None
            continue
        R, t = final_camera_Rt[cam]
        rvec = _rodrigues_from_R(R)
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

    def unpack_params(x):
        cams = {}
        offset = 0
        for cam in range(num_cameras):
            if cam == final_idx_of_ref_cam:
                cams[cam] = {'rvec': np.zeros(3), 't': np.zeros(3), 'K': Ks[cam].copy()}
                continue
            if optimize_intrinsics:
                block = x[offset : offset + 8]
                rvec, tvec, fx, fy = block[:3], block[3:6], block[6], block[7]
                offset += 8
                Kcam = Ks[cam].copy()
                Kcam[0, 0], Kcam[1, 1] = fx, fy
            else:
                block = x[offset : offset + 6]
                rvec, tvec = block[:3], block[3:6]
                offset += 6
                Kcam = Ks[cam].copy()
            cams[cam] = {'rvec': rvec, 't': tvec, 'K': Kcam}
        pts = x[offset:].reshape(-1, 3) if total_points > 0 else np.zeros((0, 3))
        return cams, pts

    # --- Residual computation ---
    def residuals(x):
        cams, pts = unpack_params(x)
        res = []

        # Reprojection residuals
        for obs in observations:
            cam_idx, pid, u_obs, v_obs = obs
            camp = cams[cam_idx]
            Kcam = camp['K']
            if cam_idx == final_idx_of_ref_cam:
                Rmat, tvec = np.eye(3), np.zeros(3)
            else:
                Rmat = _R_from_rodrigues(camp['rvec'])
                tvec = camp['t']
            X = pts[pid]
            Xc = Rmat @ X + tvec
            if Xc[2] == 0:
                Xc[2] += 1e-12
            uv = Kcam @ Xc
            u_proj, v_proj = uv[0] / uv[2], uv[1] / uv[2]
            res.append(u_proj - u_obs)
            res.append(v_proj - v_obs)

        # Liu-style |t| = 1 constraint
        if constrained_camera in cams:
            tvec = cams[constrained_camera]['t']
            t_norm = np.linalg.norm(tvec)
            constraint = constraint_weight * (t_norm - 1.0)
            res.append(constraint)

        return np.array(res, dtype=np.float64)

    # --- Optimization ---
    if verbose:
        print("[BA] Starting optimization with |t|=1 constraint")

    # lsq_result = least_squares(
    #     residuals,
    #     x0,
    #     method="trf",
    #     jac="2-point",
    #     verbose=1,
    #     max_nfev=max_nfev,
    #     ftol=1e-8,
    #     xtol=1e-8,
    #     gtol=1e-8,
    #     loss="huber",
    #     diff_step=1e-4,
    # )

    lsq_result = least_squares(
        residuals,
        x0,
        method="trf",
        jac="2-point",
        verbose=1,
        max_nfev=max_nfev,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        loss="huber",
        diff_step=1e-4,
    )

    cams_final, pts_final = unpack_params(lsq_result.x)

    refined_Ks = [K.copy() for K in Ks]
    refined_Rs, refined_ts = {}, {}
    for cam in range(num_cameras):
        camp = cams_final[cam]
        refined_Ks[cam] = camp["K"]
        Rm = _R_from_rodrigues(camp["rvec"])
        tm = camp["t"].reshape(3, 1)
        refined_Rs[cam] = Rm
        refined_ts[cam] = tm

    # Split points per pair
    refined_points_per_pair = []
    for idx, base in enumerate(point_base_idx):
        npts = points3d_list[idx].shape[0]
        seg = pts_final[base : base + npts, :]
        refined_points_per_pair.append(seg.copy())

    t1 = time.time()
    if verbose:
        print(f"[BA] Done in {t1 - t0:.1f}s | success={lsq_result.success}")
        print(f"[BA] Initial residual norm: {np.linalg.norm(residuals(x0)):.3f}")
        print(f"[BA] Final residual norm:   {np.linalg.norm(residuals(lsq_result.x)):.3f}")
        print(f"[BA] Translation constraint camera {constrained_camera} final |t| = {np.linalg.norm(refined_ts[constrained_camera]):.4f}")

    return refined_Ks, refined_Rs, refined_ts, refined_points_per_pair