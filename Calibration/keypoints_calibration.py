
"""

This script is for calibration. Inputs would be the precalculated intrinsics values from a calibration toml file.
The output is a calibration toml file with the calculated extrinsics value.
The original script is based on Mr. Hunminkim's implementation of Liu's paper with some added as suggested by Dr. Pagnon.  
List of things TODO
1. Translate plane into -x // DONE
2. Create a callback function that saves the best extrinsics values (there might've been better local minima passed)
3. Find another good estimate of intrinsic values 
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


from contextlib import redirect_stdout

from Calibration.Bundle_Adjustment import bundle_adjustment_refine
from Calibration.Data_transformations_helper import run_pose2sim_triangulation
from Calibration.Staged_calibration import optimize_camera_parameters, select_reference_camera
from CasCalib.run_cascalib import process_alphapose_directory
from utilities.Loader import get_latest_trc_file, load_intrinsics_from_toml, load_json_files
from utilities.trc_Xup_to_Yup import trc_Xup_to_Yup_func
from utilities.OpenPose_to_AlphaPose import OpenPose_to_AlphaPose_func
from utilities.write_to_toml import write_to_toml

from Calibration.calculate_scale import calculate_scale_factor, apply_scale_to_results




def calibrate_cameras(path_to_openpose_keypoints_dir, path_to_segments_file, 
                      confidence_threshold_keypoints,  path_to_pose2sim_project_dir, output_path_calibration,output_filename, img_width, img_height, calc_intrinsics_method='default', optimization_method='Liu', path_to_intrinsics_file=None, confidence_threshold_cascalib=0.7):
    """
    Perform hybrid camera calibration using 2D keypoints, 3D markers, and segment definitions.

    Args:
        path_to_openpose_keypoints_dir (str): 
            Path to the directory containing 2D keypoints estimated by OpenPose for all cameras.
            Each subfolder should correspond to a camera, containing per-frame JSON.

        path_to_intrinsics_file (str): 
            Path to the TOML file containing intrinsic camera parameters (if available). 
            If None, intrinsics will be estimated using the method specified by `calc_intrinsics_method`. #TODO: Implement intrinsics estimation if file is None.

        path_to_segments_file (str): 
            Path to the TOML file defining the body segment connections used for skeleton modeling.

        confidence_threshold_keypoints (float): 
            Minimum keypoint confidence score required to include a 2D observation in calibration for 2-stage calibration calculations.

        path_to_pose2sim_project_dir (str): 
            Path to the Pose2Sim project root directory (contains folders like `pose-2d`, `pose-3d`, and `calibration`).
        
        output_path_calibration (str):
            Directory where the calibration TOML file will be saved.
            
        output_filename (str): 
            Name of the TOML file where the final calibration parameters will be saved.

        img_width (int): 
            Image width in pixels for the camera frames.

        img_height (int): 
            Image height in pixels for the camera frames.

        calc_intrinsics_method (str, optional): 
            Method for calculating intrinsic parameters:
            - `'default'`: Uses Pose2Sim’s checkerboard-based calibration (standard method).
            - `'CasCalib'`: Uses CasCalib algorithm for intrinsic estimation.
            - `'Custom'`: Uses user-provided initial focal length or parameters.

        optimization_method (str, optional): 
            Extrinsic optimization strategy:
            - `'Liu'`: Single-stage Liu et al. method.
            - `'Combined'`: Hybrid optimization combining multiple refinement stages (recommended).

        confidence_threshold_cascalib (float, optional): 
            Confidence threshold used specifically for CasCalib-based intrinsic estimation.

    Returns:
        bool: True if calibration succeeded, False otherwise.
    """

    start_time = time.time()
    
    try:
        # Create Path objects
        # ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(alphapose_dir)
        ROOT_PATH_FOR_OPENPOSE_KEYPOINTS = Path(path_to_openpose_keypoints_dir)
        
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
            print(f"No subdirectories found in OpenPose directory: {path_to_openpose_keypoints_dir}")
            return False
        
        print("Found directories:", OPENPOSE_KEYPOINTS_DIRECTORY)
        
        # TODO: Make a decision tree for intrinsics calculation method
        if calc_intrinsics_method == 'default':
            print("Using Pose2Sim Checkerboard method for intrinsics calculation")
            
            # Temporary, assumes the current toml file has the intrinsics
            
            # Load camera intrinsics from TOML file
            Ks, _ = load_intrinsics_from_toml(path_to_intrinsics_file)
            if Ks is None:
                print("Failed to load intrinsics from TOML file")
                return False
            
        elif calc_intrinsics_method == 'CasCalib':
            print("Using CasCalib method for intrinsics calculation")

            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Temporary output folder for alphapose conversion: {temp_dir}")

                # Loop through all OpenPose keypoint directories
                for i, path_to_openpose_keypoints_dir in enumerate(OPENPOSE_KEYPOINTS_DIRECTORY):
                    output_file = os.path.join(temp_dir, f"alphapose_results_cam{i+1}.json")

                    # Call the converter function directly
                    OpenPose_to_AlphaPose_func(path_to_openpose_keypoints_dir, output_file)

                    print(f"Converted: {path_to_openpose_keypoints_dir}")
                    print(f"Output: {output_file}")
                # TODO: Call CasCalib function here to get Ks
                ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(temp_dir)

                Ks = process_alphapose_directory(img_width, img_height, ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS, confidence_threshold_cascalib)
            
            
            if Ks is None:
                print("Failed to calculate intrinsics using CasCalib method")
                return False
            

        elif calc_intrinsics_method == 'Custom':

            #TODO: Implement custom intrinsics calculation method
            print("Using user-provided initial estimate for intrinsics calculation")
            # Not yet implemented, skipping for now
            print("Custom intrinsics calculation method is not yet implemented")
            return False


        print("Intrinsic Parameters:", Ks)

        image_size = [img_width, img_height]

        print(f"Confidence threshold keypoints: {confidence_threshold_keypoints}")
        
        # Load JSON files from OpenPose directories
        all_cam_data = load_json_files(OPENPOSE_KEYPOINTS_DIRECTORY)
        
        # Select the best reference camera
        final_idx_of_ref_cam, constrained_camera, paired_keypoints_list, inliers_pair_list, inlier2_list, final_camera_Rt = select_reference_camera(
            all_cam_data, OPENPOSE_KEYPOINTS_DIRECTORY, Ks, confidence_threshold_keypoints
        )
        
        if optimization_method == 'Liu':
        # Optimize camera parameters
            all_best_results = optimize_camera_parameters(
                final_idx_of_ref_cam, final_camera_Rt, Ks, inliers_pair_list, inlier2_list, constrained_camera
            )
        elif optimization_method == 'BundleAdjustment':
            # --- Subsample points for faster BA ---
            max_points = 200  # adjust as needed

            for i in range(len(inliers_pair_list)):
                pts = np.asarray(inliers_pair_list[i])
                if pts.shape[0] > max_points:
                    idx = np.random.choice(pts.shape[0], max_points, replace=False)
                    inliers_pair_list[i] = pts[idx]
                    if isinstance(inlier2_list[i], (list, np.ndarray)):
                        inlier2_list[i] = np.asarray(inlier2_list[i])[idx]
            
            refined_Ks, refined_Rs, refined_ts, _ = bundle_adjustment_refine(
                Ks,
                final_idx_of_ref_cam,
                final_camera_Rt,
                inliers_pair_list,
                inlier2_list,
                optimize_intrinsics=True,
                constrained_camera=constrained_camera,
                constraint_weight=1000,
                max_nfev=100,
                verbose=1,
            )
            # Build all_best_results from BA outputs
            all_best_results = {}
            ref_cam = final_idx_of_ref_cam  # e.g., 0

            for j in range(len(refined_Ks)):
                if j == ref_cam:
                    continue  # skip reference camera
                
                pair_key = f"Camera{ref_cam}_{j}"
                all_best_results[pair_key] = {
                    "K1": refined_Ks[ref_cam],
                    "K2": refined_Ks[j],
                    "R": refined_Rs[j],
                    "t": refined_ts[j].flatten(),   # flatten to 1D like in your example
                    "error": np.float64(0.0)        # placeholder if you don’t compute reprojection error
                }

            print("\nPackaged all_best_results for TOML export:")
            for key, val in all_best_results.items():
                print(f"{key}:")
                print("K1:\n", val["K1"])
                print("K2:\n", val["K2"])
                print("R:\n", val["R"])
                print("t:", val["t"])
        
        elif optimization_method == 'Combined':
            # Implement combined optimization logic here

            # Optimize camera parameters
            all_best_results = optimize_camera_parameters(
                final_idx_of_ref_cam, final_camera_Rt, Ks, inliers_pair_list, inlier2_list, constrained_camera
            )
            # for debugging
            inliers_pair_list_copy = inliers_pair_list.copy()

            print ("Initial all_best_results from Liu optimization:", all_best_results)
            print("Ks before BA:", Ks)
            print("Final camera Rt before BA:", final_camera_Rt)

            Ks_original = Ks
            # Extract Ks and final_camera_Rt from all_best_results
            num_cams = len(Ks_original)               # use the original Ks loaded from file earlier
            Ks_new = [None] * num_cams
            final_camera_Rt = {}

            # parse pair keys like "Camera0_2" and place K1/K2 into the correct index slots
            for pair_key, res in all_best_results.items():
                # parse indices
                if not pair_key.startswith("Camera"):
                    continue
                # remove "Camera" and split "0_2" -> ['0','2']
                try:
                    after = pair_key.replace("Camera", "")
                    ref_idx_str, cam_j_str = after.split("_")
                    ref_idx = int(ref_idx_str)
                    cam_j = int(cam_j_str)
                except Exception:
                    # if non-standard key, skip or handle differently
                    continue

                # store intrinsics at their explicit indices
                Ks_new[ref_idx] = res["K1"]
                Ks_new[cam_j]  = res["K2"]

                # store extrinsics for camera j (relative to ref_idx)
                t_arr = np.array(res["t"])
                # normalize shape to (3,1) or (3,) as you prefer — triangulation code accepts either usually
                final_camera_Rt[cam_j] = [np.array(res["R"]), t_arr.reshape(3, 1)]

            # Fill any Ks that were not written (fall back to original)
            for i in range(num_cams):
                if Ks_new[i] is None:
                    Ks_new[i] = Ks_original[i].copy()

            Ks = Ks_new

            # BA refinement
            max_points = 300  # adjust as needed

            for i in range(len(inliers_pair_list)):
                pts = np.asarray(inliers_pair_list[i])

                # skip if points are fewer than or equal to max_points
                if pts.shape[0] <= max_points:
                    continue

                # downsample if more than max_points
                idx = np.random.choice(pts.shape[0], max_points, replace=False)
                inliers_pair_list[i] = pts[idx]

                if isinstance(inlier2_list[i], (list, np.ndarray)):
                    inlier2_list[i] = np.asarray(inlier2_list[i])[idx]

            print("Stop")
            refined_Ks, refined_Rs, refined_ts, _ = bundle_adjustment_refine(
                Ks,
                final_idx_of_ref_cam,
                final_camera_Rt,
                inliers_pair_list,
                inlier2_list,
                optimize_intrinsics=True,
                constrained_camera=constrained_camera,
                constraint_weight=1000,
                max_nfev=1000,
                verbose=1,
            )
            # Build all_best_results from BA outputs
            all_best_results = {}
            ref_cam = final_idx_of_ref_cam  # e.g., 0

            for j in range(len(refined_Ks)):
                if j == ref_cam:
                    continue  # skip reference camera
                
                pair_key = f"Camera{ref_cam}_{j}"
                all_best_results[pair_key] = {
                    "K1": refined_Ks[ref_cam],
                    "K2": refined_Ks[j],
                    "R": refined_Rs[j],
                    "t": refined_ts[j].flatten(),   # flatten to 1D like in your example
                    "error": np.float64(0.0)        # placeholder if you don’t compute reprojection error
                }

            print("\nPackaged all_best_results for TOML export:")
            for key, val in all_best_results.items():
                print(f"{key}:")
                print("K1:\n", val["K1"])
                print("K2:\n", val["K2"])
                print("R:\n", val["R"])
                print("t:", val["t"])


        print("All best results:", all_best_results)

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
        #output_path_calibration = os.path.join(path_to_pose2sim_project_dir, 'calibration')
        # Write results to TOML file
        write_to_toml(
            all_best_results, 
            [0.0, 0.0, 0.0], 
            image_size, 
            output_path=output_path_calibration, 
            output_filename=output_filename
        )

        run_pose2sim_triangulation(path_to_pose2sim_project_dir)
        trc_file_dir = os.path.join(path_to_pose2sim_project_dir, 'pose-3d')
        latest_trc_file = get_latest_trc_file(trc_file_dir)

        # TODO: ADD THE APPLY SCALE FUNCTION HERE
        scale_factor = calculate_scale_factor(path_to_segments_file, latest_trc_file)
        all_best_results = apply_scale_to_results(all_best_results, scale_factor, final_idx_of_ref_cam)
        

        write_to_toml(
        all_best_results, 
        [0.0, 0.0, 0.0], 
        image_size, 
        output_path=output_path_calibration, 
        output_filename=output_filename
        )
        # Print final optimized results
        print("\nFinal optimized camera parameters:")
        for pair_key, results in all_best_results.items():
            print(f"Results for {pair_key}:")
            print(f"- t: {results['t']}")

        run_pose2sim_triangulation(path_to_pose2sim_project_dir)

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

# TODO: Clean Unneeded arguments
def main():
    """
    Parse command-line arguments and run the calibration.
    """
    parser = argparse.ArgumentParser(description='Hybrid camera calibration from keypoints')
    

    parser.add_argument('--path_to_openpose_keypoints_dir', type=str, required=True,
                        help='Path to the directory containing 2D keypoints estimated by OpenPose for all cameras.')

    parser.add_argument('--path_to_intrinsics_file', type=str, required=True,
                        help='Path to the TOML file containing camera intrinsic parameters. If not provided, intrinsics will be estimated.')

    parser.add_argument('--path_to_segments_file', type=str, required=True,
                        help='Path to the TOML file defining body segment connections for the skeleton model.')

    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Minimum keypoint confidence score for including 2D observations in calibration (default: 0.7).')

    parser.add_argument('--path_to_trc_file_dir', type=str, required=True,
                        help='Path to the directory containing .trc files (3D trajectories) used for extrinsic calibration.')

    parser.add_argument('--path_to_pose2sim_project_dir', type=str, required=True,
                        help='Path to the Pose2Sim project root directory containing pose-2d, pose-3d, and calibration folders.')

    parser.add_argument('--path_to_output_dir', type=str, default=".",
                        help='Directory to save the resulting calibration files and intermediate outputs (default: current directory).')

    parser.add_argument('--output_filename', type=str, default="calibration.toml",
                        help='Name of the output TOML file where calibration parameters will be saved (default: calibration.toml).')

    parser.add_argument('--img_width', type=int, required=True,
                        help='Image width in pixels for the input video frames.')

    parser.add_argument('--img_height', type=int, required=True,
                        help='Image height in pixels for the input video frames.')

    parser.add_argument('--calc_intrinsics_method', type=str, default='default',
                        choices=['default', 'CasCalib', 'Custom'],
                        help=('Method for calculating intrinsic parameters: '
                            '"default" uses Pose2Sim’s checkerboard calibration, '
                            '"CasCalib" uses CasCalib-based estimation, '
                            '"Custom" uses user-provided initial focal length (default: default).'))

    parser.add_argument('--optimization_method', type=str, default='Combined',
                        choices=['Liu', 'BundleAdjustment', 'Combined'],
                        help=('Optimization method for extrinsics: '
                            '"Liu" for Liu’s method, '
                            '"BundleAdjustment" for full bundle adjustment, '
                            '"Combined" for hybrid optimization (default: Combined).'))

    
    
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