
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




def calibrate_cameras(openpose_dir, intrinsics_file, segments_file, 
                      confidence_threshold, trc_file_dir, pose2sim_project_dir, output_path, output_filename, img_width, img_height, calc_intrinsics_method='default', optimization_method='Liu'):
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
                ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(temp_dir)

                Ks = process_alphapose_directory(img_width, img_height, ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS, confidence_threshold)
            
            
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

        print(f"Confidence threshold: {confidence_threshold}")
        
        # Load JSON files from OpenPose directories
        all_cam_data = load_json_files(OPENPOSE_KEYPOINTS_DIRECTORY)
        
        # Select the best reference camera
        final_idx_of_ref_cam, constrained_camera, paired_keypoints_list, inliers_pair_list, inlier2_list, final_camera_Rt = select_reference_camera(
            all_cam_data, OPENPOSE_KEYPOINTS_DIRECTORY, Ks, confidence_threshold
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
            max_points = 200  # adjust as needed

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

# TODO: Clean Unneeded arguments
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