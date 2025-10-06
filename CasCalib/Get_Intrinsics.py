
import json
from datetime import datetime
from run_calibration_ransac import run_calibration_ransac
import data

def get_simplified_cam_params(camera_names, DATA_PATHS, img_width, img_height):
    today = datetime.now()
    name = str(today.strftime('%Y%m%d_%H%M%S')) + '_custom_calibration'
    
    
    
    cam_intrinsics_array = []
    
    # First pass: Get camera calibration and 2D poses
    for num, vid in enumerate(camera_names):
        # Load 2D points
        
        with open(DATA_PATHS['detections'][num], 'r') as f:
            points_2d = json.load(f)
        print("Loaded 2D keypoints")
        # Get calibration data
        datastore_cal = data.alphapose_dataloader(points_2d)
        (ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, 
         ransac_focal, datastore_filtered) = run_calibration_ransac(
            datastore_cal, 'hyperparameter.json', None,
            img_width, img_height, 
            confidence=0.7,
            skip_frame=1, # use every frame
            max_len=10000,
            min_size=10,
            save_dir=None
        )

        print("Calibration RANSAC Completed")
        
        cam_intrinsics_array.append(cam_matrix)
        
    
    return cam_intrinsics_array