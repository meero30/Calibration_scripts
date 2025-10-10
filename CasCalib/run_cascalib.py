
import time

#from my_bundle_adjustment import CustomBundleAdjustment
from Get_Intrinsics import get_simplified_cam_params
start_time = time.time()
#from write_to_toml import write_to_toml
from pathlib import Path
import glob
import argparse





# Use the parsed arguments
img_width = 480
img_height = 640
image_size = [img_width, img_height]

u0 = image_size[0] / 2  # principal point u0
v0 = image_size[1] / 2  # principal point v0

#confidence_threshold = 0.7 # used for keypoints filtering
# Create Path object
# ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(
#     r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\CasCalib\Hunminkim_data_alphapose"
# )

ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = Path(
    r"C:\Users\Miro Hernandez\Documents\Github Projects\Calibration_scripts\CasCalib\pose_alphapose"
)

# # Create Path objects from command line arguments
# ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS = r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\CasCalib\Hunminkim_data_alphapose"


# Find all JSON files
json_files = list(ROOT_PATH_FOR_ALPHAPOSE_KEYPOINTS.glob('*.json'))

# Extract camera names from filenames (just the base filename without extension)
camera_names = [file_path.stem for file_path in json_files]


# Create DATA_PATHS (full paths to all JSON files)
DATA_PATHS = {
    'detections': json_files
}

# Print to verify
print("Camera Names:", camera_names)
print("\nDetection Paths:")
for path in DATA_PATHS['detections']:
    print(path)




cam_intrinsics_array = get_simplified_cam_params(camera_names, DATA_PATHS, img_width, img_height)
print("Camera Intrinsics Array:", cam_intrinsics_array)