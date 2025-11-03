# Calibration_scripts

The following scripts are for calculating extrinsic parameters using the keypoints output of pose estimators. An initial intrinsic value is needed. The algorithm is based on Liu's paper which some changes that Mr.Pagnon and Mr.Hunminkim suggested. See run_calib.py for an example 

As of 03/11/2025, I've added the CasCalib method and a simple heuristic guess to supplement a good initial intrinsic value. BA_Jax script was also added to further speed up Bundle Adjustment

## How to use

Args:
Supply the needed arg paths in the  form of strings

path_to_openpose_keypoints_dir: Path to where your Keypoints directory. Each subfolder should correspond to a camera, containing per-frame JSON.
path_to_intrinsics_file: TOML file containing the calculated Pose2Sim Checkerboard Intrinsics (You can interrupt the program at extrinsics calculations and it will still generate a toml format with the intrinsics), arg can be None if calc_intrinsics_method is not set to 'default'
path_to_segments_file: TOML file containing the sizes in meters of each segment length
confidence_threshold_keypoints: default is 0.7, override if preferred, this filters out the confidence values of keypoints
path_to_pose2sim_project_dir: Path to your Pose2Sim dir, assumes its similar to Demo_SinglePerson folder in Pose2Sim
output_path_calibration:  Dir Path to set where the calibration TOML file will be outputted. It needs to be set on the calibration folder of the Pose2Sim project dir
output_filename: Desired str file name of the calibration TOML file
img_width: px width of the cameras (Right now assumes all cameras are same)
img_height: px width of the cameras (Right now assumes all cameras are same)

calc_intrinsics_method: 
            Method for calculating intrinsic parameters:
            'default': Uses Pose2Sim’s checkerboard-based calibration (standard method).
            'CasCalib': Uses CasCalib algorithm for intrinsic estimation.
            'heuristic': Uses a heuristic based on image dimensions.
            'Custom': Uses user-provided initial focal length or parameters. NOT YET IMPLEMENTED

        optimization_method: 
            Extrinsic optimization strategy:
            'Liu': Two Staged Liu et al. method.
            'Combined': Hybrid optimization combining multiple refinement stages (Liu and Bundle_Adjustment).


Example Function call. You can run this from run_calib.py
```python
from Calibration.keypoints_calibration import calibrate_cameras

result = calibrate_cameras(
    path_to_keypoints_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose",
    path_to_intrinsics_file=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\copy of calib\Calib_scene.toml",
    path_to_segments_file=r"C:\Users\Miro Hernandez\Documents\Github Projects\Calibration_scripts\segments_Body_25.toml",
    confidence_threshold_keypoints=0.7,
    path_to_pose2sim_project_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T04_Normal",
    output_path_calibration=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\calibration",
    output_filename="calibration.toml",
    img_width=480, 
    img_height=640,
    calc_intrinsics_method='CasCalib',
    optimization_method="Combined",
)
```

