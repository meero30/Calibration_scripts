from Calibration.keypoints_calibration import calibrate_cameras



# Pose2Sim Demo
# result = calibrate_cameras(
#     path_to_openpose_keypoints_dir=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\Demo_SinglePerson\pose-sync",
#     path_to_intrinsics_file=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\Demo_SinglePerson\calibration\extrinsics\Calib_qualisys.toml",
#     path_to_segments_file=r"C:\Users\Miro Hernandez\Documents\Github Projects\Calibration_scripts\segments_Body_25.toml",
#     confidence_threshold_keypoints=0.7,
#     path_to_pose2sim_project_dir=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\Demo_SinglePerson",
#     output_path_calibration=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\Demo_SinglePerson\calibration",
#     output_filename="calibration.toml",
#     img_width=1080,
#     img_height=1920,
#     calc_intrinsics_method='default',
#     optimization_method="Liu",
# )
# TODO: Implement an automatic img_width and img_height detection based on a video frame.
# My Data
result = calibrate_cameras(
    path_to_openpose_keypoints_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose",
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

# Tom Cheng Data

# result = calibrate_cameras(
#     path_to_openpose_keypoints_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\tom chengs data\pose",
#     path_to_intrinsics_file=None,
#     path_to_segments_file=r"C:\Users\Miro Hernandez\Documents\Github Projects\Calibration_scripts\segments_Body_25.toml",
#     confidence_threshold_keypoints=0.7,
#     path_to_pose2sim_project_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\tom chengs data",
#     output_path_calibration=r"C:\Users\Miro Hernandez\Documents\GaitScape\tom chengs data\calibration",
#     output_filename="calibration.toml",
#     img_width=2560,
#     img_height=1440,
#     calc_intrinsics_method='CasCalib',
#     optimization_method="Combined",
# )
