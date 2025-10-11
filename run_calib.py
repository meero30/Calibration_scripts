from Liu_Bundle_Adjustment.keypoints_calibration import calibrate_cameras

# result = calibrate_cameras(
#     openpose_dir=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose",
#     intrinsics_file=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\copy of calib\Calib_scene.toml",
#     segments_file=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\calibration\segments.toml",
#     confidence_threshold=0.7,
#     trc_file_dir = r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\T04_Normal\pose-3d",
#     pose2sim_project_dir = r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\T04_Normal",
#     output_path=r"C:\Users\Miro Hernandez\Documents\Pose2sim Calibration Project\GaitScape\S01_STRC111_New\P07_paper\calibration",
#     output_filename=r"calibration.toml"
# )

# result = calibrate_cameras(
#     openpose_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose",
#     intrinsics_file=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\copy of calib\Calib_scene.toml",
#     segments_file=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\calibration\segments.toml",
#     confidence_threshold=0.7,
#     trc_file_dir = r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose-3d",
#     pose2sim_project_dir = r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless",
#     output_path=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\calibration",
#     output_filename=r"calibration.toml"
# )



result = calibrate_cameras(
    openpose_dir=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose",
    intrinsics_file=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\copy of calib\Calib_scene.toml",
    segments_file=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\calibration\segments.toml",
    confidence_threshold=0.7,
    trc_file_dir = r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T04_Normal\pose-3d",
    pose2sim_project_dir = r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T04_Normal",
    output_path=r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\calibration",
    output_filename=r"calibration.toml",
    img_width=480,
    img_height=640,
    calc_intrinsics_method='CasCalib',
    optimization_method="BundleAdjustment",
)
