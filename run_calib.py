from keypoints_calibration import calibrate_cameras

result = calibrate_cameras(
    openpose_dir=r".\Demo_SinglePerson\pose",
    intrinsics_file=r".\Calib_scene_best_8pts.toml",
    segments_file=r".\segments.toml",
    confidence_threshold=0.7,
    output_path=r".\New Folder",
    output_filename=r"calibration.toml"
)