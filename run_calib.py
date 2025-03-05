from keypoints_calibration import calibrate_cameras

result = calibrate_cameras(
    openpose_dir=r"D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\FINAL_DATASET\Demo_SinglePerson\pose",
    intrinsics_file=r"D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\GITHUB_CALIBRATION\Calib_scene_best_8pts.toml",
    segments_file=r"D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\GITHUB_CALIBRATION\segments.toml",
    confidence_threshold=0.7,
    output_path=r"D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\Final Calibration Script\New Folder",
    output_filename=r"calibration.toml"
) 