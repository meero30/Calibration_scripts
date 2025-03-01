# Calibration_scripts

The following scripts are for calculating extrinsic parameters using the keypoints output of pose estimators. An initial intrinsic value is needed. The algorithm is based on Liu's paper which some changes that Mr.Pagnon and Mr.Hunminkim suggested. See run_calib.py for an example 

## How to use
#### CLI
```
python hybrid_calibration.py --openpose_dir <path_to_openpose_data> \
                            --intrinsics_file <path_to_intrinsics.toml> \
                            --segments_file <path_to_segments.toml> \
                            --confidence 0.7 \
                            --output_path <output_directory> \
                            --output_filename calibration.toml

```


#### Python 
```python
from keypoints_calibration import calibrate_cameras

result = calibrate_cameras(
    alphapose_dir="path/to/alphapose_data",
    openpose_dir="path/to/openpose_data",
    intrinsics_file="path/to/intrinsics.toml",
    segments_file="path/to/segments.toml",
    confidence_threshold=0.7,
    output_path="output_dir",
    output_filename="calibration.toml"
)

