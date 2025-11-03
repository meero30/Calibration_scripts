# Calibration_scripts

The following scripts are for calculating **extrinsic parameters** using the keypoints output of pose estimators.  
An initial intrinsic value is needed.  

The algorithm is based on **Liu's paper** with some changes suggested by **Mr. Pagnon** and **Mr. Hunminkim**.  
See `run_calib.py` for an example.

As of **03/11/2025**, I've added the **CasCalib** method and a simple **heuristic guess** to supplement a good initial intrinsic value.  
`BA_Jax` script was also added to further speed up **Bundle Adjustment**.

---

## How to use

### Args

Supply the needed arg paths in the form of strings:

- **`path_to_openpose_keypoints_dir`**  
  Path to where your Keypoints directory.  
  Each subfolder should correspond to a camera, containing per-frame JSON files.

- **`path_to_intrinsics_file`**  
  TOML file containing the calculated Pose2Sim Checkerboard Intrinsics.  
  (You can interrupt the Pose2Sim calibration program at extrinsics calculations and it will still generate a TOML format with the intrinsics.)  
  Argument can be `None` if `calc_intrinsics_method` is not set to `'default'`.

- **`path_to_segments_file`**  
  TOML file containing the sizes in meters of each segment length.

- **`confidence_threshold_keypoints`**  
  Default is `0.7`. Override if preferred.  
  This filters out the confidence values of keypoints.

- **`path_to_pose2sim_project_dir`**  
  Path to your Pose2Sim directory.  
  Assumes it’s similar to the `Demo_SinglePerson` folder in Pose2Sim.

- **`output_path_calibration`**  
  Directory path to set where the calibration TOML file will be outputted.  
  It needs to be set inside the calibration folder of the Pose2Sim project directory.

- **`output_filename`**  
  Desired string filename of the calibration TOML file.

- **`img_width`**  
  Pixel width of the cameras (currently assumes all cameras are the same).

- **`img_height`**  
  Pixel height of the cameras (currently assumes all cameras are the same).

---

### `calc_intrinsics_method`

Method for calculating intrinsic parameters:

- `'default'`: Uses Pose2Sim’s checkerboard-based calibration (standard method).  
- `'CasCalib'`: Uses CasCalib algorithm for intrinsic estimation.  
- `'heuristic'`: Uses a heuristic based on image dimensions.  
- `'Custom'`: Uses user-provided initial focal length or parameters. **(NOT YET IMPLEMENTED)**

---

### `optimization_method`

Extrinsic optimization strategy:

- `'Liu'`: Two-Staged Liu et al. method.  
- `'Combined'`: Hybrid optimization combining multiple refinement stages (Liu and Bundle Adjustment).

---

## Example Function Call

You can run this from `run_calib.py`:

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
