
import time

#from my_bundle_adjustment import CustomBundleAdjustment
from CasCalib.Get_Intrinsics import get_simplified_cam_params
start_time = time.time()
#from write_to_toml import write_to_toml
from pathlib import Path



def process_alphapose_directory(img_width, img_height, alphapose_dir, confidence):
    """
    Processes AlphaPose JSON files to extract camera names and simplified intrinsics.

    Parameters
    ----------
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    alphapose_dir : str or Path
        Path to the directory containing AlphaPose JSON keypoint files. Each file corresponds to a different camera.

    Returns
    -------
    cam_intrinsics_array : any
        The camera intrinsics array returned by get_simplified_cam_params().

        Example Output:
        cam_intrinsics_array =            
        [
            array([[492.70615785,   0.        , 240.        ],
                    [  0.        , 492.70615785, 320.        ],
                    [  0.        ,   0.        ,   1.        ]]),
            array([[366.37723903,   0.        , 240.        ],
                    [  0.        , 366.37723903, 320.        ],
                    [  0.        ,   0.        ,   1.        ]]),
            array([[595.31194042,   0.        , 240.        ],
                    [  0.        , 595.31194042, 320.        ],
                    [  0.        ,   0.        ,   1.        ]])
            ]

    camera_names : list
        List of camera names derived from JSON filenames.
    data_paths : dict
        Dictionary with key 'detections' mapping to full JSON file paths.
    """
    
    # Convert to Path object
    alphapose_dir = Path(alphapose_dir)

    # Principal point (center of the image)
    u0 = img_width / 2
    v0 = img_height / 2

    # Find all JSON files in directory
    json_files = list(alphapose_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {alphapose_dir}")

    # Extract camera names (filenames without extension)
    camera_names = [file_path.stem for file_path in json_files]

    # Create DATA_PATHS dictionary
    DATA_PATHS = {'detections': json_files}

    # Print debug info
    print("Camera Names:", camera_names)
    print("\nDetection Paths:")
    for path in DATA_PATHS['detections']:
        print(path)

    cam_intrinsics_array = get_simplified_cam_params(camera_names, DATA_PATHS, img_width, img_height, confidence)

    print("\nCamera Intrinsics Array:", cam_intrinsics_array)

    return cam_intrinsics_array