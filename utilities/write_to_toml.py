from tomlkit import document, table, array
import cv2
import os
import re


def nparray_to_list(nparray):
    """Convert NumPy array to a nested list of Python floats."""
    return [[float(v) for v in row] for row in nparray]


def parse_pair_key(pair_key):
    """Extract numeric camera indices from a pair_key like 'Camera5_10'."""
    nums = re.findall(r'\d+', pair_key)
    if len(nums) == 2:
        return int(nums[0]), int(nums[1])
    else:
        raise ValueError(f"Invalid pair_key format: {pair_key}")


def write_to_toml(all_best_results, t, set_size, output_path=".", output_filename="output.toml"):
    # Initialize the TOML document
    doc = document()
    do_once = False

    for pair_key, results in all_best_results.items():
        cam1, cam2 = parse_pair_key(pair_key)

        # Add reference camera (only once)
        if not do_once:
            camera_data = table()
            camera_data.add("name", f"int_cam{cam1}_img")
            camera_data.add("size", array([float(v) for v in set_size]))
            camera_data.add("matrix", nparray_to_list(all_best_results[pair_key]['K1']))
            camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
            camera_data.add("rotation", array([0.0, 0.0, 0.0]))
            camera_data.add("translation", array([float(v) for v in t]))
            camera_data.add("fisheye", False)
            doc.add(f"int_cam{cam1}_img", camera_data)
            do_once = True

        # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(results['R'])

        camera_data = table()
        camera_data.add("name", f"int_cam{cam2}_img")
        camera_data.add("size", array([float(v) for v in set_size]))
        camera_data.add("matrix", nparray_to_list(results['K2']))
        camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
        camera_data.add("rotation", array([float(v) for v in rvec.squeeze().tolist()]))
        camera_data.add("translation", array([float(v) for v in results['t']]))
        camera_data.add("fisheye", False)

        # Add this camera section to the TOML document
        doc.add(f"int_cam{cam2}_img", camera_data)

    # Add metadata section
    metadata = table()
    metadata.add("adjusted", False)
    metadata.add("error", 0.0)
    doc.add("metadata", metadata)

    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, output_filename)

    # Write TOML to file
    with open(output_file, "w") as toml_file:
        toml_file.write(doc.as_string())

# def nparray_to_list(nparray):
#     """Convert NumPy array to a nested list of Python floats."""
#     return [[float(v) for v in row] for row in nparray]

# def write_to_toml(all_best_results, t, set_size, output_path=".", output_filename="output.toml"):
#     # Initialize the TOML document
#     doc = document()
#     do_once = False

#     for pair_key, results in all_best_results.items():
#         # Add reference camera (only once)
#         if not do_once:
#             camera_data = table()
#             camera_data.add("name", f"int_cam{pair_key[6]}_img")
#             camera_data.add("size", array([float(v) for v in set_size]))
#             camera_data.add("matrix", nparray_to_list(all_best_results[pair_key]['K1']))
#             camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#             camera_data.add("rotation", array([0.0, 0.0, 0.0]))
#             camera_data.add("translation", array([float(v) for v in t]))
#             camera_data.add("fisheye", False)
#             doc.add(f"int_cam{pair_key[6]}_img", camera_data)
#             do_once = True

#         # Convert rotation matrix to Rodrigues vector
#         rvec, _ = cv2.Rodrigues(results['R'])

#         camera_data = table()
#         camera_data.add("name", f"int_cam{pair_key[-1]}_img")
#         camera_data.add("size", array([float(v) for v in set_size]))
#         camera_data.add("matrix", nparray_to_list(results['K2']))
#         camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#         camera_data.add("rotation", array([float(v) for v in rvec.squeeze().tolist()]))
#         camera_data.add("translation", array([float(v) for v in results['t']]))
#         camera_data.add("fisheye", False)

#         # Add this camera section to the TOML document
#         doc.add(f"int_cam{pair_key[-1]}_img", camera_data)

#     # Add metadata section
#     metadata = table()
#     metadata.add("adjusted", False)
#     metadata.add("error", 0.0)
#     doc.add("metadata", metadata)

#     # Create output directory if needed
#     os.makedirs(output_path, exist_ok=True)
#     output_file = os.path.join(output_path, output_filename)

#     # Write TOML to file
#     with open(output_file, "w") as toml_file:
#         toml_file.write(doc.as_string())


# def nparray_to_list(nparray):
#     return [list(row) for row in nparray]

# def write_to_toml(all_best_results, t, set_size, output_path=".", output_filename="output.toml"):
#     # Initialize the TOML document
    
#     doc = document()

#     # Create camera data for reference camera
#     do_once = False
#     # Create camera data for the rest of the other cameras
#     for pair_key, results in all_best_results.items():
#         if not do_once: # Add reference camera once
#             camera_data = table()
#             camera_data.add("name", f"int_cam{pair_key[6]}_img")
#             camera_data.add("size", array(set_size))
#             camera_data.add("matrix", array(nparray_to_list(all_best_results[pair_key]['K1'])))
#             camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#             camera_data.add("rotation", array([0.0, 0.0, 0.0]))
#             camera_data.add("translation", array(t))
#             camera_data.add("fisheye", False)
#             # Adds the camera data to the TOML document
#             doc.add(f"int_cam{pair_key[6]}_img", camera_data)
#             do_once = True
        
#         # Convert rotation matrix to Rodrigues vector
#         rvec, _ = cv2.Rodrigues(results['R'])
#         # 3840.0, 2160.0
#         camera_data = table()
#         camera_data.add("name", f"int_cam{pair_key[-1]}_img") # ex: string pair_key = Camera0_1 where pair_key[-1] is the last character '1'
#         camera_data.add("size", array(set_size))
#         camera_data.add("matrix", array(nparray_to_list(results['K2'])))
#         camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#         camera_data.add("rotation", array(list(rvec.squeeze())))
#         camera_data.add("translation", array(list(results['t'])))
#         camera_data.add("fisheye", False)
        

#         doc.add(f"int_cam{pair_key[-1]}_img", camera_data)

#     # Add metadata
#     metadata = table()
#     metadata.add("adjusted", False)
#     metadata.add("error", 0.0)
#     doc.add("metadata", metadata)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_path, exist_ok=True)

#     output_file = os.path.join(output_path, output_filename)

#     # Write toml to file
#     with open(output_file, "w") as toml_file:
#         toml_file.write(doc.as_string())


# def write_to_toml(all_best_results, reference_t, constrained_t, constrained_camera, set_size, output_path=".", output_filename="output.toml"):
#     # Initialize the TOML document
#     doc = document()

#     # Create camera data for reference camera
#     do_once = False
#     # Create camera data for the rest of the other cameras
#     for pair_key, results in all_best_results.items():
#         if not do_once: # Add reference camera once
#             camera_data = table()
#             camera_data.add("name", f"int_cam{pair_key[6]}_img")
#             camera_data.add("size", array(set_size))
#             camera_data.add("matrix", array(nparray_to_list(all_best_results[pair_key]['K1'])))
#             camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#             camera_data.add("rotation", array([0.0, 0.0, 0.0]))
#             camera_data.add("translation", array(reference_t))
#             camera_data.add("fisheye", False)
#             # Adds the camera data to the TOML document
#             doc.add(f"int_cam{pair_key[6]}_img", camera_data)
#             do_once = True
        
#         # Convert rotation matrix to Rodrigues vector
#         rvec, _ = cv2.Rodrigues(results['R'])
        
#         camera_data = table()
#         current_camera = int(pair_key[-1])  # Get the current camera number
#         camera_data.add("name", f"int_cam{current_camera}_img")
#         camera_data.add("size", array(set_size))
#         camera_data.add("matrix", array(nparray_to_list(results['K2'])))
#         camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
#         camera_data.add("rotation", array(list(rvec.squeeze())))
        
#         # Set translation based on whether this is the constrained camera
#         if current_camera == constrained_camera:
#             camera_data.add("translation", array(constrained_t))
#         else:
#             camera_data.add("translation", array(list(results['t'])))
            
#         camera_data.add("fisheye", False)

#         doc.add(f"int_cam{current_camera}_img", camera_data)

#     # Add metadata
#     metadata = table()
#     metadata.add("adjusted", False)
#     metadata.add("error", 0.0)
#     doc.add("metadata", metadata)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_path, exist_ok=True)

#     output_file = os.path.join(output_path, output_filename)

#     # Write toml to file
#     with open(output_file, "w") as toml_file:
#         toml_file.write(doc.as_string())