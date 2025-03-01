import os
import json




# # This script assumes that all json directories containt the same number of frames / json files

# def extract_paired_keypoints_with_reference(ref_cam_dir, other_cam_dirs, confidence_threshold):
#     """
#     Extracts paired keypoints (x, y) with reference camera and each of the other cameras.

#     Args:
#     - ref_cam_dir: Directory containing JSON files for the reference camera.
#     - other_cam_dirs: List of directories containing JSON files for other cameras.
#     - confidence_threshold: Confidence value threshold for keypoints.

#     Returns:
#     - all_paired_keypoints: A list containing paired keypoints for each camera pair.
#     """
#     all_paired_keypoints = []

#     # Load and sort JSON files from the reference camera directory
#     ref_cam_files = sorted([os.path.join(ref_cam_dir, f) for f in os.listdir(ref_cam_dir) if f.endswith('.json')])

#     index_of_faulty_json = []

#     for cam_dir in other_cam_dirs:
#         # Load and sort JSON files from the current camera directory
#         cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])

#         paired_keypoints_list = []

#         i = 0
#         for ref_file, cam_file in zip(ref_cam_files, cam_files):
#             if i in index_of_faulty_json:
#                 i += 1
#                 continue

#             with open(ref_file, 'r') as file1, open(cam_file, 'r') as file2:
#                 ref_data = json.load(file1)
#                 cam_data = json.load(file2)
 
#                 try:
#                     ref_keypoints = ref_data['people'][0]['pose_keypoints_2d']
#                     cam_keypoints = cam_data['people'][0]['pose_keypoints_2d']

#                     # Extract keypoints with confidence
#                     ref_keypoints_conf = [(ref_keypoints[i], ref_keypoints[i+1], ref_keypoints[i+2]) for i in range(0, len(ref_keypoints), 3)]
#                     cam_keypoints_conf = [(cam_keypoints[i], cam_keypoints[i+1], cam_keypoints[i+2]) for i in range(0, len(cam_keypoints), 3)]

#                     # Filter keypoints based on confidence threshold and pair them
#                     paired_keypoints = [((kp1[0], kp1[1]), (kp2[0], kp2[1])) for kp1, kp2 in zip(ref_keypoints_conf, cam_keypoints_conf) if kp1[2] >= confidence_threshold and kp2[2] >= confidence_threshold]
#                     # e.g. [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...] per frame

#                     # "if paired_keypoints is not empty", below is a pythonic way of doing this.
#                     if paired_keypoints:
#                         paired_keypoints_list.append(paired_keypoints)
#                     else:
#                         print(f"No paired keypoints found for frame {i}.")
#                     i += 1
#                 except:
#                     index_of_faulty_json.append(i)
#                     i += 1
#                     continue
                
#         all_paired_keypoints.append(paired_keypoints_list)
#         if index_of_faulty_json:
#             print("Warning: Faulty JSON files found for frames:", index_of_faulty_json)
#         # break
#     return all_paired_keypoints



def extract_paired_keypoints_with_reference(all_cam_data, index_of_ref, confidence_threshold):
    """
    Extracts paired keypoints (x, y) with reference camera and each of the other cameras.

    Args:
    - all_cam_dirs: List of directories containing JSON files for all cameras.
    - index_of_ref: Index of the reference camera in the list of all cameras.
    - confidence_threshold: Confidence value threshold for keypoints.

    Returns:
    - all_paired_keypoints: A list containing paired keypoints for each camera pair.
    """

    all_paired_keypoints = []
    index_of_faulty_json = []
    
    for idx_of_other_cams , cams in enumerate(all_cam_data): # For each  in all_cam_data:
        if idx_of_other_cams == index_of_ref:
            continue
        paired_keypoints_list = []
        for j in range(len(cams)): # each element is a frame
            ref_data = all_cam_data[index_of_ref][j]  # single frame
            cam_data = all_cam_data[idx_of_other_cams][j] # single frame
            try:
                ref_keypoints = ref_data['people'][0]['pose_keypoints_2d']
                ref_keypoints_conf = [(ref_keypoints[i], ref_keypoints[i+1], ref_keypoints[i+2]) for i in range(0, len(ref_keypoints), 3)]
                cam_keypoints = cam_data['people'][0]['pose_keypoints_2d']
                cam_keypoints_conf = [(cam_keypoints[i], cam_keypoints[i+1], cam_keypoints[i+2]) for i in range(0, len(cam_keypoints), 3)]
                paired_keypoints = [((kp1[0], kp1[1]), (kp2[0], kp2[1])) for kp1, kp2 in zip(ref_keypoints_conf, cam_keypoints_conf) if kp1[2] >= confidence_threshold and kp2[2] >= confidence_threshold]
                
                # e.g. [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...] per frame
                
                if paired_keypoints:
                    paired_keypoints_list.append(paired_keypoints)
                # else:
                #     print(f"No paired keypoints found for frame {i}.")
            except:
                index_of_faulty_json.append(j)
                continue
        all_paired_keypoints.append(paired_keypoints_list)
    if index_of_faulty_json:
        print("Warning: Faulty JSON files found for frames:", index_of_faulty_json)        


    return all_paired_keypoints