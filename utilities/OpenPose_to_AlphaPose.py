#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ########################################################
    ## Convert OpenPose json files to AlphaPose json file  ##
    ########################################################
    
    Converts OpenPose frame-by-frame files to AlphaPose single json file.
        
    Usage: 
    python -m OpenPose_to_AlphaPose -i input_openpose_folder -o output_alphapose_json_file
    OR python -m OpenPose_to_AlphaPose -i input_openpose_folder
'''

#command
# python OpenPose_to_AlphaPose.py -i "D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\Keypoint Calibration\Our_data\json1" -o "D:\Miro Hernandez\Documents\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\CasCalib\OUR_data1.json"

import json
import os
import argparse
from glob import glob

## AUTHORSHIP INFORMATION
__author__ = "Based on David Pagnon's AlphaPose converter"
__copyright__ = "Copyright 2024"
__license__ = "BSD 3-Clause License"
__version__ = "1.0.0"

def OpenPose_to_AlphaPose_func(*args):
    '''
    Converts OpenPose frame-by-frame files to AlphaPose single json file.
        
    Args:
        input_openpose_folder: Folder containing OpenPose JSON files
        output_alphapose_json_file: Output path for the AlphaPose JSON file
    '''
    
    try:
        input_openpose_folder = os.path.realpath(args[0]['input_openpose_folder'])  # invoked with argparse
        if args[0]['output_alphapose_json_file'] == None:
            output_alphapose_json_file = os.path.join(input_openpose_folder, 'alphapose-results.json')
        else:
            output_alphapose_json_file = os.path.realpath(args[0]['output_alphapose_json_file'])
    except:
        input_openpose_folder = os.path.realpath(args[0])  # invoked as a function
        try:
            output_alphapose_json_file = os.path.realpath(args[1])
        except:
            output_alphapose_json_file = os.path.join(input_openpose_folder, 'alphapose-results.json')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_alphapose_json_file), exist_ok=True)

    # Initialize empty list for AlphaPose format
    alpha_format = []
    
    # Get all json files in the input folder
    json_files = sorted(glob(os.path.join(input_openpose_folder, '*.json')))
    
    for json_file in json_files:
        # Get frame number from filename
        frame_num = ''.join(filter(str.isdigit, os.path.splitext(os.path.basename(json_file))[0]))
        
        # Read OpenPose JSON file
        with open(json_file, 'r') as f:
            openpose_data = json.load(f)
            
        # Convert each person in the frame
        for person in openpose_data['people']:
            # Get keypoints
            keypoints = person['pose_keypoints_2d']
            
            # Calculate score as average confidence of all keypoints
            # Confidence values are every third value in OpenPose format
            confidences = keypoints[2::3]
            score = sum(confidences) / len(confidences) if confidences else 0
            
            # Create AlphaPose format entry
            alpha_entry = {
                "image_id": f"{frame_num}.jpg",  # Adding .jpg extension as it's common in AlphaPose
                "category_id": 1,  # 1 for person
                "keypoints": keypoints,
                "score": score
            }
            
            alpha_format.append(alpha_entry)
    
    # Save to AlphaPose format JSON file
    with open(output_alphapose_json_file, 'w') as f:
        json.dump(alpha_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_openpose_folder', required=True, 
                        help='input folder containing OpenPose json files')
    parser.add_argument('-o', '--output_alphapose_json_file', required=False,
                        help='output AlphaPose single json file')
    args = vars(parser.parse_args())
    
    OpenPose_to_AlphaPose_func(args)