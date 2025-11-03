import numpy as np

def trc_file_to_structured_points_3d(trc_file_path):
    """
    Converts a TRC file to a structured list of 3D points.
    
    Args:
        trc_file_path (str): Path to the TRC file.
        
    Returns:
        list: A list where each element represents a frame. Each frame is a numpy 
        array of shape (21,1) containing numpy arrays of shape (3,1) for each keypoint's
        X, Y, Z coordinates.
    """
    print(f"Opening TRC file: {trc_file_path}")
    
    # Read the TRC file
    with open(trc_file_path, 'r') as file:
        lines = file.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Print first few header lines for verification
    print("\nHeader information:")
    for i in range(min(5, len(lines))):
        print(f"Line {i}: {lines[i].strip()}")

    # Parse header information
    header_info = lines[2].strip().split()
    num_frames = int(header_info[2])
    num_markers = int(header_info[3])
    
    print(f"\nDetected {num_frames} frames and {num_markers} markers from header")
    
    # Verify that we have the expected number of markers (21)
    # assert num_markers == 21, f"Expected 21 markers, but found {num_markers} in the TRC file"
    # Modified to be flexible with number of markers
    # Create the structured_points_3d list
    structured_points_3d = []
    
    # Start reading from line 5 (0-indexed) which contains the first frame's data
    print("\nProcessing frames...")
    
    for i in range(5, min(10, len(lines))):  # Process first 5 frames for testing
        line = lines[i].strip()
        if not line:  # Skip empty lines
            print(f"Skipping empty line at index {i}")
            continue
            
        print(f"\nSample data from frame {i-5} (line {i}):")
        print(line[:100] + "..." if len(line) > 100 else line)  # Print first 100 chars
            
        # Split the line into values
        values = line.split()
        
        # Show frame number and time
        print(f"Frame#: {values[0]}, Time: {values[1]}")
        
        # Skip the first two columns (Frame# and Time)
        data_values = values[2:]

        print(f"Total data values for keypoints: {len(data_values)}")
        
        # Old Check for 21 markers
        # if len(data_values) != 63:
        #     print(f"Warning: Frame {i-5} has {len(data_values)} values instead of 63")
        #     continue

        # Create a frame array to hold the num_markers keypoints
        frame_keypoints = np.zeros((num_markers, 1), dtype=object)
        
        # Show sample data for first 3 keypoints
        print("\nSample keypoint data:") # TODO: Here is another case of 21 keypoints, make it flexible
        for j in range(min(3, 21)):
            x = float(data_values[j*3])
            y = float(data_values[j*3 + 1])
            z = float(data_values[j*3 + 2])
            print(f"Keypoint {j}: X={x}, Y={y}, Z={z}")
            
            # Create a (3,1) array for this keypoint
            keypoint = np.array([[x], [y], [z]])
            frame_keypoints[j, 0] = keypoint
        
        # Populate remaining keypoints 
        for j in range(3, num_markers): 
            x = float(data_values[j*3])
            y = float(data_values[j*3 + 1])
            z = float(data_values[j*3 + 2])
            keypoint = np.array([[x], [y], [z]])
            frame_keypoints[j, 0] = keypoint
        
        # Add the frame to the structured_points_3d list
        structured_points_3d.append(frame_keypoints)
    
    # Process remaining frames without detailed printing
    for i in range(10, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        values = line.split()
        data_values = values[2:]
        
        # Old Check for 21 markers
        # if len(data_values) != 63: 
        #     print(f"Warning: Frame {i-5} has {len(data_values)} values instead of 63")
        #     continue

        frame_keypoints = np.zeros((num_markers, 1), dtype=object)

        for j in range(num_markers):
            x = float(data_values[j*3])
            y = float(data_values[j*3 + 1])
            z = float(data_values[j*3 + 2])
            keypoint = np.array([[x], [y], [z]])
            frame_keypoints[j, 0] = keypoint
        
        structured_points_3d.append(frame_keypoints)
    
    # Verify the structure of the output
    print("\nStructured points summary:")
    print(f"Total frames processed: {len(structured_points_3d)}")
    
    if structured_points_3d:
        first_frame = structured_points_3d[0]
        print(f"First frame shape: {first_frame.shape}")
        
        first_keypoint = first_frame[0, 0]
        print(f"First keypoint shape: {first_keypoint.shape}")
        print(f"First keypoint data: {first_keypoint}")
        
        last_frame = structured_points_3d[-1]
        last_keypoint = last_frame[-1, 0]
        print(f"Last keypoint in last frame shape: {last_keypoint.shape}")
        print(f"Last keypoint data: {last_keypoint}")
    
    return structured_points_3d

# Example usage:
#structured_points = trc_file_to_structured_points_3d(r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P07_paper\T07_Padless\pose-3d\T07_Padless_0-1953.trc")
#structured_points = trc_file_to_structured_points_3d(r"C:\Users\Miro Hernandez\Documents\GaitScape\S01_STRC111_New\P01_Hernandez_Miro\max_speed\pose-3d\T00_max_speed_0-357.trc")