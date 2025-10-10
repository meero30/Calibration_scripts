
'''
    ##################################################
    ## Convert trc X-up files to Y-up files         ##
    ##################################################
    
    Convert trc files with X-up system coordinates to Y-up files.
    
    Usage: 
    from Pose2Sim.Utilities import trc_Xup_to_Yup; trc_Xup_to_Yup.trc_Xup_to_Yup_func(r'<input_trc_file>', r'<output_trc_file>')
    trc_Xup_to_Yup -i input_trc_file
    trc_Xup_to_Yup -i input_trc_file -o output_trc_file
'''
## INIT
import pandas as pd
import numpy as np
import argparse


## FUNCTIONS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='trc Xup input file')
    parser.add_argument('-o', '--output', required=False, help='trc Yup output file')
    args = vars(parser.parse_args())
    
    trc_Xup_to_Yup_func(args)

def trc_Xup_to_Yup_func(*args):
    '''
    Turns trc files with X-up system coordinates into Y-up files.
    Usage: 
    import trc_Xup_to_Yup; trc_Xup_to_Yup.trc_Xup_to_Yup_func(r'<input_trc_file>', r'<output_trc_file>')
    trcXup_to_Yup -i input_trc_file
    trcXup_to_Yup -i input_trc_file -o output_trc_file
    '''
    try:
        trc_path = args[0]['input'] # invoked with argparse
        if args[0]['output'] == None:
            trc_yup_path = trc_path.replace('.trc', '_Yup.trc')
        else:
            trc_yup_path = args[0]['output']
    except:
        trc_path = args[0] # invoked as a function
        trc_yup_path = trc_path.replace('.trc', '_Yup.trc')
    
    # header
    with open(trc_path, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]
    
    # data
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4)
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
    
    # Create a new dataframe with the same structure
    Q_Yup = pd.DataFrame(index=Q_coord.index)
    
    # Process each marker (every 3 columns)
    for i in range(int(len(Q_coord.columns)/3)):
        marker_name = Q_coord.columns[i*3].split('.')[0]
        
        # Original coordinates
        x = Q_coord.iloc[:, i*3]    # Original X
        y = Q_coord.iloc[:, i*3+1]  # Original Y
        z = Q_coord.iloc[:, i*3+2]  # Original Z
        
        # Create new columns with the transformation:
        # New X = Original Y
        # New Y = Original X
        # New Z = Original Z
        # Invert Y axis to fix upside-down issue
        Q_Yup[f"{marker_name}.X"] = y
        Q_Yup[f"{marker_name}.Y"] = -x  # Negate X to fix upside-down issue
        Q_Yup[f"{marker_name}.Z"] = z
    
    # write file
    with open(trc_yup_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_Yup.insert(0, 'Frame#', frames_col)
        Q_Yup.insert(1, 'Time', time_col)
        Q_Yup.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')
    
    print(f"trc file converted from X-up to Y-up: {trc_yup_path}")
    
if __name__ == '__main__':
    main()