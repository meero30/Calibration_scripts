# Simple test to compare original AlphaPose output with reconverted output
# to ensure the conversion functions are working correctly


import json, os, glob

orig = sorted(glob.glob("test_data/*.json"))
reconv = sorted(glob.glob("test_data/from_alphapose/*.json"))

for o, r in zip(orig, reconv):
    jo = json.load(open(o))
    jr = json.load(open(r))
    print(os.path.basename(o), jo['people'][0]['pose_keypoints_2d'] == jr['people'][0]['pose_keypoints_2d'])
