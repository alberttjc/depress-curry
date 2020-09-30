#!/usr/bin/env python
import cv2
import numpy as np
import os
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

LABEL_DIR = '/home/albie/Data/labels/'
OUTPUT_DIR = '/home/albie/Data/labels.txt'

if not os.path.isdir(LABEL_DIR):
    os.makedirs(LABEL_DIR)
#if not os.path.isdir(OUTPUT_DIR):
#    os.makedirs(OUTPUT_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(LABEL_DIR):
    files += len(filenames)
    folders += len(dirnames)

print "{:,} files, {:,} folders".format(files, folders)

def main():

    for folder_idx in range(files):
        folder_idx += 1
        INPUT_DIR = LABEL_DIR + ("%s" % folder_idx) + ".txt"

        print("Processing current folder: %s", INPUT_DIR)

        if open(INPUT_DIR,"r").read() == "None":
            data = 0
        elif open(INPUT_DIR,"r").read() == "Wave":
            data = 1

        if os.path.exists(OUTPUT_DIR):
            with open(OUTPUT_DIR, "a") as label_file:
                label_file.write(str(data).strip("[]")+"\n")
        else:
            with open(OUTPUT_DIR, "w") as label_file:
                label_file.write(str(data).strip("[]")+"\n")


    print("Done")
    return 0



if __name__ == "__main__":
    main()
