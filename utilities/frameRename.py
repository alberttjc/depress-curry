import cv2
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
IMAGE_DIR = '/home/caris/Data/UTD_MHAD/frames/'

if not os.path.isdir(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(IMAGE_DIR):
    files += len(filenames)
    folders += len(dirnames)
#print "{:,} files, {:,} folders".format(files, folders)

file_idx = 0
for folder_idx in range(folders):
    folder_idx += 1

    INPUT_DIR = IMAGE_DIR + ("%s" % folder_idx) + "/"
    print("Processing current folder:", INPUT_DIR)
    _, _, frames = next(os.walk(INPUT_DIR))

    for frame_idx in range(len(frames)):
        frame_idx       += 1
        last_file_idx   = frame_idx
        new_frame       = INPUT_DIR + str(frame_idx) + '.jpg'

        flag = True
        while flag:
            current_frame = INPUT_DIR + str(last_file_idx) + '.jpg'
            if os.path.exists(current_frame):
                if current_frame != new_frame:
                    os.rename(current_frame, new_frame)
                flag = not flag
            last_file_idx += 1

print('done')
