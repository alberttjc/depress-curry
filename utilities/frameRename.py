import cv2
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
IMAGE_DIR = '/home/caris/Data/UTD_MHAD/3D_frames/'

if not os.path.isdir(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(IMAGE_DIR):
    files += len(filenames)
    folders += len(dirnames)
#print "{:,} files, {:,} folders".format(files, folders)

REDUCE_FRAMES = False
"""
    This section is to reduce the number of overall frames
"""
if REDUCE_FRAMES:
    for folder_idx in range(folders):
        ith_image = 1
        folder_idx += 1
        array = []
        array_remove = []
        array_size = 30
        toggle = False

        INPUT_DIR = IMAGE_DIR + ("%s" % folder_idx) + "/"
        _, _, frames = next(os.walk(INPUT_DIR))

        for frame_idx in range(len(frames)):
            array.append(ith_image)
            ith_image += 1

        #if len(array) < array_size:
        #    print(INPUT_DIR)
        #    interrupt = True

        while len(array) > array_size:
            if toggle is True:
                array_remove.append(array[0])
                array = array[1:]
            else:
                array_remove.append(array[-1])
                array = array[:-1]
            toggle = not toggle

        for frame_idx in range(len(frames)):
            frame_idx += 1

            if frame_idx in array_remove:
                IMAGE_IDX = INPUT_DIR + ("%s" % frame_idx) + ".jpg"
                os.remove(IMAGE_IDX)

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
