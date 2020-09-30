import cv2
import numpy as np
import os

VIDEO_DIR = '/home/albie/Data/videos/'
OUTPUT_DIR = '/home/albie/Data/frames/'

if not os.path.isdir(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(VIDEO_DIR):
    files += len(filenames)
    folders += len(dirnames)
#print "{:,} files, {:,} folders".format(files, folders)


for file_idx in range(files):
    file_idx += 1
    last_file_idx = 1
    current_frame = 1
    cap = cv2.VideoCapture(VIDEO_DIR+("%s.avi" % file_idx))

    FRAME_DIR = OUTPUT_DIR + ("/%s" % file_idx)

    if not os.path.isdir(FRAME_DIR):
        os.makedirs(FRAME_DIR)

    while(True):
        ret, frame = cap.read()

        if current_frame is 1:
            current_frame += 1
        elif current_frame % 5 is 0:
            # Save frame as a jpg file
            name = 'frame' + str(last_file_idx) + '.jpg'
            #print ('Creating: ' + name)
            cv2.imwrite(os.path.join(FRAME_DIR, name), frame)
            last_file_idx += 1

        #keep track of how many images you end up with
        current_frame += 1

        #stop loop when video ends
        if not ret:
            break

#release capture
cap.release()
print('done')
