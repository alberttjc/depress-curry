#!/usr/bin/env python
import cv2
import numpy as np
import os
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

FRAME_DIR = '/home/albie/Data/frames/'
OUTPUT_DIR = '/home/albie/Data/00.txt'

if not os.path.isdir(FRAME_DIR):
    os.makedirs(FRAME_DIR)
#if not os.path.isdir(OUTPUT_DIR):
#    os.makedirs(OUTPUT_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(FRAME_DIR):
    files += len(filenames)
    folders += len(dirnames)

print "{:,} files, {:,} folders".format(files, folders)

def humans_to_skels_list(humans, scale_h=1):

    skeletons = []
    NaN = 0
    for human in humans:
        skeleton = [NaN]*(18*2)
        for i, body_part in human.body_parts.items():
            idx = body_part.part_idx
            skeleton[2*idx]=body_part.x
            skeleton[2*idx+1]=body_part.y * scale_h
        skeletons.append(skeleton)
    return skeletons, scale_h

def main():

    w = 432
    h = 368

    e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w,h))

    for folder_idx in range(files):
        folder_idx += 1
        last_file_idx = 1
        #current_frame = 1

        INPUT_DIR = FRAME_DIR + ("%s" % folder_idx) + "/"
        print("Processing current folder: %s", INPUT_DIR)

        for frame_idx in range(30):

            name = 'frame' + str(last_file_idx) + '.jpg'
            current_frame = INPUT_DIR + name

            image = common.read_imgfile(current_frame, None, None)

            if image is None:
                logger.error('Image cannot be read')
                sys.exit(-1)

            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            #image = TfPoseEstimator.draw_humans(image,humans,imgcopy=False)
            #cv2.imshow("display",image)
            #cv2.waitKey()

            skeletons, scale_h = humans_to_skels_list(humans, 1)

            if os.path.exists(OUTPUT_DIR):
                with open(OUTPUT_DIR, "a") as label_file:
                    #print(str(skeletons).strip("[]"))
                    label_file.write(str(skeletons).strip("[]")+"\n")
            else:
                with open(OUTPUT_DIR, "w") as label_file:
                    label_file.write(str(skeletons).strip("[]")+"\n")

            last_file_idx += 1

    print("Done")
    return 0


if __name__ == "__main__":
    main()
