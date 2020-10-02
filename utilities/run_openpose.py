# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

IMAGE_DIR = '/home/caris/Data/UTD_MHAD/3D_frames/'
OUTPUT_DIR = '/home/caris/Data/UTD_MHAD/3D_json/'

def openpose_inference(filename, image_path):
    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        datum.name=str(filename)
        imageToProcess = cv2.imread(image_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        """
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        print("Face keypoints: \n" + str(datum.faceKeypoints))
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)
        """

    except Exception as e:
        print(e)
        sys.exit(-1)


def main():

    for folder_idx in range(folders):
        folder_idx      +=  1
        last_file_idx   =   1

        INPUT_DIR       =   IMAGE_DIR + ("%s" % folder_idx) + "/"
        FOLDER_DIR      =   OUTPUT_DIR + ("%s" % (folder_idx)) + "/"

        if not os.path.isdir(FOLDER_DIR):
            os.makedirs(FOLDER_DIR)

        params["write_json"] = FOLDER_DIR

        _, _, frames = next(os.walk(INPUT_DIR))

        for frame_idx in range(len(frames)):
            current_frame = INPUT_DIR + str(last_file_idx) + '.jpg'
            openpose_inference(last_file_idx, current_frame)
            last_file_idx += 1

    print("Done")

if __name__ == "__main__":

    # Initialize constants
    files = folders = 0

    for _, dirnames, filenames in os.walk(IMAGE_DIR):
        files += len(filenames)
        folders += len(dirnames)

    print "{:,} files, {:,} folders".format(files, folders)

    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    params = dict()
    params["model_folder"] = "/home/caris/openpose/models/"
    params["hand"] = False
    params["face"] = False

    main()
