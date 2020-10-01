import cv2
import numpy as np
import os

LABEL_DIR = "/home/caris/Data/UTD_MHAD/labels.txt"

"""
        1.  right arm swipe to the left         (swipt_left)
        2.  right arm swipe to the right        (swipt_right)
        3.  right hand wave                     (wave)
        4.  two hand front clap                 (clap)
        5.  right arm throw                     (throw)
        6.  cross arms in the chest             (arm_cross)
        7.  basketball shooting                 (basketball_shoot)
        8.  draw x                              (draw_x)
        9.  draw circle  (clockwise)            (draw_circle_CW)
        10. draw circle  (counter clockwise)    (draw_circle_CCW)
        11. draw triangle                       (draw_triangle)
        12. bowling (right hand)                (bowling)
        13. front boxing                        (boxing)
        14. baseball swing from right           (baseball_swing)
        15. tennis forehand swing               (tennis_swing)
        16. arm curl (two arms)                 (arm_curl)
        17. tennis serve                        (tennis_serve)
        18. two hand push                       (push)
        19. knock on door                       (knock)
        20. hand catch                          (catch)
        21. pick up and throw                   (pickup_throw)
"""
# action is the index array that you want to use for the action associated above

action = [1,2,3,4,9,10]

"""
    file_idx        :   referes to the number of files and folders
    last_file_idx   :   number of frames extracted from a single video files (output)
    current_frame   :   number of frames extracted from a single video files (process)

"""
# Change this if you change action
file_idx = 1
for action_idx in range(len(action)):
    subject_idx = 1
    rep_idx = 1

    while subject_idx < 9:
        """
            #Function used to rename
        """

        if file_idx < (1*32+1):   label_choice = 1
        elif file_idx < (2*32+1):   label_choice = 2
        elif file_idx < (3*32+1):   label_choice = 3
        elif file_idx < (4*32+1):   label_choice = 4
        elif file_idx < (5*32+1):   label_choice = 5
        elif file_idx < (6*32+1):   label_choice = 6
        elif file_idx < (7*32+1):   label_choice = 7

        if os.path.exists(LABEL_DIR):
            with open(LABEL_DIR, "a") as f:
                f.write(str(label_choice)+"\n")
        else:
            with open(LABEL_DIR, 'w') as f:
                f.write(str(label_choice)+"\n")

        file_idx += 1
        if rep_idx is 4:
            subject_idx += 1
            rep_idx = 0
        if rep_idx < 5:
            rep_idx += 1
