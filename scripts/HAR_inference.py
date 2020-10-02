#!/usr/bin/env python

''' Imports '''
#import os
import sys
#import time
import rospy
import signal
import cv2 as cv
import numpy as np
#import json
#from threading import Lock

import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable as V
#from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
from collections import OrderedDict, deque
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from openpose_ros_msgs.msg import BoundingBox, OpenPoseHuman, OpenPoseHumanList, PointWithProb

from model import LSTMClassifier

LABELS = [
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "WAVE",
    "CLAP",
    "CLOCKWISE",
    "COUNTER_CLOCKWISE",
]

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]



# initialise some variables
qsize = 32  # size of queue to retain for 3D conv input
sqsize = 7 # size of queue for prediction stabilisation
num_classes = k = 6
threshold = 0.75

def cb_pose(data):
    skeletons = []

    for human_idx in range(data.num_humans):
        human = data.human_list[human_idx]
        skeleton    =   [0]*(18*2)

        for body_idx in range(len(human.body_key_points_with_prob)):
            if body_idx < 18:
                skeleton[2*body_idx]=human.body_key_points_with_prob[body_idx].x
                skeleton[2*body_idx+1]=human.body_key_points_with_prob[body_idx].y
            #else:
            #    _nonhuman.append(human.body_key_points_with_prob[body_idx].x)
            #    _nonhuman.append(human.body_key_points_with_prob[body_idx].y)
        skeletons.append(skeleton)

    Q.append(skeletons)

    frames  =   []
    for frame_idx in Q:
        frame_idx   =   torch.tensor(frame_idx)
        frames.append(frame_idx)

    data    =   torch.cat(frames)
    data    =   data.to(device=device)
    data    =   data[None,:]

    if data.size()[1] is qsize:
        #scores      =   model(torch.autograd.Variable(data))
        #prediction	=	torch.max(scores, 1)[1]
        #print(LABELS[prediction.item()])

        output  =   model(torch.autograd.Variable(data))
        ts, pred = output.detach().cpu().topk(k, 1, True, True)
        top5 = [LABELS[pred[0][i].item()] for i in range(k)]

        pi = [pred[0][i].item() for i in range(k)]
        ps = [ts[0][i].item() for i in range(k)]
        top1 = top5[0] if ps[0] > threshold else "DOING SOMETHING ELSE"

        #print(top1)

        hist = {}

        for i in range(6):
            hist[i] = 0

        for i in range(len(pi)):
            hist[pi[i]] = ps[i]

        SQ.append(list(hist.values()))
        ave_pred = np.array(SQ).mean(axis=0)
        top1 = LABELS[np.argmax(ave_pred)] if max(ave_pred) > threshold else LABELS[k+1]
        top1 = top1.lower()
        act.append(top1)

        print(act)


if __name__ == '__main__':
    ''' Initialize node '''
    rospy.loginfo('Initialization+')
    rospy.init_node("depressed_curry", anonymous=True)

    ''' Initialize parameters '''
    #debug           =   rospy.get_param('~debug', 'True')
    ckpt_fn         =   rospy.get_param('~ckpt', "/home/caris/catkin_ws/src/depressed_curry/model/Berkeley_MHAD/lstm005.ckpt")#"/home/caris/catkin_ws/src/depressed_curry/model/lstm005.ckpt")
    image_topic     =   rospy.get_param('~camera', "/kinect2/qhd/image_color_rect")

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided')
        sys.exit(-1)

    ''' Initialize Constant '''
    Q   = deque(maxlen=qsize)
    SQ  = deque(maxlen=sqsize)
    act = deque(['No gesture','No gesture'], maxlen=3)

    cv_bridge   =   CvBridge()
    #sub_image   =   rospy.Subscriber(image_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    sub_pose    =   rospy.Subscriber("/openpose_ros/human_list", OpenPoseHumanList, cb_pose, queue_size=1)

    ''' Initialize inference layer '''
    use_cuda    =   torch.cuda.is_available()
    device      =   torch.device('cuda' if use_cuda else 'cpu')
    #model       =   LSTMClassifier(50,2,70,12,6,use_cuda)
    model       =   LSTMClassifier(36,2,34,32,6,use_cuda)
    model       =   model.cuda() if use_cuda else model

    ''' Loading checkpoint file '''
    model.load_state_dict(torch.load(ckpt_fn))
    model.eval()

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
