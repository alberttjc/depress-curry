#!/usr/bin/env python
import os
import sys
import argparse
import time
import random
import signal
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict, deque

from data_loader import TextLoader
from model import LSTMClassifier

LABELS = [
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "WAVE",
    "CLAP",
    "CLOCKWISE",
    "COUNTER_CLOCKWISE",
]

x_test_data 	=	"x_test.txt"
y_test_data 	=	"y_test.txt"

use_cuda    =   torch.cuda.is_available()
device	    =	torch.device("cuda" if use_cuda else "cpu")

def main(args):
    dataset_name	=	"UTD_MHAD"
    model_name		=	"remodel000"
    model_dir		=	os.path.join(args.model_dir, dataset_name)
    ckpt_file		=	os.path.join(model_dir, model_name + ".ckpt")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
		#os.makedirs(os.path.join(model_dir, 'plots'))
    print("=> Output folder for this run -- {}".format(ckpt_file))

    if args.use_gpu:
        gpus = [int(i) for i in args.gpus.split(',')]
        print("=> active GPUs: {}".format(args.gpus))

    model 			= 	LSTMClassifier(args.input_size, args.num_layers, args.hidden_size, args.seq_len, args.num_classes, use_cuda)
    if use_cuda:
        model 	    = 	model.cuda()
        model.load_state_dict(torch.load(ckpt_file))
    else:
        model.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))

    dataset_test   =   TextLoader(x_path = args.data_dir + x_test_data, y_path = args.data_dir + y_test_data, \
									transform = transforms.ToTensor, n_steps=args.seq_len)

    test_loader     = 	DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True)

    ###

    k = 6
    threshold = 0.7
    SQ = deque(maxlen=10)
    act = deque(['No gesture','No gesture'], maxlen=3)

    count = 0
    total_size = 0

    for batch_idx, (data, targets) in enumerate(test_loader):
        data        =   data.to(device=device)
        targets     =   targets.to(device=device)

        scores      =   model(torch.autograd.Variable(data))

        ###
        scores      =   model(data)
        #scores      =   torch.nn.functional.softmax(scores, dim=1)
        ts, pred    =   scores.detach().cpu().topk(k, 1, True, True)
        top5        =   [LABELS[pred[0][i].item()] for i in range(k)]

        pi          =   [pred[0][i].item() for i in range(k)]
        ps          =   [ts[0][i].item() for i in range(k)]
        top1        =   top5[0] if ps[0] > threshold else LABELS[0]

        #prediction  =   torch.max(scores, 1)
        #print(prediction.item())

        """
        hist = {}

        for i in range(k):
            hist[i] = 0
        for i in range(len(pi)):
            hist[pi[i]] = ps[i]
        SQ.append(list(hist.values()))
        ave_pred = np.array(SQ).mean(axis=0)
        top1 = LABELS[np.argmax(ave_pred)] if max(ave_pred) > threshold else LABELS[0]
        top1 = top1.lower()
        act.append(top1)
        """
        #print(act)
        ###
        #prediction	=	torch.max(scores, 1)[1]

        label_idx   =   targets.int().cpu()

        if LABELS[label_idx.item()] == top1:
            count += 1
        total_size += 1

    print("Total right {0} out of {1}".format(count, total_size))

    """
        for index, (value1, value2) in enumerate(zip(prediction, label_idx)):
            if value1.item() is value2.item():
                count += 1
                #print("True")
            #else:
                #print("False")
            total_size += 1
            #print(index,value1.item(),value2.item())
        print("Total right {0} out of {1}".format(count, total_size))
    """

# Constants used - more for a reminder
#	input_size 	= 36
#	num_layers 	= 2
#	hidden_size = 34
#	seq_len		= 32
#	num_classes = 6

str2bool = lambda x: (str(x).lower() == 'true')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', 	type=str, default='../model/',
											help='data_directory')
	parser.add_argument('--data_dir', 		type=str, default='/home/caris/Data/UTD_MHAD/',
											help='data_directory')
	parser.add_argument('--hidden_size',	type=int, default=70,
											help='LSTM hidden dimensions')
	parser.add_argument('--batch_size', 	type=int, default=1,
											help='size for each minibatch')
	parser.add_argument('--input_size', 	type=int, default=50,
											help='x and y dimension for 18 joints')
	parser.add_argument('--num_layers', 	type=int, default=2,
											help='number of hidden layers')
	parser.add_argument('--seq_len', 		type=int, default=12,
											help='number of steps/frames of each action')
	parser.add_argument('--num_classes',	type=int, default=6,
											help='number of classes/type of each action')

	parser.add_argument('--use_gpu',		type=str2bool, default=False,
											help="flag to use gpu or not.")
	parser.add_argument('--gpus',			type=int, default=0,
											help='gpu ids for use')
	parser.add_argument('--transfer',		type=str2bool, default=False,
											help='resume training from given checkpoint')

	args = parser.parse_args()
	main(args)
