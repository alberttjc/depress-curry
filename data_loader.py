import os
import sys
import math
import random
import argparse
import operator
import pdb
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
from collections import Counter
from torch.autograd import Variable


class TextLoader(Dataset):
    def __init__(self, data_dir, type="train", transform=None):

        self.n_steps        =   32
        self.transform      =   transform
        if type is "train":
            self.x_data     =   self.load_x(data_dir + "X_train.txt",  self.n_steps)
            self.y_data     =   self.load_y(data_dir + "Y_train.txt",  self.n_steps)
        elif type is "test":
            self.x_data     =   self.load_x(data_dir + "X_test.txt",   self.n_steps)
            self.y_data     =   self.load_y(data_dir + "Y_test.txt",   self.n_steps)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        y_data = torch.tensor(int(self.y_data[index,0]))
        x_data = self.x_data[index,:,:]

        #if self.transform:
        #    x_data = self.transform(x_data)

        # DEBUG USAGE:
        # y_data is the label of each x_data, dimension is (sample_size,1)
        # x_data is the training data with the deimnsion of (sample_size, 32, 36)
        # 32 is the time steps for each action (i.e. 32 frames)
        # 36 is the number of x-position and y-position from openpose
        # Desired dimension of x_data is (batch_size, 32, 36) and y_data is (batch_size,1)
        #print(y_data) # output: tensor(0) or tensor(1) --
        #print(x_data) # output: dimension (32, 36)

        return (x_data, y_data)

    def load_x(self, x_path, n_steps):
        file = open(x_path, 'r')
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X_) / n_steps)

        X_ = np.array(np.split(X_,blocks))

        return X_

    def load_y(self, y_path, n_steps):
        file = open(y_path, 'r')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        # for 0-based indexing
        return y_ - 1
