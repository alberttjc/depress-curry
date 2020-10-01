# depressed-curry

This repository was restructured from several repos and integrated with Python 2.7 and Tensorflow 1.14 for the purposes of implementing in in ROS Melodic.
The purpose is to utilise LSTM model and OpenPose real-time human pose detection to recognition human actions and gestures using a 2D pose time series dataset.

## Objectives

The aims of this project are:
- To recognise human gestures based on 2D pose time series dataset
- To use real-time human pose skeletal detection by OpenPose by recognise human gestures in real-time

## Requirements

OpenPose
PyTorch


## Acknowledgment

The network used in this experiment is based on that of Guillaume Chevalier, 'LSTMs for Human Activity Recognition, 2016'  https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition, available under the MIT License.

Please also refer to the previous work from stuarteiffert https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
