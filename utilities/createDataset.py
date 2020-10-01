import json
from pprint import pprint
import glob, os
import numpy as np

data_file = "data.txt"
label_file = "labels.txt"

test_file_X = "x_test.txt"
test_file_Y = "y_test.txt"

train_file_X = "x_train.txt"
train_file_Y = "y_train.txt"

data_dir = "/home/caris/Data/UTD_MHAD/"

num_steps = 12
dataset_split = 0.8
split = True

#check if a test/train split is wanted
if split:
	print("Splitting training and testing dataset")

	data_file	=	open(data_dir + data_file,'r')
	label_file 	= 	open(data_dir + label_file,'r')

	x_data		=	data_file.readlines()
	data_file.close()

	y_data 		= 	label_file.readlines()
	label_file.close()

	msk 		= 	np.random.rand(len(y_data)) < dataset_split

	for i in range(len(x_data)):#len(x_data)
		num = int(i / num_steps)

		if num is len(y_data):
			print("Done")
			break

		if msk[num] == True:
			X_train = open(data_dir + train_file_X,'a')
			X_train.write(x_data[i])
			X_train.close()
		else:
			X_test = open(data_dir + test_file_X,'a')
			X_test.write(x_data[i])
			X_test.close()

	for i in range(len(y_data)):
		num = i % num_steps

		if num > len(y_data):
			print("Done")
			break

		if msk[i] == True:
			Y_train = open(data_dir + train_file_Y,'a')
			Y_train.write(y_data[i])
			Y_train.close()
		else:
			Y_test = open(data_dir + test_file_Y,'a')
			Y_test.write(y_data[i])
			Y_test.close()
	#print msk
