import os
import sys
import argparse
import time
import random
import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_loader import TextLoader
from model import LSTMClassifier

device	=	torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

	dataset_name	=	"HAR_pose"
	model_name		=	"lstm002"
	model_dir		=	os.path.join(args.output_dir, dataset_name)
	ckpt_file		=	os.path.join(model_dir, model_name + ".ckpt")
	plot_file		=	os.path.join(model_dir, model_name)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
		#os.makedirs(os.path.join(model_dir, 'plots'))
	print("=> Output folder for this run -- {}".format(model_dir))

	#if args.use_gpu:
	#	gpus = [int(i) for i in args.gpus.split(',')]
    #	print("=> active GPUs: {}".format(args.gpus))

	dataset_train	= 	TextLoader(data_dir = args.data_dir, type = "train", transform = transforms.ToTensor)
	dataset_test 	= 	TextLoader(data_dir = args.data_dir, type = "test",  transform = transforms.ToTensor)

	train_loader 	= 	DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
	test_loader 	= 	DataLoader(dataset=dataset_test, 	batch_size=args.batch_size, shuffle=True)

	model 			= 	LSTMClassifier(args.input_size, args.num_layers, args.hidden_size, args.seq_len, args.num_classes)
	optimizer 		= 	optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)#, momentum=0.9)

	#if args.use_gpu:
	#	model 		= torch.nn.DataParallel(model, device_ids=gpus).to(device)

	if args.transfer:
		if os.path.isfile(ckpt_file):
			print("=> loading checkpoint '{}'".format(args.transfer))
			checkpoint	=	torch.load(ckpt_file)
		else:
			print("=> no checkpoint found at '{}'".format(ckpt_file))

	train(model, optimizer, train_loader, test_loader, args.num_epochs, args.batch_size, plot_file)
	evaluate_test_set(model, test_loader, args.batch_size)
	torch.save(model.state_dict(), ckpt_file)


def train(model, optimizer, train_dataset, test_dataset, num_epochs, batch_size, model_dir):

	epoch_plot 		= []
	accuracy_plot 	= []
	criterion 		= nn.CrossEntropyLoss()
	print("=> Training is getting started...")

	for current_epoch in tqdm(range(num_epochs)):
		y_true 		= list()
		y_pred 		= list()
		total_loss 	= 0

		for batch_idx, (data, targets) in enumerate(train_dataset):
			if data.size()[0] < batch_size:
				#print("Insufficient for batch size, skip to next epoch")
				#print("Batch size: ", data.size()[0])
				break
			data 	= data.to(device=device)
			targets = targets.to(device=device)
			# Forward Propagation
			scores 	= model(torch.autograd.Variable(data))
			loss 	= criterion(scores, torch.autograd.Variable(targets))
			# Backward Propagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			pred_idx 	= torch.max(scores, 1)[1]
			y_true 		+= list(targets.int())
			y_pred 		+= list(pred_idx.data.int())
			total_loss 	+= loss

		acc = accuracy_score(y_true, y_pred)
		val_loss, val_acc = evaluate_model(model, test_dataset, criterion, batch_size)
		if current_epoch % 10 is 0:
			print("=> Train loss: {0} - Accuracy: {1} \n=> Validation loss: {2} - Accuracy: {3}".format(
					total_loss.data.float()/len(train_dataset), acc, val_loss, val_acc))
		epoch_plot.append(current_epoch)
		accuracy_plot.append(acc)

	create_plot(epoch_plot,accuracy_plot,model_dir)


def evaluate_model(model, test_dataset, criterion, batch_size):
	y_true = list()
	y_pred = list()
	total_loss = 0

	for batch_idx, (test, targets) in enumerate(test_dataset):
		if test.size()[0] < batch_size:
			break
		test 		= 	test.to(device=device)
		targets 	= 	targets.to(device=device)
		pred 		= 	model(torch.autograd.Variable(test))
		loss		=	criterion(pred, torch.autograd.Variable(targets))
		pred_idx	=	torch.max(pred, 1)[1]
		y_true		+=	list(targets.int())
		y_pred		+=	list(pred_idx.data.int())
		total_loss	+=	loss
	accuracy	=	accuracy_score(y_true, y_pred)
	return total_loss.data.float()/len(test_dataset), accuracy

def evaluate_test_set(model, test, batch_size):
    y_true = list()
    y_pred = list()

    for batch_idx, (test, targets) in enumerate(test):
		if test.size()[0] < batch_size:
			break

		test		=	test.to(device=device)
		targets 	=	targets.to(device=device)
		pred		=	model(torch.autograd.Variable(test))
		pred_idx 	= 	torch.max(pred, 1)[1]
		y_true 		+= 	list(targets.int())
		y_pred 		+= 	list(pred_idx.data.int())

    #print(len(y_true), len(y_pred))
    #print(classification_report(y_true, y_pred))
    #print(confusion_matrix(y_true, y_pred))

def create_plot(epoch_list,acc_list,directory):
	''' Creates graph of loss, training accuracy, and test accuracy '''

	font = {
	    'family' : 'Bitstream Vera Sans',
	    'weight' : 'bold',
	    'size'   : 12
	}
	matplotlib.rc('font', **font)
	plt.figure(figsize=(8, 8))
	plt.axis([0, len(epoch_list), 0, 1])
	plt.plot(epoch_list, acc_list)
	plt.title("Training Accuracy over Epochs")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	#plt.show()

	#''' Save graph '''
	plt.savefig(directory)

# Constants used - more for a reminder
#	input_size 	= 36
#	num_layers 	= 2
#	hidden_size = 34
#	seq_len		= 32
#	num_classes = 6

str2bool = lambda x: (str(x).lower() == 'true')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', 	type=str, default='checkpoints/',
											help='data_directory')
	parser.add_argument('--data_dir', 		type=str, default='data/HAR_pose_activities/',
											help='data_directory')
	parser.add_argument('--num_epochs', 	type=int, default=300,
											help='maximum number of epochs')
	parser.add_argument('--hidden_size',	type=int, default=34,
											help='LSTM hidden dimensions')
	parser.add_argument('--batch_size', 	type=int, default=512,
											help='size for each minibatch')
	parser.add_argument('--input_size', 	type=int, default=36,
											help='x and y dimension for 18 joints')
	parser.add_argument('--num_layers', 	type=int, default=2,
											help='number of hidden layers')
	parser.add_argument('--seq_len', 		type=int, default=32,
											help='number of steps/frames of each action')
	parser.add_argument('--num_classes',	type=int, default=6,
											help='number of classes/type of each action')
	parser.add_argument('--learning_rate',	type=float, default=0.0025,
											help='initial learning rate')
	parser.add_argument('--weight_decay', 	type=float, default=1e-4,
											help='weight_decay rate')

	parser.add_argument('--use_gpu',		type=str2bool, default=True,
											help="flag to use gpu or not.")
	parser.add_argument('--gpus',			type=int, default=0,
											help='gpu ids for use')
	parser.add_argument('--transfer',		type=str2bool, default=False,
											help='resume training from given checkpoint')
	#parser.add_argument('--gpus',			type=int, default=0,
	#										help='gpu ids for use')

	args = parser.parse_args()
	main(args)
