import os
import sys
import argparse
import time
import random
import utils
import pdb
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import PaddedTensorDataset
from data import TextLoader
from data_loader import TextLoader as TL


from model import LSTMClassifier, RNNClass, RNN_LSTM

device = torch.device("cpu")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/HAR_pose_activities/',
											help='data_directory')
	parser.add_argument('--hidden_size', type=int, default=34,
											help='LSTM hidden dimensions')
	parser.add_argument('--batch_size', type=int, default=512,
											help='size for each minibatch')
	parser.add_argument('--num_epochs', type=int, default=600000,
											help='maximum number of epochs')
	parser.add_argument('--char_dim', type=int, default=128,
											help='character embedding dimensions')
	parser.add_argument('--learning_rate', type=float, default=0.0025,
											help='initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.96,#1e-4,
											help='weight_decay rate')
	parser.add_argument('--seed', type=int, default=123,
											help='seed for random initialisation')
	args = parser.parse_args()
	run(args)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs):
    criterion = nn.NLLLoss(size_average=False)
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in utils.create_dataset(train, x_to_ix, y_to_ix, batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float()/len(train), acc,
                                                                                val_loss, val_acc))
    return model


def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in utils.create_dataset(devset, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float()/len(devset), acc


def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()

    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())

    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def run(args):

	random.seed(args.seed)
	#data_loader = TextLoader(args.data_dir)
	dataset_train = TL(data_dir = args.data_dir, type = "train", transform = transforms.ToTensor)
	dataset_test = TL(data_dir = args.data_dir, type = "test", transform = transforms.ToTensor)

	train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True)

	# Constants used - more for a reminder
	input_size = 36
	num_layers = 2
	hidden_size = 34
	seq_len		= 32
	num_classes = 6

	model = RNN_LSTM(input_size, num_layers, hidden_size, seq_len, num_classes)
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	train(model, optimizer, train_loader, test_loader, args.num_epochs, args.batch_size)
#def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs):

def train(model, optimizer, train_dataset, test_dataset, num_epochs, batch_size):
	criterion = nn.CrossEntropyLoss()
	for epoch in range(num_epochs):
		print('Epoch:', epoch)
		y_true = list()
		y_pred = list()
		total_loss = 0

		for batch_idx, (data, targets) in enumerate(train_dataset):
			if data.size()[0] < batch_size:
				#print("Insufficient for batch size, skip to next epoch")
				#print("Batch size: ", data.size()[0])
				break
			data 	= data.to(device=device)
			targets = targets.to(device=device)
			# Forward Propagation
			scores 	= model(data)
			loss 	= criterion(scores, targets)
			# Backward Propagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			pred_idx = torch.max(scores, 1)[1]
			y_true 	+= list(targets.int())
			y_pred 	+= list(pred_idx.data.int())
			total_loss += loss

		acc = accuracy_score(y_true, y_pred)
		val_loss, val_acc = evaluate_model(model, test_dataset, criterion, batch_size)
		print("Train loss: {0} - acc: {1} \nValidation loss: {2} - acc: {3}".format(
						total_loss.data.float()/len(train_dataset), acc, val_loss, val_acc))

def evaluate_model(model, test_dataset, criterion, batch_size):
	y_true = list()
	y_pred = list()
	total_loss = 0

	for batch_idx, (test, targets) in enumerate(test_dataset):
		if test.size()[0] < batch_size:
			break
		test 		= 	test.to(device=device)
		targets 	= 	targets.to(device=device)
		pred 		= 	model(test)
		loss		=	criterion(pred, targets)
		pred_idx	=	torch.max(pred, 1)[1]
		y_true		+=	list(targets.int())
		y_pred		+=	list(pred_idx.data.int())
		total_loss	+=	loss
	accuracy	=	accuracy_score(y_true, y_pred)
	return total_loss.data.float()/len(test_dataset), accuracy

if __name__ == '__main__':
	main()
