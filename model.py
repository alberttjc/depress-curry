import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cpu")

class LSTMClassifier(nn.Module):

	def __init__(self, input_size, num_layers, hidden_size, seq_len, num_features, dropout):

		super(LSTMClassifier, self).__init__()

		self.input_size 	= 	input_size
		self.num_layers 	= 	num_layers
		self.hidden_size	=	hidden_size
		self.dropout 		=	dropout
		self.lstm_drop		=	self.dropout
		self.seq_len		=	seq_len
		self.num_classes	=	6

		self.lstm 	= 	nn.LSTM(input_size, hidden_size, \
								num_layers, dropout=self.lstm_drop )
		self.fc1	=	nn.Linear(seq_len*num_features*hidden_size, self.hidden1)
		self.fc2	=	nn.Linear(self.hidden1, self.hidden2)
		self.fc3 	=	nn.Linear(self.hidden2,	self.num_classes)
		self.drop	=	nn.Dropout(p=dropout)



		#self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

		#self.hidden2out = nn.Linear(hidden_dim, output_size)
		#self.softmax = nn.LogSoftmax()

		#self.dropout_layer = nn.Dropout(p=0.2)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch, lengths):

		self.hidden = self.init_hidden(batch.size(-1))

		embeds = self.embedding(batch)
		packed_input = pack_padded_sequence(embeds, lengths)
		outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		output = self.softmax(output)

		return output


class RNNClass(nn.Module):

	def __init__(self, input_size, num_layers, hidden_size, seq_len, num_classes):

		super(RNNClass, self).__init__()

		self.hidden_size	=	hidden_size
		self.num_layers 	= 	num_layers
		self.rnn			=	nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.fc				=	nn.Linear(hidden_size*seq_len, num_classes)


	def init_hidden(self, batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, x):
		#self.hidden = self.init_hidden(x.)

		h0 =	torch.zeros(self.num_layers, 512, self.hidden_size)
		out, _ = self.rnn(x, h0)
		output = out.reshape(out.shape[0], -1)

		output = self.fc(output)
		return output


class RNN_LSTM(nn.Module):

	def __init__(self, input_size, num_layers, hidden_size, seq_len, num_classes):

		super(RNN_LSTM, self).__init__()

		self.hidden_size	=	hidden_size
		self.num_layers 	= 	num_layers
		self.lstm			=	nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc				=	nn.Linear(hidden_size*seq_len, num_classes)


	def forward(self, x):
		#self.hidden = self.init_hidden(x.)

		h0		=	torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 		=	torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		out, _ = self.lstm(x, (h0,c0))
		# out: tensor of shape (batch_size, seq_length, hidden_size)
		output = out.reshape(out.shape[0], -1)

		output = self.fc(output)
		return output
