import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNNNet(nn.Module):
	"""
	RNNNet is the class for a RNN-Linear regressor model used for effect modeling.

	Input Parameters:
		rnn_type:			String, The type of RNN for the model. 
							'rnn', 'gru', and 'lstm' are acceptable inputs.
		num_features:		Integer, The dimensionality of the input.
		num_layers:			Integer, Number of RNN layers.
		num_hidden:			Integer, Number of hidden units per layer
		batch_first:		Boolean, Batch is the first dimension of an input tensor
		bidirectional:		Boolean, RNN units are bidirectional
		step_for_linear:	Integer, Time step from last RNN layer to use as input
							to Linear layer.
	"""
	def __init__(self, rnn_type = 'gru', num_features = 1, num_layers = 1, num_hidden = 64, 
					batch_first = True, bidirectional = False, step_for_linear = -1):
		
		super(RNNNet, self).__init__()
		if rnn_type.lower() == 'rnn':
			self.rnn = nn.RNN(num_features, num_layers = num_layers, hidden_size = num_hidden,
								batch_first = batch_first, bidirectional = bidirectional)
		
		elif rnn_type.lower() == 'gru':
			self.rnn = nn.GRU(num_features, num_layers = num_layers, hidden_size = num_hidden,
								batch_first = batch_first, bidirectional = bidirectional)
		
		elif rnn_type.lower() == 'lstm':
			self.rnn = nn.LSTM(num_features, num_layers = num_layers, hidden_size = num_hidden,
								batch_first = batch_first, bidirectional = bidirectional)
		else:
			raise('Incorrect RNN Type')
		self.fully_connected = nn.Linear(num_hidden, 1)
		self.batch_first = batch_first
		self.final_activation = nn.PReLU()
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.num_directions = 2 if bidirectional else 1
		self.step_for_linear = step_for_linear

	def init_hidden(self, batch_size):
		return Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden).cuda())

	def forward(self, x, hidden):
		out, _ = self.rnn(x, hidden)
		try:
			if self.batch_first:
				out = out[:, self.step_for_linear, :]
			else:
				out = out[self.step_for_linear, :, :]
		except IndexError:
			raise('Incorrect time step chosen for Linear after RNN')
		out = self.fully_connected(out)
		out = self.final_activation(out)
		return out