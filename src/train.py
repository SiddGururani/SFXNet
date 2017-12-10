import gc
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from model import RNNNet as SFXNet
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from utils import *

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='SFXNet Trainer')

parser.add_argument('--train_data', default='', metavar='DIR', help='path to training hdf5 file')

parser.add_argument('--test_data', default='', metavar='DIR', help='path to test hdf5 file')

parser.add_argument('--val_data', default='', metavar='DIR', help='path to validation hdf5 file')

parser.add_argument('--effect', default='', metavar='DIR', help='effect to model')

parser.add_argument('--num_seconds_train', default=1000, type=int, 
						help='Number of seconds to use for training')
parser.add_argument('--num_seconds_val', default=10, type=int, 
						help='Number of seconds to use for validation')
parser.add_argument('--num_seconds_test', default=10, type=int, 
						help='Number of seconds to use for testing')
parser.add_argument('--sample_rate', default=44100, type=int, 
						help='Sample rate of audio in hdf5 files')
parser.add_argument('--context', default=32, type=int, help='Number of samples as context')

parser.add_argument('--batch_size', default=100, type=int, help='Batch size')

parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')

parser.add_argument('--rnn_type', default='gru', help='Type of RNN unit for SFXNet')

parser.add_argument('--rnn_layers', default=1, type=int, help='Number of RNN layers')

parser.add_argument('--rnn_size', default=64, type=int, 
						help='Number of RNN hidden units per layer')
parser.add_argument('--bidirectional', default=False, help='Bidirectional RNN or not')

parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate for optimizer')

parser.add_argument('--save_path', default='', metavar='DIR', help='path to save folder')

def main():
	args = parser.parse_args()
	save_path = args.save_path
	batch_size = args.batch_size
	epochs = args.epochs

	make_dir(save_path)
	run_name = get_run_name(args.effect, args.bidirectional, args.rnn_type, args.rnn_layers, args.rnn_size, args.lr, args.context)
	run_file = os.path.join('./runs', run_name)
	if os.path.isfile(run_file):
		os.remove(run_file)
	configure(os.path.join('./runs', run_name) , flush_secs = 2)

	train_data = h5py.File(args.train_data, 'r')
	test_data = h5py.File(args.test_data, 'r')
	val_data = h5py.File(args.val_data, 'r')

	train_data = get_chunks(train_data, args.effect, args.num_seconds_train, 
								args.sample_rate, args.context)
	test_data = get_chunks(test_data, args.effect, args.num_seconds_test, 
								args.sample_rate, args.context)
	val_data = get_chunks(val_data, args.effect, args.num_seconds_val, 
								args.sample_rate, args.context)

	model = SFXNet(rnn_type=args.rnn_type, num_layers=args.rnn_layers, 
					num_hidden=args.rnn_size, bidirectional=args.bidirectional).cuda()

	model = train_and_validate(model, train_data, val_data, args.lr, save_path, run_name, batch_size, epochs)

	#Load best validation loss model
	model.load_state_dict(torch.load(os.path.join(save_path, run_name+'.pth')))
	test_loss = test(model, test_data, batch_size)

def train_and_validate(model, train_data, val_data, lr, save_path, run_name, batch_size, epochs):
	num_batches = int(train_data['y'].shape[0]/batch_size)
	criterion = nn.MSELoss()
	optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	log_step = 0
	best_val_loss = 1
	for epoch in range(epochs):
		print('Training epoch: {}'.format(epoch))
		model.train()
		total_loss = 0.0
		for batch_id in range(num_batches):
			batch_x = Variable(torch.from_numpy(train_data['x'][batch_id*batch_size:(batch_id+1)*batch_size, :, :])).cuda()
			batch_y = Variable(torch.from_numpy(train_data['y'][batch_id*batch_size:(batch_id+1)*batch_size, :])).cuda()
			hidden = model.init_hidden(batch_x.size(0))
			out = model(batch_x, hidden)
			loss = criterion(out, batch_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.data[0]
			if batch_id % 1000 == 0 and batch_id > 0:
				log_value('Train Loss (MSE)', total_loss/batch_id, log_step)
				log_step += 1
			print(epoch+1, batch_id+1, num_batches, loss.data[0])
		
		# Validate Model
		print('Validating epoch: {}'.format(epoch))
		validation_loss = evaluate(model, val_data, batch_size)
		log_value('Validation Loss (MSE)', validation_loss, epoch)
		if validation_loss < best_val_loss:
			best_val_loss = validation_loss
			torch.save(model.state_dict(), os.path.join(save_path, run_name+'.pth'))


def evaluate(model, data, batch_size):
	total_loss = 0.0
	criterion = nn.MSELoss()
	num_batches = int(data['y'].shape[0]/batch_size) + 1
	for batch_id in range(num_batches):
		batch_x = Variable(torch.from_numpy(data['x'][batch_id*batch_size:(batch_id+1)*batch_size, :, :]).cuda())
		batch_y = Variable(torch.from_numpy(data['y'][batch_id*batch_size:(batch_id+1)*batch_size, :]).cuda())
		hidden = model.init_hidden(batch_x.size(0))
		out = model(batch_x, hidden)
		loss = criterion(out, batch_y)
		total_loss += loss.data[0]
	loss = total_loss/num_batches
	return loss

if __name__ == '__main__':
	main()