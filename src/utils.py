import os
import numpy as np
from librosa.util import frame
import errno

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_chunks(dataset, effect, num_seconds, sr, context):
	x = dataset['clean'][:]
	y = dataset[effect][:]
	# Frame into 1 second long segments first
	x = frame(x, sr, sr).T
	y = frame(y, sr, sr).T
	# Sample num_seconds number of segments
	ind = np.random.choice(np.arange(x.shape[0]), num_seconds, replace=False)
	x = x[ind]
	y = y[ind]
	# Segment further into context frames
	x = np.apply_along_axis(frame, 1, x, frame_length=context, hop_length=1)
	x = np.transpose(x, (0,2,1))
	x = x.reshape(-1, context, 1)
	y = y[:,context - 1:]
	y = y.reshape(-1, 1)
	return {'x': x, 'y': y}

def get_run_name(effect, bidirectional, rnn_type, rnn_layers, rnn_size, lr, context):
	b = 'bi' if bidirectional else ''
	return 'SFXNet_'+effect+'_'+b+rnn_type+'_'+str(rnn_layers)+'_'+str(rnn_size)+'lr'+str(lr)+str(context)