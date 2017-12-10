import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import librosa
import numpy as np

torch.manual_seed(0)

audio_file = '/Code/AClassicEducation_NightOwl_RAW_05_01.wav'

hidden_size = 20
class RNN_Net(nn.Module):
    def __init__(self):
        super(RNN_Net, self).__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first = True)
        self.act = nn.PReLU()
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = output[:,-1,:]
        output = self.fc(output)
        output = self.act(output)
        return output

x, sr = librosa.load(audio_file, mono=True, sr = None)
y = torch.zeros_like(torch.from_numpy(x))
x = x/2 + 1
x = torch.from_numpy(librosa.util.frame(x, 41, 1).T).type(torch.FloatTensor)

net = RNN_Net().cuda()
net.load_state_dict(torch.load('model.pth'))

batch_size = 10000
num_batches = int(x.size(0)/batch_size) + 1
for batch_id in range(num_batches):
   batch_x = Variable(x[batch_id*batch_size:(batch_id+1)*batch_size, :].unsqueeze(2).type(torch.FloatTensor).cuda())
   hidden = Variable(torch.zeros(1, batch_x.size(0), hidden_size).cuda())
   out = net(batch_x, hidden)
   print(out.size())
   y[batch_id*batch_size:batch_id*batch_size+out.size(0)] = out.data[:,0]


librosa.output.write_wav('rnn_dist.wav', y.numpy(), sr)