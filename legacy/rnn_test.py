import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
from librosa.util import frame
import numpy as np

torch.manual_seed(0)
h = h5py.File('data_nocab_norm.h5','r')

X = h['x'][:]
Y = h['y'][:]

x = X[0:100000]/2+1
y = Y[0:100000]/2+1

x = torch.from_numpy(frame(x, 41, 1).T).type(torch.FloatTensor)
y = torch.from_numpy(y[40:]).type(torch.FloatTensor)

hidden_size = 128

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

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, hidden_size), 
            nn.PReLU(), 
            nn.Linear(hidden_size, 1), 
            nn.PReLU()
        )
    def forward(self, x):
        return self.fc(x)

# Batch gradient descent
trainx = Variable(x[0:90000])
trainy = Variable(y[0:90000])
testx = Variable(x[90000:])
testy = Variable(y[90000:])

# Simple Net
#net = Simple_Net()
#criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(net.parameters(), lr = 0.005)
#for i in range(1000):
#    out = net(trainx)
#    loss = criterion(out, trainy)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    print(loss.data[0])

# RNN_Net
# net = RNN_Net().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr = 0.005)
# trainx = trainx.t().unsqueeze(2).cuda()
# for i in range(1000):
#     hidden = Variable(torch.zeros(2, trainx.size(1), hidden_size).cuda())
#     out = net(trainx, hidden)
#     loss = criterion(out, trainy.cuda())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss.data[0])
# torch.save(net, 'model.pth')

# Minibatch gradient descent
net = RNN_Net().cuda()
batch_size = 100
num_epochs = 100
num_batches = int(y.shape[0]/batch_size) + 1
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0005)
for epoch in range(num_epochs):
    avg_loss = 0.0
    for batch_id in range(num_batches):
       batch_x = Variable(x[batch_id*batch_size:(batch_id+1)*batch_size, :].unsqueeze(2).type(torch.FloatTensor).cuda())
       batch_y = Variable(y[batch_id*batch_size:(batch_id+1)*batch_size].unsqueeze(1).type(torch.FloatTensor).cuda())
       hidden = Variable(torch.zeros(1, batch_x.size(0), hidden_size).cuda())
       out = net(batch_x, hidden)
       loss = criterion(out, batch_y)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       avg_loss += loss.data[0]
    print(avg_loss/batch_id)
torch.save(net.state_dict(), 'model.pth')
