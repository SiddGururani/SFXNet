import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import h5py


h = h5py.File('data_nocab.h5','r')

X = h['x'][:]
Y = h['y'][:]

x = torch.from_numpy(X[0:100000])
# y = torch.from_numpy(Y[0:100000])
# x = torch.Tensor(100000).uniform_(-1,1)
y = torch.sigmoid(x)
trainx = x[0:90000].type(torch.FloatTensor)
trainy = y[0:90000].type(torch.FloatTensor)
testx = x[90000:].type(torch.FloatTensor)
testy = y[90000:].type(torch.FloatTensor)

trainx = Variable(trainx.view(-1,1), requires_grad = False)
trainy = Variable(trainy.view(-1,1), requires_grad = False)
testx = Variable(testx.view(-1,1), requires_grad = False)
testy = Variable(testy.view(-1,1), requires_grad = False)

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 20), 
            nn.ELU(), 
            nn.Linear(20, 1), 
            nn.ELU()
        )
    def forward(self, x):
        return self.fc(x)


model = Simple_Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)
    
for i in range(10000):
    out = model(trainx)
    loss = criterion(out, trainy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    out_v = model(testx)
    loss_v = criterion(out_v, testy)
    print(loss.data[0], '\t', loss_v.data[0])