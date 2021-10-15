import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(108,60)
        #self.fc2 = nn.Linear(60,30)
        self.fc2 = nn.Linear(60,2)
    
    def forward(self,x):
        x = self.fc1(x)
        x = F.tanh(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc2(x)
        return x

    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        if pred[0]>pred[1]:
            ans.append(0)
        else:
            ans.append(1)
        return torch.tensor(ans)

def train(data,y):
    X = torch.Tensor(data).type(torch.FloatTensor)
    X_ = X.cuda()
    y = torch.Tensor(y).type(torch.LongTensor)
    y_ = y.cuda()
    model = Net().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    epochs = 5000
    losses = []

    for i in range(epochs):
        if i%1000==0:
            print('doing iter_'+str(i)+'...')
        y_pred = model(X_)
        loss = criterion(y_pred,y_)
        #loss = F.nll_loss(y_pred,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def predict(model,x):
    x = torch.Tensor(x).type(torch.FloatTensor)
    x = x.cuda()
    ans = model.predict(x)
    return ans.numpy()