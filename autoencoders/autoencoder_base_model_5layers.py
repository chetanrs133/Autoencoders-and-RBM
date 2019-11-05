import os
import struct
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import time
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test


path = "C:/Users/Chetan RS/Desktop/chetan/study/8th sem/dl/Assignment_2/task1/dataset1"
class_num=dict()
class_num={"coast":0,"highway":1,"insidecity":2,"street":3,"tallbuilding":4}
files = []
for i in os.listdir(path):
    path_ = path + '/' + i
    cls=class_num[i]
    for j in os.listdir(path_):
        files.append((open(path_ + '/' + j, 'r'),cls))
n = 0
X = []
y=[]
for file in files:
    example_list = list()
    for line in file[0]:
        str_list = line.split()
        num_list = list(map(float, str_list))
        example_list.append(num_list)
    example_arr = np.asarray(example_list)
    X.append(example_arr.flatten())
    y.append(file[1])

X = np.array(X)
X = StandardScaler().fit_transform(X)
y=np.array(y)
y = y[:,np.newaxis]
dataset = np.concatenate((X,y),axis = 1)
np.random.seed(1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
x_train = dataset_train[:,0:828]
Y_train = dataset_train[:,828:]
Y_train = Y_train.squeeze()
#val
dataset_val=np.asarray(dataset_val)
x_val = dataset_val[:,0:828]
Y_val = dataset_val[:,828:]
Y_val = Y_val.squeeze()
#test
dataset_test=np.asarray(dataset_test)
x_test = dataset_test[:,0:828]
Y_test = dataset_test[:,828:]
Y_test = Y_test.squeeze()

trainLabel=Y_train.astype(int)
testLabel=Y_test.astype(int)
valLabel=Y_val.astype(int)
trainFeats=x_train
testFeats=x_test
valFeats=x_val


X_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(Y_val)
X_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(Y_train)
X_test = torch.from_numpy(x_test)
y_test= torch.from_numpy(Y_test)

y_val = y_val.float()
y_train=y_train.float()
y_test=y_test.float()
X_train = X_train.float()
X_test = X_test.float()
X_val = X_val.float()
# =============================================================================
# #define autoencoders
# 
class autoencoder(nn.Module):
    def __init__(self,d,n1,l):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, n1),     #d,n1
            nn.Tanh(),
            nn.Linear(n1, l),    #n1,l
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(l, n1),
            nn.Tanh(),
            nn.Linear(n1, d),
            nn.ReLU())
# 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# 
# 
net = autoencoder(828,500,350)
print(net)
# 

net = net.float()

cr = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr=0.9, momentum=0.9)
#training
iteration = 300
BatchSize = 90
prev_loss = 0
for epoch in range(iteration):
    runningLoss = 0
    for i in range(int(X_train.size()[0]/BatchSize)):
        inputs = torch.index_select(X_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize)
                                  .long()).float()
        inputs = Variable(inputs)
        opt.zero_grad()
        outputs = net(inputs)
        loss = cr(outputs, inputs)
        loss.backward()
        opt.step()
        runningLoss += loss.item()
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iteration,runningLoss/
                                                                (X_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(X_train.size()[0]/BatchSize)
    if (abs(runningLoss - prev_loss)<=0.00005):
        break
    prev_loss = runningLoss
print('Finished Training')


inputs_val = Variable(X_val)
outputs_val = net(inputs_val)
loss_val = cr(outputs_val, inputs_val)
loss_val = loss_val.item()
print('Reconstruction error on validation data : ',loss_val)

inputs_test = Variable(X_test)
outputs_test = net(inputs_test)
loss_test = cr(outputs_test, inputs_test)
loss_test = loss_test.item()
print('Reconstruction error on testing data : ',loss_test)