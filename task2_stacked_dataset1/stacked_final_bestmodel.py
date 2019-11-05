# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:16:40 2019

@author: jairam
"""

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
from sklearn.metrics import confusion_matrix

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.8 * len(df))
   # split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
   # dataset_val = df[split_1:split_2]
    dataset_test = df[split_1:]
    return dataset_train,dataset_test



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
#y=y.astype(np.int_)
y = y[:,np.newaxis]
dataset = np.concatenate((X,y),axis = 1)
np.random.seed(1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
x_train = dataset_train[:,0:828]
Y_train = dataset_train[:,828:]
Y_train = Y_train.squeeze()
#val
# =============================================================================
# dataset_val=np.asarray(dataset_val)
# x_val = dataset_val[:,0:828]
# Y_val = dataset_val[:,828:]
# Y_val = Y_val.squeeze()
# =============================================================================
#test
dataset_test=np.asarray(dataset_test)
x_test = dataset_test[:,0:828]
Y_test = dataset_test[:,828:]
Y_test = Y_test.squeeze()

trainLabel=Y_train
testLabel=Y_test
trainFeats=x_train
testFeats=x_test
X_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(Y_train)
X_test = torch.from_numpy(x_test)
y_test= torch.from_numpy(Y_test)
# =============================================================================
# =============================================================================
X_train=X_train.float()
X_test=X_test.float()
# =============================================================================
# =============================================================================
y_train=y_train.long()
y_test=y_test.long()
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
        
#aann1
net1 = autoencoder(828,650,450)
print(net1)
# 
net1 = net1.float()

cr = nn.MSELoss()
opt = optim.SGD(net1.parameters(), lr=0.9, momentum=0.9)
#training
iteration = 250
B_Size = 90
for epoch in range(iteration):
    runningLoss = 0
    for i in range(int(X_train.size()[0]/B_Size)):
        inputs = torch.index_select(X_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()

        inputs = Variable(inputs)
        opt.zero_grad()
        outputs = net1(inputs)
        loss = cr(outputs, inputs)
        loss.backward()
        opt.step()
        runningLoss += loss.item()
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iteration,runningLoss/
                                                                (X_train.size()[0]/B_Size)))
print('Finished Training')

new_classifier1 = nn.Sequential(*list(net1.children())[0])
X_train_var = Variable(X_train)
Z1_train = new_classifier1(X_train_var)

#aann2
net2 = autoencoder(450,350,250)
print(net2)
# 
net2 = net2.float()

cr = nn.MSELoss()
opt = optim.SGD(net2.parameters(), lr=0.5, momentum=0.9)
#training
iteration = 200
B_Size = 90
for epoch in range(iteration):
    runningLoss = 0
    for i in range(int(Z1_train.size()[0]/B_Size)):
        inputs = torch.index_select(Z1_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()

        inputs = Variable(inputs)
        opt.zero_grad()
        outputs = net2(inputs)
        loss = cr(outputs, inputs)
        loss.backward()
        opt.step()
        runningLoss += loss.item()
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iteration,runningLoss/
                                                                (Z1_train.size()[0]/B_Size)))
print('Finished Training')

new_classifier2 = nn.Sequential(*list(net2.children())[0])
Z1_train_var = Variable(Z1_train)
Z2_train = new_classifier2(Z1_train_var)

#aann3
net3 = autoencoder(250,80,20)
print(net3)
# 
net3 = net3.float()

cr = nn.MSELoss()
opt = optim.SGD(net3.parameters(), lr=0.5, momentum=0.9)
#training
iteration = 150
B_Size = 90
for epoch in range(iteration):
    runningLoss = 0
    for i in range(int(Z2_train.size()[0]/B_Size)):
        inputs = torch.index_select(Z2_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()

        inputs = Variable(inputs)
        opt.zero_grad()
        outputs = net3(inputs)
        loss = cr(outputs, inputs)
        loss.backward()
        opt.step()
        runningLoss += loss.item()
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iteration,runningLoss/
                                                                (Z2_train.size()[0]/B_Size)))
print('Finished Training')

new_classifier3 = nn.Sequential(*list(net3.children())[0])
Z2_train_var = Variable(Z2_train)
Z3_train = new_classifier3(Z2_train_var)

a = list(new_classifier1.children())
b = list(new_classifier2.children())
c = list(new_classifier3.children())
d = a + b + c
net_joined = nn.Sequential(*d)
net_joined.add_module('classifier', nn.Sequential(nn.Linear(20, 5),nn.LogSoftmax()))
print(net_joined)

a_ran = net_joined[0].weight.data

cr = nn.NLLLoss()
opt = optim.SGD(net_joined.parameters(), lr=0.07, momentum=0.7)

iteration = 100
B_Size = 90
prev_loss = 0
n_epochs = 1

for epoch in range(iteration):
    runningLoss = 0
    for i in range(int(X_train.size()[0]/B_Size)):
        inputs = torch.index_select(X_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()
        labels = torch.index_select(y_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).long()

        inputs, labels = Variable(inputs), Variable(labels)
        opt.zero_grad()
        outputs = net_joined(inputs)
        loss = cr(outputs, labels)
        loss.backward()
        opt.step()
        runningLoss += loss.item()
    inputs = X_test.float()
    inputs_train = X_train.float()

    inputs = Variable(inputs)
    outputs = net_joined(inputs)
    _, predicted = torch.max(outputs.data, 1)
    inputs_train = Variable(inputs_train)
    outputs_train = net_joined(inputs_train)
    __, predicted_train = torch.max(outputs_train.data, 1)
    correct = 0
    total = 0
    total += y_test.size(0)
    correct += (predicted == y_test).sum()
    
    correct_train = 0
    total_train = 0
    total_train += y_train.size(0)
    correct_train += (predicted_train == y_train).sum()
    print('At Iteration: %d / %d  ;  Training Loss: %f ; Testing Acc: %f ; Training accuracy: %f '%(epoch + 1,iteration,runningLoss/
                                                                            (X_train.size()[0]/
                                                                             B_Size),(100 * correct/ float(total)),(100 * correct_train/ float(total_train))))
    print(net_joined[4].weight.data[0][0])
    runningLoss = runningLoss/(X_train.size()[0]/B_Size)
    if (abs(runningLoss - prev_loss) <= 0.000016):
        break
        #break
    #print(abs(runningLoss - prev_loss))
    prev_loss = runningLoss
    n_epochs = n_epochs + 1
print('Finished Training')

print('Training confusion matrix:',confusion_matrix(y_train, predicted_train))
print('Training confusion matrix:',confusion_matrix(y_test, predicted))

print('No of epochs:',n_epochs)

x_test = torch.from_numpy(x_test)
x_test = x_test.float()
Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.long()
labels_test = Variable(Y_test)
inputs_test = Variable(x_test)
outputs_test = net_joined(inputs_test)
loss_test = cr(outputs_test,labels_test).item()
_____, predicted_test = torch.max(outputs_test.data, 1)
predicted_test = predicted_test.cpu()
total_test = Y_test.size(0)
correct_test = (predicted_test == Y_test).sum()
acc_test = 100 * correct_test/ float(total_test)