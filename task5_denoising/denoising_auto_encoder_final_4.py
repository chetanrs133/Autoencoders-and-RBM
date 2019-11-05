# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:29:47 2019

@author: jairam
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:43:20 2019

@author: jairam
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:38:12 2019

@author: jairam
"""

#denoising auto_encoders
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test


from pandas import read_csv
filename='C:/Users/Chetan RS/Desktop/chetan/study/8th sem/dl/Assignment_2/stacked_dataset2/10/data.csv'
data=read_csv(filename)
array=data.values
X=array[:,0:784]#means first 8 col
Y=array[:,784]
Y = Y[:,np.newaxis]
dataset = np.concatenate((X,Y),axis = 1)

np.random.seed(1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train

dataset_train=np.asarray(dataset_train)
x_train = dataset_train[:,0:784]
Y_train = dataset_train[:,784:]
Y_train = Y_train.squeeze()
#val
dataset_val=np.asarray(dataset_val)
x_val = dataset_val[:,0:784]
Y_val = dataset_val[:,784:]
Y_val = Y_val.squeeze()
#test
dataset_test=np.asarray(dataset_test)
x_test = dataset_test[:,0:784]
Y_test = dataset_test[:,784:]
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
 #       x = L1Penalty.apply(x, 0.1)        
        x = self.decoder(x)
        return x


#aann1
net1 = autoencoder(784,720,520)
print(net1)

net1 = net1.float()

BatchSize = 1500

iterations = 70
learning_rate = 1e-3
noise_mean = 0.1
noise_std = 0.2
criterion = nn.MSELoss()
# Creating dataloader


optimizer = optim.Adam(net1.parameters(), lr = learning_rate) # Adam optimizer for optimization
prev_loss = 0
for epoch in range(iterations):  # loop over the dataset multiple times
    runningLoss = 0
    
    
    for i in range(int(X_train.size()[0]/BatchSize)):
        
        inputs = torch.index_select(X_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize).long()).float()

        ideal_outputs = Variable(inputs)
        noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
        inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float())

        optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = net1(inputs) # forward 
        loss = criterion(outputs, ideal_outputs) # calculate loss
        loss.backward() #  backpropagate the loss
        optimizer.step()
        runningLoss += loss.item()
    
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,
                                                                        runningLoss/(X_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(X_train.size()[0]/BatchSize)
    if (abs(runningLoss-prev_loss)<=0.0001):
        break
    prev_loss = runningLoss
print('Finished Training with ',epoch,' epochs')

new_classifier1 = nn.Sequential(*list(net1.children())[0])
X_train_var = Variable(X_train)
Z1_train = new_classifier1(X_train_var)


#aann2
net2 = autoencoder(520,220,195)
print(net2)

net2 = net2.float()

BatchSize = 1500

iterations = 70
learning_rate = 1e-3
noise_mean = 0.1
noise_std = 0.2
criterion = nn.MSELoss()
# Creating dataloader


optimizer = optim.Adam(net2.parameters(), lr = learning_rate) # Adam optimizer for optimization
prev_loss = 0
for epoch in range(iterations):  # loop over the dataset multiple times
    runningLoss = 0
    
    
    for i in range(int(Z1_train.size()[0]/BatchSize)):
        
        inputs = torch.index_select(Z1_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize).long()).float()

        ideal_outputs = Variable(inputs)
        noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
        inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float())

        optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = net2(inputs) # forward 
        loss = criterion(outputs, ideal_outputs) # calculate loss
        loss.backward() #  backpropagate the loss
        optimizer.step()
        runningLoss += loss.item()
    
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,
                                                                        runningLoss/(Z1_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(Z1_train.size()[0]/BatchSize)
    if (abs(runningLoss-prev_loss)<=0.0001):
        break
    prev_loss = runningLoss
print('Finished Training with ',epoch,' epochs')

new_classifier2 = nn.Sequential(*list(net2.children())[0])
Z1_train_var = Variable(Z1_train)
Z2_train = new_classifier2(Z1_train_var)
#aann3
net3 = autoencoder(195,150,90)
print(net3)

net3 = net3.float()

BatchSize = 1500

iterations = 70
learning_rate = 1e-3
noise_mean = 0.1
noise_std = 0.2
criterion = nn.MSELoss()
# Creating dataloader


optimizer = optim.Adam(net3.parameters(), lr = learning_rate) # Adam optimizer for optimization
prev_loss = 0
for epoch in range(iterations):  # loop over the dataset multiple times
    runningLoss = 0
    
    
    for i in range(int(Z2_train.size()[0]/BatchSize)):
        
        inputs = torch.index_select(Z2_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize).long()).float()

        ideal_outputs = Variable(inputs)
        noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
        inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float())

        optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = net3(inputs) # forward 
        loss = criterion(outputs, ideal_outputs) # calculate loss
        loss.backward() #  backpropagate the loss
        optimizer.step()
        runningLoss += loss.item()
    
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,
                                                                        runningLoss/(Z2_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(Z2_train.size()[0]/BatchSize)
    if (abs(runningLoss-prev_loss)<=0.0001):
        break
    prev_loss = runningLoss
print('Finished Training with ',epoch,' epochs')

new_classifier3 = nn.Sequential(*list(net3.children())[0])
Z2_train_var = Variable(Z2_train)
Z3_train = new_classifier3(Z2_train_var)

#aann4
net4 = autoencoder(90,60,15)
print(net3)

net4 = net4.float()

BatchSize = 1500

iterations = 70
learning_rate = 1e-3
noise_mean = 0.1
noise_std = 0.2
criterion = nn.MSELoss()
# Creating dataloader


optimizer = optim.Adam(net4.parameters(), lr = learning_rate) # Adam optimizer for optimization
prev_loss = 0
for epoch in range(iterations):  # loop over the dataset multiple times
    runningLoss = 0
    
    
    for i in range(int(Z3_train.size()[0]/BatchSize)):
        
        inputs = torch.index_select(Z3_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize).long()).float()

        ideal_outputs = Variable(inputs)
        noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
        inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float())

        optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = net4(inputs) # forward 
        loss = criterion(outputs, ideal_outputs) # calculate loss
        loss.backward() #  backpropagate the loss
        optimizer.step()
        runningLoss += loss.item()
    
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,
                                                                        runningLoss/(Z3_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(Z3_train.size()[0]/BatchSize)
    if (abs(runningLoss-prev_loss)<=0.0001):
        break
    prev_loss = runningLoss
print('Finished Training with ',epoch,' epochs')

new_classifier4= nn.Sequential(*list(net4.children())[0])




new_classifier5 = nn.Sequential(*list(net4.children())[1])
new_classifier6 = nn.Sequential(*list(net3.children())[1])
new_classifier7 = nn.Sequential(*list(net2.children())[1])
new_classifier8 = nn.Sequential(*list(net1.children())[1])


a = list(new_classifier1.children())
b = list(new_classifier2.children())
c = list(new_classifier3.children())
d = list(new_classifier4.children())
e = list(new_classifier5.children())
f = list(new_classifier6.children())
g = list(new_classifier7.children())
h = list(new_classifier8.children())
i= a + b  + c+d+e+f+g+h

net_joined = nn.Sequential(*i)
net_joined = net_joined.float()

BatchSize = 1500

iterations = 50
learning_rate = 1e-3
noise_mean = 0.1
noise_std = 0.2
criterion = nn.MSELoss()
# Creating dataloader


optimizer = optim.Adam(net_joined.parameters(), lr = learning_rate) # Adam optimizer for optimization
prev_loss = 0
for epoch in range(iterations):  # loop over the dataset multiple times
    runningLoss = 0
    
    
    for i in range(int(X_train.size()[0]/BatchSize)):
        
        inputs = torch.index_select(X_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize).long()).float()

        ideal_outputs = Variable(inputs)
        noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
        inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float())

        optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
        outputs = net_joined(inputs) # forward 
        loss = criterion(outputs, ideal_outputs) # calculate loss
        loss.backward() #  backpropagate the loss
        optimizer.step()
        runningLoss += loss.item()
    
    print('At Iteration : %d / %d  ;  Mean-Squared Error : %f'%(epoch + 1,iterations,
                                                                        runningLoss/(X_train.size()[0]/BatchSize)))
    runningLoss = runningLoss/(X_train.size()[0]/BatchSize)
    if (abs(runningLoss-prev_loss)<=0.0001):
        break
    prev_loss = runningLoss
print('Finished Training with ',epoch,' epochs')

x_val = torch.from_numpy(x_val)
x_val = x_val.float()

inputs_val = Variable(x_val)
outputs_val = net_joined(inputs_val)
loss_val = criterion(outputs_val,inputs_val).item()



x_test = torch.from_numpy(x_test)
x_test = x_test.float()

inputs_test = Variable(x_test)
outputs_test = net_joined(inputs_test)
loss_test = criterion(outputs_test,inputs_test).item()

