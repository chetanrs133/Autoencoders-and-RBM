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
    
    split_1 = int(0.8 * len(df))
    #split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    #dataset_val = df[split_1:split_2]
    dataset_test = df[split_1:]
    return dataset_train,dataset_test

use_gpu=False
#reading data
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
dataset_train,dataset_test=train_validate_test_split(df,0.7,0.1)
#train

dataset_train=np.asarray(dataset_train)
x_train = dataset_train[:,0:784]
Y_train = dataset_train[:,784:]
Y_train = Y_train.squeeze()
#val
# =============================================================================
# dataset_val=np.asarray(dataset_val)
# x_val = dataset_val[:,0:784]
# Y_val = dataset_val[:,784:]
# Y_val = Y_val.squeeze()
# =============================================================================
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
net1 = autoencoder(784,620,420)
print(net1)

if use_gpu:
    net1 = net1.float().cuda()
else:
    net1 = net1.float()
init_weightsE = copy.deepcopy(net1.encoder[0].weight.data)
init_weightsD = copy.deepcopy(net1.decoder[0].weight.data)

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
        #inputs = inputs/255
        if use_gpu:
            ideal_outputs = Variable(inputs).cuda()
            noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float()).cuda()
            # Adding Noise (Noisy Input)
            inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float()).cuda()            
        
        else:
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
net2 = autoencoder(420,120,70)
print(net2)

if use_gpu:
    net2 = net2.float().cuda()
else:
    net2 = net2.float()
init_weightsE = copy.deepcopy(net2.encoder[0].weight.data)
init_weightsD = copy.deepcopy(net2.decoder[0].weight.data)

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
        #inputs = inputs/255
        if use_gpu:
            ideal_outputs = Variable(inputs).cuda()
            noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float()).cuda()
            # Adding Noise (Noisy Input)
            inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float()).cuda()            
        
        else:
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




new_classifier5 = nn.Sequential(*list(net2.children())[1])
new_classifier6 = nn.Sequential(*list(net1.children())[1])


a = list(new_classifier1.children())
b = list(new_classifier2.children())

e = list(new_classifier5.children())
f = list(new_classifier6.children())
g = a + b + e + f

net_joined = nn.Sequential(*g)
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
        #inputs = inputs/255
        if use_gpu:
            ideal_outputs = Variable(inputs).cuda()
            noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float()).cuda()
            # Adding Noise (Noisy Input)
            inputs = Variable(torch.clamp((ideal_outputs + noise).data,0,1).float()).cuda()            
        
        else:
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




x_test = torch.from_numpy(x_test)
x_test = x_test.float()

inputs_test = Variable(x_test)
outputs_test = net_joined(inputs_test)
loss_test = criterion(outputs_test,inputs_test).item()

def imgdisp(img, label, fname):
    imgarr = img.numpy()
    imgarr = np.abs(imgarr)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure()
    plt.title(label)
    plt.imshow(np.transpose(imgarr, (1, 2, 0)))
    plt.savefig(fname+'.jpg')



ideal_outputs = Variable(X_train[6].view(-1,28*28).float())
noise = Variable(ideal_outputs.data.new(ideal_outputs.size()).normal_(noise_mean, noise_std).float())
inputs = ideal_outputs + noise
outImg = net_joined(inputs).data
outImg = outImg.view(-1,28,28)

Img = torch.Tensor(2,1,28,28)
Img[0] = torch.clamp(inputs.data.view(-1,28,28).cpu(),0,1)
Img[1] = outImg


imgdisp(torchvision.utils.make_grid(Img), 'Noisy Input                                              Denoised Output','noisy_vs_reconstr')

Img[0] = torch.clamp(ideal_outputs.data.view(-1,28,28).cpu(),0,1)
Img[1] = outImg

imgdisp(torchvision.utils.make_grid(Img), 'Actual Input                                              Denoised Output','actual_vs_reconstr')