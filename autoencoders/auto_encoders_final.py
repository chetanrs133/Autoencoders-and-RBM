# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:34:58 2019

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

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test



use_gpu = torch.cuda.is_available()
use_gpu=False
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
net = autoencoder(828,600,450)
print(net)
# 

net = net.float()
#         
init_weights = copy.deepcopy(net.encoder[0].weight.data)
#optimization technique
cr = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
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
# 
#
net_compressed = nn.Sequential(*list(net.children())[0])
net_compressed = net_compressed.float()
X_train_var = Variable(X_train)
X_test_var = Variable(X_test)
X_val_var = Variable(X_val)

z_train = net_compressed(X_train_var)
z_test = net_compressed(X_test_var)
z_val = net_compressed(X_val_var)

z_train = z_train.detach()
z_test = z_test.detach()
z_val = z_val.detach()
z_train = z_train.numpy()
z_test = z_test.numpy()
z_val = z_val.numpy()

from torch import nn
from torch.utils.data import TensorDataset,DataLoader

class mlp(nn.Module):
    def __init__(self,i,h1,h2): 
        super(mlp, self).__init__()
        self.N1 = nn.Linear(i,h1)        
        self.N2 = nn.Linear(h1,h2) 
        self.N3 = nn.Linear(h2,5)
    def forward(self,x): 
        x = self.N1(x)  
        x = F.relu(x)
        x = self.N2(x)          
        x = F.relu(x) 
        x = self.N3(x)
        x = F.softmax(x)
        return x

trainLabel2 = np.zeros((1103,5))
testLabel2 = np.zeros((316,5))
valLabel2 = np.zeros((157,5))
for i in range(trainLabel.shape[0]):
    trainLabel2[i,trainLabel[i]] = 1
for j in range(testLabel.shape[0]):
    testLabel2[j,testLabel[j]] = 1
for k in range(valLabel.shape[0]):
    valLabel2[k,valLabel[k]] = 1

data_train = np.concatenate((X_train,trainLabel2), axis = 1)
data_test = np.concatenate((X_test,testLabel2), axis = 1)
data_test = np.concatenate((X_val,valLabel2), axis = 1)
    

Z_train = torch.from_numpy(z_train)
y_train_onehot = torch.from_numpy(trainLabel2) 


# Definining the training routine
def model_train(model,cr,n_epochs,l_rate):
           
        prev_loss = 0
        
        for epoch in range(n_epochs):
            
            print('Epoch {}/{}'.format(epoch, n_epochs - 1))

            running_loss = 0.0           
            
            batch = 0
            BatchSize = 90
            
            for i in range(int(data_train.shape[0]/BatchSize)):
                inputs = torch.index_select(Z_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize)
                                          .long()).float()
                labels = torch.index_select(y_train_onehot,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize)
                                          .long()).float()

               
                inputs, labels = Variable(inputs), Variable(labels)   
                
                model.zero_grad() 
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                
                loss = cr(outputs, labels)
                
                running_loss += loss.item()
                
                if batch == 0:
                    total_Loss = loss
                    total_Preds = preds
                    batch += 1                    
                else:
                    total_Loss += loss
                    total_Preds = torch.cat((total_Preds,preds),0)  
                    batch += 1
            
          
            total_Loss = total_Loss/batch
            total_Loss.backward()
            
            # Updating the model parameters
            for f in model.parameters():
                f.data.sub_(f.grad.data * l_rate)                
           
            epoch_loss = running_loss/int(data_train.shape[0]/BatchSize)
                     
            
            print('Epoch loss: {:.6f}'.format(epoch_loss))
            
            if(abs(epoch_loss - prev_loss)<=0.00005):
                break
            
            prev_loss = epoch_loss

        return model


model = mlp(450,300,75) 
cr = nn.MSELoss() 
model = model_train(model,cr,n_epochs=300,l_rate=0.3) 




print('Finished training')

Z_test = torch.from_numpy(z_test)
y_test_onehot = torch.from_numpy(testLabel2)
y_test_onehot = y_test_onehot.float()
Z_test = Z_test.float()
output_test = model(Z_test) 
__,preds_ = output_test.data.max(1) 
loss_test = cr(output_test,y_test_onehot)
ts_corr = np.sum(np.equal(preds_.cpu().numpy(),testLabel))
ts_acc = ts_corr/3.16
print('Testing accuracy = '+str(ts_acc))

Z_val = torch.from_numpy(z_val)
y_val_onehot = torch.from_numpy(valLabel2)
y_val_onehot = y_val_onehot.float()
Z_val = Z_val.float()
output_val = model(Z_val) 
___,preds_val = output_val.data.max(1) 
loss_val = cr(output_val,y_val_onehot)
ts_corr_val = np.sum(np.equal(preds_val.cpu().numpy(),valLabel))
ts_acc_val = ts_corr_val/1.52
print('Validation accuracy = '+str(ts_acc_val))


Z_train = Z_train.float()
output_train = model(Z_train) 
___,preds_train = output_train.data.max(1) 

ts_corr_train = np.sum(np.equal(preds_train.cpu().numpy(),trainLabel))
ts_acc_train = ts_corr_train/11.03
print('Training accuracy = '+str(ts_acc_train))