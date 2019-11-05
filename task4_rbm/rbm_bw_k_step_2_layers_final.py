# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:07:35 2019

@author: jairam
"""
from sklearn.preprocessing import StandardScaler

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





class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay


        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias 
        hidden_probabilities = self._tanh(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t())  + self.visible_bias
        visible_probabilities = self._tanh(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)



        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size


        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def _tanh(self,x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)


        return random_probabilities





########## LOADING DATASET ##########
    
def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test

use_gpu=False
#reading data
from pandas import read_csv
filename='C:/Users/Chetan RS/Desktop/chetan/study/8th sem/dl/Assignment_2/task3_stacked_dataset2/10/data.csv'
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
X_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(Y_val)
# =============================================================================
# =============================================================================
X_train=X_train.float()
X_test=X_test.float()
X_val = X_val.float()
# =============================================================================
# =============================================================================
y_train=y_train.long()
y_test=y_test.long()
y_val = y_val.float()



########## CONFIGURATION ##########
B_Size = 1000
VISIBLE_UNITS = 784  
HIDDEN_UNITS = 420
CD_K = 4
EPOCHS = 80
########## TRAINING RBM ##########
print('Training RBM...')

rbm1 = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K)
prev_error = 0
for epoch in range(EPOCHS):
    epoch_error = 0.0

    for i in range(int(X_train.size()[0]/B_Size)):

        inputs = torch.index_select(X_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()
        if i == list(range(int(X_train.size()[0]/B_Size)))[-1]:
            inputs = X_train[(i*B_Size):].float()
        batch_error = rbm1.contrastive_divergence(inputs)

        epoch_error += batch_error

    epoch_error = epoch_error/(X_train.size()[0])
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
    
    if (abs(epoch_error - prev_error)<=0.00001):
        break
    prev_error = epoch_error

########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((X_train.size()[0], HIDDEN_UNITS))
train_labels = np.zeros(X_train.size()[0])
test_features = np.zeros((X_test.size()[0], HIDDEN_UNITS))
test_labels = np.zeros(X_test.size()[0])
val_features = np.zeros((X_val.size()[0], HIDDEN_UNITS))
val_labels = np.zeros(X_val.size()[0])

train_features = rbm1.sample_hidden(X_train).cpu().numpy()
train_labels = y_train.numpy() 

test_features = rbm1.sample_hidden(X_test).cpu().numpy()
test_labels = y_test.numpy() 

val_features = rbm1.sample_hidden(X_val).cpu().numpy()
val_labels = y_val.numpy() 

        
        
        
###########################2nd layer###################################
########## CONFIGURATION ##########
B_Size = 1000
VISIBLE_UNITS = 420  
HIDDEN_UNITS = 320
CD_K = 4
EPOCHS = 80


Z1_train = torch.from_numpy(train_features)
v1_train = torch.from_numpy(train_labels)

Z1_test = torch.from_numpy(test_features)
v1_test = torch.from_numpy(test_labels)

Z1_val = torch.from_numpy(val_features)
v1_val = torch.from_numpy(val_labels)

########## TRAINING RBM ##########
print('Training RBM...')

rbm2 = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K)
prev_error = 0
for epoch in range(EPOCHS):
    epoch_error = 0.0

    for i in range(int(Z1_train.size()[0]/B_Size)):

        inputs = torch.index_select(Z1_train,0,torch.linspace(i*B_Size,(i+1)*B_Size - 1,steps=B_Size)
                                  .long()).float()
        if i == list(range(int(Z1_train.size()[0]/B_Size)))[-1]:
            inputs = Z1_train[(i*B_Size):].float()
        batch_error = rbm2.contrastive_divergence(inputs)

        epoch_error += batch_error

    epoch_error = epoch_error/(Z1_train.size()[0])
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
    
    if (abs(epoch_error - prev_error)<=0.00001):
        break
    prev_error = epoch_error

########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features_2 = np.zeros((Z1_train.size()[0], HIDDEN_UNITS))
train_labels_2 = np.zeros(Z1_train.size()[0])
test_features_2 = np.zeros((Z1_test.size()[0], HIDDEN_UNITS))
test_labels_2 = np.zeros(Z1_test.size()[0])
val_features_2 = np.zeros((Z1_val.size()[0], HIDDEN_UNITS))
val_labels_2 = np.zeros(Z1_val.size()[0])

train_features_2 = rbm2.sample_hidden(Z1_train).cpu().numpy()
train_labels_2 = v1_train.numpy() 

test_features_2 = rbm2.sample_hidden(Z1_test).cpu().numpy()
test_labels_2 = v1_test.numpy() 

val_features_2 = rbm2.sample_hidden(Z1_val).cpu().numpy()
val_labels_2 = v1_val.numpy()   


###########Error###############

Z1_train_re = rbm2.sample_visible(torch.from_numpy(train_features_2)).cpu().numpy()

Z1_test_re = rbm2.sample_visible(torch.from_numpy(test_features_2)).cpu().numpy()

Z1_val_re = rbm2.sample_visible(torch.from_numpy(val_features_2)).cpu().numpy()

X_train_re = rbm1.sample_visible(torch.from_numpy(Z1_train_re)).cpu().numpy()

X_test_re = rbm1.sample_visible(torch.from_numpy(Z1_test_re)).cpu().numpy()

X_val_re = rbm1.sample_visible(torch.from_numpy(Z1_val_re)).cpu().numpy()

loss_train_rec = X_train - torch.from_numpy(X_train_re)
loss_train_rec = loss_train_rec.numpy()
loss_train_rec = loss_train_rec**2
loss_train_rec = loss_train_rec.sum()

loss_val_rec = X_val - torch.from_numpy(X_val_re)
loss_val_rec = loss_val_rec.numpy()
loss_val_rec = loss_val_rec**2
loss_val_rec = loss_val_rec.sum()

loss_test_rec = X_test - torch.from_numpy(X_test_re)
loss_test_rec = loss_test_rec.numpy()
loss_test_rec = loss_test_rec**2
loss_test_rec = loss_test_rec.sum()

########################classifier############################
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
        x = F.tanh(x)
        x = self.N2(x)          
        x = F.tanh(x) 
        x = self.N3(x)
        x = F.softmax(x)
        return x


trainLabel2 = np.zeros((X_train.size()[0],5))
testLabel2 = np.zeros((X_test.size()[0],5))
valLabel2 = np.zeros((X_val.size()[0],5))
for i in range(trainLabel2.shape[0]):
    trainLabel2[i,train_labels_2.astype(int)[i]] = 1
for j in range(testLabel2.shape[0]):
    testLabel2[j,test_labels_2.astype(int)[j]] = 1
for k in range(valLabel2.shape[0]):
    valLabel2[k,val_labels_2.astype(int)[k]] = 1
    
#Z_train = torch.from_numpy(train_features_2)
y_train_onehot = torch.from_numpy(trainLabel2)


def train_model(model,criterion,n_epochs,l_rate):
           
        prev_loss = 0
        
        for epoch in range(n_epochs):
            
            print('Epoch {}/{}'.format(epoch, n_epochs - 1))

            running_loss = 0.0           
            
            batch = 0
            BatchSize = 1500
            
            for i in range(int(X_train.shape[0]/BatchSize)):
                inputs = torch.index_select(X_train,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize)
                                          .long()).float()
                labels = torch.index_select(y_train_onehot,0,torch.linspace(i*BatchSize,(i+1)*BatchSize - 1,steps=BatchSize)
                                          .long()).float()

               # print(inputs.size())
                inputs = Variable(inputs)
                labels = Variable(labels)   
                
                model.zero_grad() 
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                
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
           
            epoch_loss = running_loss/int(X_train.shape[0]/BatchSize)
                     
            
            print('Epoch loss: {:.6f}'.format(epoch_loss))
            
            if(abs(epoch_loss - prev_loss)<=0.00005):
                break
            
            prev_loss = epoch_loss

        return model


model = mlp(784,420,320) 
model.N1.weight.data.copy_(rbm1.weights.data.t())
model.N1.bias.data.copy_(rbm1.hidden_bias.data)
model.N2.weight.data.copy_(rbm2.weights.data.t())
model.N2.bias.data.copy_(rbm2.hidden_bias.data)
criterion = nn.MSELoss() 
model = train_model(model,criterion,n_epochs=100,l_rate=0.5) 




print('Finished training')


#Z_test = torch.from_numpy(test_features_2)
y_test_onehot = torch.from_numpy(testLabel2)
#Z_test = Z_test.float()
output_test = model(X_test) 
__,preds_ = output_test.data.max(1) 
loss_test = criterion(Variable(preds_.float()),Variable(torch.from_numpy(Y_test).float()))
ts_corr = np.sum(np.equal(preds_.cpu().numpy(),Y_test))
ts_acc = ts_corr/(X_test.size()[0]/100)
print('Testing accuracy = '+str(ts_acc))



#Z_val = torch.from_numpy(val_features_2)
y_val_onehot = torch.from_numpy(valLabel2)
#Z_val = Z_val.float()
output_val = model(X_val) 
___,preds_val = output_val.data.max(1) 
loss_val = criterion(Variable(preds_val.float()),Variable(torch.from_numpy(Y_val).float()))
ts_corr_val = np.sum(np.equal(preds_val.cpu().numpy(),Y_val))
ts_acc_val = ts_corr_val/(X_val.size()[0]/100)
print('Validation accuracy = '+str(ts_acc_val))


#Z_train = Z_train.float()
output_train = model(X_train) 
___,preds_train = output_train.data.max(1) 

ts_corr_train = np.sum(np.equal(preds_train.cpu().numpy(),Y_train))
ts_acc_train = ts_corr_train/(X_train.size()[0]/100)
print('Training accuracy = '+str(ts_acc_train))
print('Training Confusion matrix :\n',confusion_matrix(preds_train.numpy(),Y_train))
print('Testing Confusion matrix :\n',confusion_matrix(preds_.numpy(),Y_test))