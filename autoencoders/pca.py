# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:16:48 2019

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

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test

path = "C:/Users/jairam/Desktop/dl/Assignment_2/data1/data1"
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
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
X_train = dataset_train[:,0:828]
y_train = dataset_train[:,828:]
y_train = y_train.squeeze()
#val
dataset_val=np.asarray(dataset_val)
X_val = dataset_val[:,0:828]
y_val = dataset_val[:,828:]
y_val = y_val.squeeze()
#test
dataset_test=np.asarray(dataset_test)
X_test = dataset_test[:,0:828]
y_test = dataset_test[:,828:]
y_test = y_test.squeeze()

trainLabel=y_train.astype(int)
testLabel=y_test.astype(int)
valLabel=y_val.astype(int)
trainFeats=X_train
testFeats=X_test
valFeats=X_val

# Compute the mean of the data
mean_vec = np.mean(X_train, axis=0)
# Compute the covariance matrix
cov_mat = (X_train - mean_vec).T.dot((X_train - mean_vec)) / (X_train.shape[0]-1)
# OR we can do this with one line of numpy:
cov_mat_np = np.cov(X.T)
# Compute the eigen values and vectors using numpy
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Only keep a certain number of eigen vectors based on 
# the "explained variance percentage" which tells us how 
# much information (variance) can be attributed to each 
# of the principal components
exp_var_percentage = 95 # Threshold of 97% explained variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
num_vec_to_keep = 0
for index, percentage in enumerate(cum_var_exp):
  if percentage > exp_var_percentage:
    num_vec_to_keep = index + 1
    break
# Compute the projection matrix based on the top eigen vectors
num_features = X_train.shape[1]
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))
# Project the data 
X_pca_data_train = X_train.dot(proj_mat)
X_pca_data_val = X_val.dot(proj_mat)
X_pca_data_test = X_test.dot(proj_mat)







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
for i in range(trainLabel.shape[0]):
    trainLabel2[i,trainLabel[i]] = 1
for j in range(testLabel.shape[0]):
    testLabel2[j,testLabel[j]] = 1
for k in range(valLabel.shape[0]):
    valLabel2[k,valLabel[k]] = 1

data_train = np.concatenate((X_train,trainLabel2), axis = 1)
data_test = np.concatenate((X_test,testLabel2), axis = 1)
data_test = np.concatenate((X_val,valLabel2), axis = 1)
    

Z_train = torch.from_numpy(X_pca_data_train)
y_train_onehot = torch.from_numpy(trainLabel2) 


# Definining the training routine
def train_model(model,criterion,n_epochs,l_rate):
           
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
           
            epoch_loss = running_loss/int(data_train.shape[0]/BatchSize)
                     
            
            print('Epoch loss: {:.6f}'.format(epoch_loss))
            
            if(abs(epoch_loss - prev_loss)<=0.00005):
                break
            
            prev_loss = epoch_loss

        return model


model = mlp(352,200,100) 
criterion = nn.MSELoss() 
model = train_model(model,criterion,n_epochs=300,l_rate=0.5) 




print('Finished training')

Z_test = torch.from_numpy(X_pca_data_test)
y_test_onehot = torch.from_numpy(testLabel2)
Z_test = Z_test.float()
output_test = model(Z_test) 
__,preds_ = output_test.data.max(1) 

ts_corr = np.sum(np.equal(preds_.cpu().numpy(),testLabel))
ts_acc = ts_corr/3.16
print('Testing accuracy = '+str(ts_acc))

Z_val = torch.from_numpy(X_pca_data_val)
y_val_onehot = torch.from_numpy(valLabel2)
Z_val = Z_val.float()
output_val = model(Z_val) 
___,preds_val = output_val.data.max(1) 

ts_corr_val = np.sum(np.equal(preds_val.cpu().numpy(),valLabel))
ts_acc_val = ts_corr_val/1.52
print('Validation accuracy = '+str(ts_acc_val))


Z_train = Z_train.float()
output_train = model(Z_train) 
___,preds_train = output_train.data.max(1) 

ts_corr_train = np.sum(np.equal(preds_train.cpu().numpy(),trainLabel))
ts_acc_train = ts_corr_train/11.03
print('Training accuracy = '+str(ts_acc_train))









