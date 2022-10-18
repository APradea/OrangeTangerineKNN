#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:13:33 2022

@author: anthelme
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from KNN import KNeighbors

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# for i in range (len(train.label)):
#     if train.label[i] == 0:
#         train.label[i] = 'Orange'
#     else: 
#         train.label[i] = 'Tangerine'


        
ytrain = np.array(train.label)
xtrain=np.array(train[['Fruit Size','Fruit Color','Leaf Size']])
xtrain_norm = np.zeros((200,2))
xtest=np.array(test[['Fruit Size','Fruit Color','Leaf Size']])


# for i in range (len(xtrain)):
#     xtrain_norm[i,0]= (xtrain[i,0] - np.mean(xtrain[:,0]))/np.std(xtrain[:,0])
#     xtrain_norm[i,1]= (xtrain[i,1] - np.mean(xtrain[:,1]))/np.std(xtrain[:,1])

# orange = np.zeros((200,2))        
# tangerine = np.zeros((200,2))  
# for i in range (len(train.label)):
#     if train.label[i] == 'Orange':
#         orange[i,0] = xtrain_norm[i, 0]
#         orange[i,1] = xtrain_norm[i, 1]
#     else: 
#         tangerine[i,0] = xtrain_norm[i, 0]
#         tangerine[i,1] = xtrain_norm[i, 1]
        
# plt.scatter(tangerine[:,0], tangerine[:,1], color = 'r')
# plt.scatter(orange[:,0], orange[:,1], color = 'b')
# plt.show()


kn=KNeighbors(3, xtrain, ytrain, xtest)
neighbors= kn.predict(xtest[0])

predict=list()
for x in xtest:
    pred = kn.predict(x)
    predict.append(pred)
predict = np.array(predict)