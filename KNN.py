#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:21:06 2022

@author: anthelme
"""
import numpy as np 

class KNeighbors:
    
    def __init__(self, k, xtrain, ytrain, xtest):
        self.k = k
        self.xtrain = xtrain
        self.ytrain = ytrain 
        self.xtest = xtest 
        
    def distance(self, U, V):
        S = 0
        for i in range (len(U)):
            S += (U[i] - V[i]) ** 2
        return np.sqrt(S)
        
    def getneighbors(self,vecteurtest):
        distances = list()
        neighbors = list()
        for i in range (len(self.xtrain)): 
            dist = self.distance(vecteurtest, self.xtrain[i])
            interm = np.append(self.xtrain[i], self.ytrain[i])
            distances.append((interm,dist))
        distances.sort(key=lambda tup: tup[1])
        for i in range (self.k):
            neighbors.append(distances[i][0])
        return neighbors
      
        # Make a classification prediction with neighbors
    def predict(self,vecteurtest):
        neighbors= self.getneighbors(vecteurtest)
        outputval=list()
        for row in neighbors: 
            outputval.append(row[-1])
        prediction = max(set(outputval), key=outputval.count)
        return prediction
        
       
        
       
        
        