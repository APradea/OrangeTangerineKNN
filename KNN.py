#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:21:06 2022

@author: anthelme
"""
import numpy as np 

class KNeighbors:
    
    def __init__(self, k, train, xtest):
        self.k = k
        self.train = train
        self.xtest = xtest 
        
    def distance(self, U, V):
        S = 0
        for i in range (len(U)):
            S += (U[i] - V[i]) ** 2
        return np.sqrt(S)
        
    def getneighbors(self,vecteurtest):
        distances = list()
        neighbors = list()
        for i in range (len(self.train)-1): 
            #Calcule la distance euclidienne entre le Vtest et le i-eme Vtrain 
            dist = self.distance(vecteurtest, self.train[i])
            distances.append((self.train[i],dist))
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
        
       
        
       
        
        