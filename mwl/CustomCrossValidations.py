# -*- coding: utf-8 -*-
"""
Created on Fri Aug  26 11:40:19 2022

@author: ibargiotas
"""
import numpy as np
import random
#from numpy.random import default_rng


class CustomCVs():
    
    def TimeCV(T, per_test, weighting,seed_k):
        """ 
        T: is a two-column vector where COLUMN0 indicates the number of the task in time order, and 
        COLUMN1 indicates the individual.
        per_test: is the percentage (from 0 to 1) of test set
        """
        #np.random.seed(seed_k)
        per_temp = 0
        ID = (T[:,0]*0>1) #A vector full of false's
        while per_temp<=per_test:
            #idx = np.random.randint(0, ID.shape[0], size=1)
            idx =  random.choice(np.arange(0,ID.shape[0]))
            full_idx =  np.all([T[:,0]>=T[idx,0],T[:,1]==T[idx,1]],axis=0)
            ID[np.where(full_idx)[0]] = True
            
            per_temp = np.count_nonzero(ID)/ID.shape[0]
            
        test_id = np.where(ID)[0]
        train_id = np.where(~ID)[0]
        
        if weighting == True:
            U = np.unique(T[test_id,1])
            for i in U:
                x_tmp = train_id[T[train_id,1]==i]
                train_id = np.append(train_id, x_tmp)
                train_id = np.append(train_id, x_tmp)
#                print(i, x_tmp, test_id)
            
        return train_id,test_id  
        
        
    
    def add_indexes(A):
        """
        A: a single-column with the number of the task, ranked by pilots in time order.
        """
        C = np.vstack((A,np.zeros((A.size)))).T
        p = 3
        C[0,1] = p
        
        for k in range(1, A.size):
            if C[k,0] > C[k-1,0]:
                C[k,1] = p
            else:
                p = p+1
                C[k,1] = p
                
        return C
        
            