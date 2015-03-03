#!/bin/env python
#
# File name: svdd.py
# Copyright: (C) 2013 CityTouch
# 
# Incremental Support Vector Data Description with Approximate RKHS Linear Dependency Sparcity Test
#
# Initialization parameters:
#   
#   
#   kernel_function a kernel function 
#
#   state           a data sample structure (array, a text string - anything taken by a kernel function)
#   target          a target value (float)
#
#   adopt_thresh    approximate linear dependence threshold, [0, 1)
#   maxsize         maximum size of the dictionary 
#   adaptive        indicator if elimination is data-adaptive, True/False
#   forget_rate     frequency rate for forced elimination of the oldest entry, [0, 1] 


import numpy as np
import dict

import pickle


class svdd:  #  yet another InfoSphere Streams Killer

     
     def __init__(self, kernel_function, adopt_thresh, state, target, maxsize, adaptive=True, forget_rate=0.):
       
        self.dp = dict.dict(kernel_function, adopt_thresh, state, target, maxsize, adaptive, forget_rate)

        self.P = [[1]]
        self.Alpha = np.dot(self.dp.Kinv, target)
        
        self.rho = 0.1
        self.nu = 1-adopt_thresh
        
        self.C = 1/(self.dp.numel*self.rho*self.nu)

     def update(self, state, target):
        
        # dictionary update preceeds weights update. They are not independent:
        # dictionary needs to know weights Alpha to enable adaptive elimination
        self.dp.update( state, [1], self.Alpha )
        
        # now we update the weights using values precomputed in dicionary
        # to enable recursive updates here
        at = self.dp.at
        dt = self.dp.dt
        ktwid = self.dp.ktwid
        self.C = 1/(self.dp.numel*self.rho*self.nu)
                    
        if self.dp.addedFlag: # if a new entry was added  to the dictionary
          
          if self.dp.eliminatedFlag: # and an older/unrelevant entry was eliminated to make some room
          
              # I suspect the smart tricks got partly waisted here by matrix multiply
              # cause weights only become dependent on dictionary and not on all incoming samples.
              # Still need to figure out what's happening.
                                 
              self.Alpha = np.dot(self.dp.Kinv, self.dp.Targ)
              self.Alpha = np.where(self.Alpha<0,0,self.Alpha)
              self.Alpha = np.where(self.Alpha>self.C,self.C,self.Alpha)
              
          
          else:  # was enough room, so update as per original paper
          
              self.P = np.vstack( [np.hstack([self.P, np.zeros((self.dp.numel-1,1))]), np.hstack( [np.zeros((1,self.dp.numel-1)), [[1]]] ) ] )
              inno = ( target - np.dot(ktwid.T,self.Alpha) )/dt         
              self.Alpha = np.vstack([self.Alpha - np.dot(at,inno), inno])
              
              #self.Alpha = np.dot(self.dp.Kinv, self.dp.Targ)
              self.Alpha = np.where(self.Alpha<0,0,self.Alpha)
              self.Alpha = np.where(self.Alpha>self.C,self.C,self.Alpha)
              
              self.addedFlag = 1;
        
        else:    # we don't add an entry but update weights not to waste the sample 
                  # kinda smart incremental reduced rank regression.
          
              tmp = np.dot(self.P, at)
              qt = tmp / ( 1 + np.dot(at.T,tmp) )
              self.P = self.P - np.dot(qt,tmp.T)
              self.Alpha = self.Alpha + np.dot(self.dp.Kinv, qt*( target - np.dot(ktwid.T,self.Alpha) ))
              
              #self.Alpha = np.dot(self.dp.Kinv, self.dp.Targ)
              self.Alpha = np.where(self.Alpha<0,0,self.Alpha)
              self.Alpha = np.where(self.Alpha>self.C,self.C,self.Alpha)
              self.addedFlag = 0
     
     
     
     def update_set(self, data, targets, talk = -1):
        
        # feed in samples in a set, one sample at a time
        for i in range(len(data)): 

              self.update(data[i], targets[i])
   
              if not talk == -1:
                if not i % int(talk):
                    print("Iteration %d completed, %d in dictionary"  % (i, self.dp.numel))
                
                
                
     def ObjFunc(self):
     
         return 0.5*np.dot(np.dot(self.Alpha.T,self.dp.K),self.Alpha) - np.dot(self.Alpha.T,np.ones((self.dp.numel,1)))
     
        
     def query(self, sample):
               
        # compute the kernel of the input with the dictionary
        kernvals = self.dp.query(sample)
        # compute the weighted sum
        target =  np.dot(kernvals, self.Alpha)
          
        return target
     
        
     def save(self, f):
     
        pickle.dump(self.Alpha, open(f+'.alpha','w'))
        pickle.dump(self.dp.Dict, open(f+'.sv','w'))
     
        
     def load(self, f):
     
        self.Alpha = pickle.load(open(f+'.alpha','r'))
        self.dp.Dict = pickle.load(open(f+'.sv','r'))   
        
