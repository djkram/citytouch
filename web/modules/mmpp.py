#!/bin/env python
#
# File name: mmpp.py
# Copyright: (C) 2013 CityTouch
# 
# Markov-modulated non-homogeneous Poisson process
#
# Initialization parameters:
#
#   event_length    expected event length (in percent of time step)
#
# Training parameters:
#
#   N         data as a 2D array of count time series per 'week'
#   iters     MCMC iterations as [total, burn-in]
#
# Output:
#
#   routine   periodic routine component ('day' and 'time of the day' modulated)
#   events    probability of an event

import numpy as np

from scipy.cluster.vq import kmeans2, whiten
from scipy.special import gammaln


class mmpp:
     
  def __init__(self, event_length=0.05):
    
    self.priors = {}
    self.priors['aL'], self.priors['bL'] = 100.0, 1.0     # lambda0, baseline rate
    self.priors['aD'] = np.zeros((1,7))   + 5.0      # day effect dirichlet params
    self.priors['aH'] = np.zeros((24,7)) + 1.0      # time of day effect dirichlet params

    self.mn = 10000
    self.priors['z00'], self.priors['z01'] = 0.999*self.mn, 0.001*self.mn     # z(t) event process
    self.priors['z10'], self.priors['z11'] = (1-event_length)*self.mn, event_length*self.mn     
    self.priors['aE'], self.priors['bE'] = 10.0, 10.0       # gamma(t), or NBin, for event # process

    self.priors['MODE'] = 0

       
  def dirpdf(self,X,A):			# evaluate a dirichlet distribution
    k = X.size
    if k==1:
      p=1
    else:   
      logp = np.sum( (A-1) * np.log(X) ) - np.sum(gammaln(A)) + gammaln(np.sum(A))
      p = np.exp(logp);  
    return p  
      
  def dirlnpdf(self,X,A):			# eval log(dirichlet)
    k = X.size 
    if k==1:
      p=1
      logp = 0
    else:
      logp = np.sum( (A-1) * np.log(X) ) - np.sum(gammaln(A)) + gammaln(np.sum(A))  
    return logp
    
  def poisspdf(self,X,L):			# poisson distribution
    lnp = -L - gammaln(X+1) + np.log(L)*X
    p = np.exp(lnp)
    return p
    
  def poisslnpdf(self,X,L):			# log(poisson)
    lnp = -L -gammaln(X+1) + np.log(L)*X
    return lnp
    
  def nbinpdf(self,X,R,P):			# negative binomial distribution
    lnp = gammaln(X+R)-gammaln(R)-gammaln(X+1)+np.log(P)*R+np.log(1-P)*X
    p = np.exp(lnp)
    return p
    
  def nbinlnpdf(self,X,R,P):			# log(neg binomial)
    lnp = gammaln(X+R)-gammaln(R)-gammaln(X+1)+np.log(P)*R+np.log(1-P)*X
    return lnp
   

  def draw_M_Z(self,Z):
    # GIVEN Z, SAMPLE M
    
    
    n01 = np.array(np.logical_and(Z.T.flat[1:-1]==0, Z.T.flat[2:]==1).astype(int).nonzero()).size
    n0 = np.array((Z.T.flat[1:-1]==0).astype(int).nonzero()).size
    
    n10 = np.array(np.logical_and(Z.T.flat[1:-1]==1, Z.T.flat[2:]==0).astype(int).nonzero()).size
    n1 = np.array((Z.T.flat[1:-1]==1).astype(int).nonzero()).size
    
    z0 = np.random.beta(n01+self.priors['z01'], n0-n01+self.priors['z00'])
    z1 = np.random.beta(n10+self.priors['z10'], n1-n10+self.priors['z11'])
    
    M = np.array([[1-z0, z1], [z0, 1-z1]]) 
   
    return M
    

  def draw_Z_NLM(self,N,L,M):

    N0, NE, Z, ep = N.copy(), np.zeros(N.shape), np.zeros(N.shape), 1e-50
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% FIRST SAMPLE Z, N0, NE:
    
    PRIOR = np.array( np.linalg.matrix_power(np.mat(M),100) * np.mat([1, 0]).T)
    po=np.zeros((2, N.size))
    p=np.zeros((2,N.size))
     
    for t in range(N.size):
      if not N.T.flat[t] == -1:
        po[0,t] = self.poisspdf(N.T.flat[t],L.T.flat[t]) + ep
        po[1,t] = np.sum( self.poisspdf(np.arange(N.T.flat[t]+1),L.T.flat[t]) * self.nbinpdf(np.arange(N.T.flat[t],-1,-1), self.priors['aE'], self.priors['bE']/(1+self.priors['bE'])) ) + ep
      else: 
        po[0,t], po[1,t] = 1, 1
    
    # Compute forward (filtering) posterior marginals
    #print PRIOR, np.array([po[:,0]])
    p[:,0] = PRIOR.T * np.array([po[:,0]])  
    p[:,0] = p[:,0]/np.sum(p[:,0])
        
    for t in range(1,N.size):
      p[:,t] = np.array( np.dot(M, p[:,t-1]) ) * po[:,t]
      p[:,t] = p[:,t]/np.sum(p[:,t])   
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Do backward sampling
    
    for t in np.arange(N.size-1,-1,-1):
      if (np.random.rand(1) > p[0,t]):                          # if event at time t
        if not N.T.flat[t] == -1:  # missing data
          Z.T.flat[t] = 1 
          # likelihood of all possible event/normal combinations
          # all possible values of N(E)
          ptmp = self.poisslnpdf(np.arange(N.T.flat[t]+1),L.T.flat[t]) + self.nbinlnpdf( np.arange(N.T.flat[t],-1,-1), self.priors['aE'], self.priors['bE']/(1+self.priors['bE'])) 
          ptmp=ptmp-np.max(ptmp)
          ptmp=np.exp(ptmp)
          ptmp=ptmp/np.sum(ptmp)
          

          rndthresh = np.random.rand(1)
          csum = np.cumsum(ptmp) 
          
          ind = (csum > rndthresh).nonzero()[0]
          
          N0.T.flat[t] = float(np.min(ind))   # 
          NE.T.flat[t] = N.T.flat[t]-N0.T.flat[t]                             # and compute NE
          
        else:
          Z.T.flat[t]=1
          N0.T.flat[t]=np.random.poisson(L.T.flat[t], 1)
          NE.T.flat[t]=np.random.negative_binomial(self.priors['aE'],self.priors['bE']/(1+self.priors['bE']))
      
      else:
        if not N.T.flat[t] == -1: # missing data
          Z.T.flat[t] = 0
          N0.T.flat[t] = N.T.flat[t]
          NE.T.flat[t] = 0              # no event at time t
        else:
          Z.T.flat[t]=0
          N0.T.flat[t]=np.random.poisson(L.T.flat[t], 1)
          NE.T.flat[t]=0

      ptmp = np.zeros((2,1))
      ptmp[Z.T.flat[t]] = 1    # compute backward influence
      if t>1:
        p[:,t-1] = np.array(np.dot(M.T,ptmp).T) * p[:,t-1]
        p[:,t-1] = p[:,t-1]/np.sum(p[:,t-1]) 
    
    return Z, N0, NE


  def draw_L_N0(self,N0,equiv_type=False):
    
    Nd, Nh = 7, N0.shape[0]
    
    # overall average rate
    if self.priors['MODE']:
      L0 = (np.sum(np.sum(N0))+self.priors['aL']) / (N0.size+self.priors['bL'])
    else:
      L0 = np.random.gamma( np.sum(np.sum(N0)) + self.priors['aL'], 1.0/float(N0.size+self.priors['bL']))
      
    L = np.zeros(N0.shape) + L0
    
    # day effect
    D = np.zeros((1,Nd))
    for i in range(Nd):
      alpha = np.sum(np.sum(N0[:,i::7])) + self.priors['aD'][0][i]
      
      if self.priors['MODE']:
        D[0][i] = alpha-1           # mode of Gamma(a,1) distribution
      else:
        D[0][i] = np.random.gamma(alpha,1)
   

    # time of day effect
    A = np.zeros((Nh,Nd))
    for tau in range(A.shape[1]):
      for i in range(A.shape[0]):
        alpha = np.sum(np.sum(N0[i,tau::7])) + self.priors['aH'][i][0]
        if self.priors['MODE']:
          A[i,tau] = alpha-1           # mode of Gamma(a,1) distribution
        else:
          A[i,tau] = np.random.gamma(alpha,1)
    
    
    if equiv_type:
      
      #  This is not implemented yet as no calendar is given. Both D (total counts) and A (profiles)
      #  have to be averaged over similar day types from a known calendar. 
      
      #for j in equiv_type:
      #
      #  D[D==j] = np.mean(D[D==j]) 
      #

      D=D/np.mean(D)
      
      #A(:,[1,7]) = repmat(mean(A(:,[1,7]),2),[1,2])
      #A(:,2:6)=repmat(mean(A(:,2:6),2),[1,5]);
      
    else:
      D = D/np.mean(D)
      A = A

    for tau in range(A.shape[1]):
      A[:,tau]=A[:,tau]/np.mean(A[:,tau])
   
    #  COMPUTE L(t)
    for d in range(L.shape[1]):
      for t in range(L.shape[0]):
        dd=np.remainder(d,7)
        L[t,d] = L0 * D[0][dd] * A[t,dd]
   
    return L


  def train(self, N, iters, equiv_type=False):
    
    N = N.T
    
    Niter, Nburn = iters[0], iters[1]
    
    Z=np.zeros(N.shape)
    N0=np.maximum(N,1) # element-wise max(N,1)
    NE=np.zeros(N.shape)
    L=(N.copy()+5)/2
    M=np.array([[0.999, 0.5], [0.001, 0.5]])
    
    Nd, Nh= 7, N.shape[0]
    
    samples= {}
    samples['L'] = np.zeros((L.shape[0],L.shape[1], Niter));
    samples['Z'] = np.zeros((Z.shape[0],Z.shape[1], Niter));
    samples['M'] = np.zeros((M.shape[0],M.shape[1], Niter));
    samples['N0'] = np.zeros((N0.shape[0],N0.shape[1],Niter));
    samples['NE'] = np.zeros((NE.shape[0],NE.shape[1],Niter));
    samples['logp_NgLM'] = np.zeros((1,Niter))
    samples['logp_NgLZ'] = np.zeros((1,Niter));
    
    
    # MCMC inference loop
    for i in range(Niter+Nburn):
    
      L = self.draw_L_N0( N0, equiv_type )
    
      Z,N0,NE = self.draw_Z_NLM( N, L, M )
            
      M = self.draw_M_Z( Z )
      
      # save states
      if i>=Nburn:
        samples['L'][:,:,i-Nburn] = L
        samples['Z'][:,:,i-Nburn] = Z
        samples['M'][:,:,i-Nburn] = M
        samples['N0'][:,:,i-Nburn] = N0
        samples['NE'][:,:,i-Nburn] = NE
      
      print('.')
    
    # posterior estimates for the routine counts and the events process    
    routine = np.mean(samples['L'][:,:,:],axis=2).T.reshape(N.size)
    events = np.mean(samples['Z'][:,:,:],axis=2).T.reshape(N.size) 
  
    return routine, events


  def nnpp(self, N, Kind=np.arange(2)):
    '''
    Find typical daily profiles with k-means clustering

    Inputs:
    N is a [number_of_days, number_of_measurements_in_a_day]
    Kind contains the indices of days for centroids initialization

    Output:
    Time series composed of centroids and an array of day labels
    '''
    
    #Nw = whiten(N)
    Nw = N
    K_mat = Nw[Kind,:]

    result, labs = kmeans2(Nw, K_mat, minit='matrix')
    
    nnpp_result = np.zeros(Nw.shape)
    for i,l in enumerate(labs):
      nnpp_result[i,:] = result[l]
    
    return nnpp_result, labs


  def nnpp_tune_one(self, N, Kfix=-1):
    '''
    Find the day index within a week with most typical profile
    '''
    if Kfix>0:
      MSE = 1e10*np.ones((1,7))
      
      for i in range(7):
        if not i in Kfix:
          Kind = Kfix.append(i)
          nnpp_result, labs = nnpp(N, Kind)
      
          MSE[i] = np.sum(np.sum( (N - nnpp_result)*(N - nnpp_result) ))
    
    else:  
      MSE = 1e10*np.ones((1,7))
      for i in range(7):
        nnpp_result, labs = nnpp(N, Kind=i)
        MSE[i] = np.sum(np.sum( (N - nnpp_result)*(N - nnpp_result) ))
    
    return np.argmin(MSE)[0]  


  def nnpp_detect(self, N, nnpp_result, labs):
    '''
    Detect events (bursts)

    Inputs:
    N is a [number_of_days, number_of_measurements_in_a_day] count
    nnpp_result is a [number_of_days, number_of_measurements_in_a_day] expected count
    labs is days labels
    
    Output:
    Events
    '''
    
    thresh = 0.5*np.mean(np.mean(N))
    print thresh
    
    events = np.zeros(N.shape)
    for i,l in enumerate(labs):
    
      delta = N[i,:] - nnpp_result[i,:]
      rel_delta = np.fabs(delta)/np.mean(N[i,:])
      
      crd = np.cumsum(rel_delta)        
      icrd = np.cumsum(rel_delta[::-1])[::-1] 
      
      events[i, (icrd*crd*rel_delta>thresh).nonzero()] = 1
    
    return events
