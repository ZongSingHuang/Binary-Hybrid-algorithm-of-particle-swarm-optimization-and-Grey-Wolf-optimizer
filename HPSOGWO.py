# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1016/j.advengsoft.2013.12.007
https://seyedalimirjalili.com/gwo
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class HPSOGWO():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter

        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = np.zeros(self.num_dim)
        self.X_beta = np.zeros(self.num_dim)
        self.X_delta = np.zeros(self.num_dim)

        self.X = (np.random.uniform(size=[self.num_particle, self.num_dim]) > 0.5)*1.0
        
        self.gBest_curve = np.zeros(self.max_iter)
        
        self.V = 0.3 * np.random.normal(size=[self.num_particle, self.num_dim])
        
        self.w = 0.5 + np.random.uniform()/2
        
        self._iter = 0
        
        self._itter = self._iter + 1

        
    def opt(self):
        while(self._iter<self.max_iter):
            print(111)
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i, :])
                
                if score<self.score_alpha:
                    self.score_alpha = score.copy()
                    self.X_alpha = self.X[i, :]
            
                if score>self.score_alpha and score<self.score_beta:
                    self.score_beta = score.copy()
                    self.X_beta = self.X[i, :]
            
                if score>self.score_alpha and score>self.score_beta and score<self.score_delta:
                    self.score_delta = score.copy()
                    self.X_delta = self.X[i, :]
            
            a = 2 - 2*self._iter/self.max_iter # (8)
            
            for i in range(self.num_particle):
                for j in range(self.num_dim):
                    r1 = np.random.uniform()
                    r2 = np.random.uniform()
                    A1 = 2*a*r1 - a # (3)
                    C1 = 0.5
                    D_alpha = np.abs(C1*self.X_alpha[j] - self.w*self.X[i, j]) #(19)
                    v1 = self.sigmoid(A1*D_alpha) # (18)
                    if v1<np.random.uniform():
                        v1 = 0.0
                    else:
                        v1 = 1.0
                    X1 = ((self.X_alpha[j]+v1)>=1)*1.0
                    
                    r1 = np.random.uniform()
                    r2 = np.random.uniform()
                    A2 = 2*a*r1 - a # (3)
                    C2 = 0.5
                    D_beta = np.abs(C2*self.X_beta[j] - self.w*self.X[i, j]) #(19)
                    v1 = self.sigmoid(A2*D_beta) # (18)
                    if v1<np.random.uniform():
                        v1 = 0.0
                    else:
                        v1 = 1.0
                    X2 = ((self.X_beta[j]+v1)>=1)*1.0
                    
                    r1 = np.random.uniform()
                    r2 = np.random.uniform()
                    r3 = np.random.uniform()
                    A3 = 2*a*r1 - a # (3)
                    C3 = 0.5
                    D_delta = np.abs(C3*self.X_delta[j] - self.w*self.X[i, j]) #(19)
                    v1 = self.sigmoid(A3*D_delta) # (18)
                    if v1<np.random.uniform():
                        v1 = 0.0
                    else:
                        v1 = 1.0
                    X3 = ((self.X_delta[j]+v1)>=1)*1.0

                    self.V[i, j] = self.w*(self.V[i, j] \
                                           + C1*r1*(X1-self.X[i, j])
                                           + C2*r2*(X2-self.X[i, j])
                                           + C3*r3*(X3-self.X[i, j])) # (20)

                    xx = self.sigmoid((X1+X2+X3)/3) + self.V[i, j] # (21)
                    if xx<np.random.uniform():
                        xx = 0.0
                    else:
                        xx = 1.0
                    self.X[i, j] = xx
                
            
                print(self.score_alpha)
                print(self.X_alpha)
                print('---')
            
            self._iter = self._iter + 1
            self.gBest_curve[self._iter-1] = self.score_alpha.copy()
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-10*(x-0.5))) # (15)
            