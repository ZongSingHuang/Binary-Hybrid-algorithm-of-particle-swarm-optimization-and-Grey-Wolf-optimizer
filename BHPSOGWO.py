# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1109/ACCESS.2019.2906757
https://www.mathworks.com/matlabcentral/fileexchange/78601-binary-optimization-using-hybrid-gwo-for-feature-selection
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class BHPSOGWO():
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
        self.gBest_X = np.zeros(self.num_dim)
        
        self.C1 = 0.5
        self.C2 = 0.5
        self.C3 = 0.5
        
        self.X = (np.random.uniform(size=[self.num_particle, self.num_dim]) > 0.5)*1.0
        
        self.gBest_curve = np.zeros(self.max_iter)
        self.gBest_score = 0
        
        self.V = 0.3 * np.random.normal(size=[self.num_particle, self.num_dim])
        
        self.w = 0.5 + np.random.uniform()/2
        
        self._iter = 0
        
        self._itter = self._iter + 1

        
    def opt(self):
        while(self._iter<self.max_iter):
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i, :])
                
                if score<self.score_alpha:
                    # # ---EvoloPy ver.---
                    # self.score_delta = self.score_beta
                    # self.X_delta = self.X_beta.copy()
                    # self.score_beta = self.score_alpha
                    # self.X_beta = self.X_alpha.copy()
                    # # ------------------
                    self.score_alpha = score.copy()
                    self.X_alpha = self.X[i, :].copy()
            
                if score>self.score_alpha and score<self.score_beta:
                    # # ---EvoloPy ver.---
                    # self.score_delta = self.score_beta
                    # self.X_delta = self.X_beta.copy()
                    # # ------------------
                    self.score_beta = score.copy()
                    self.X_beta = self.X[i, :].copy()
            
                if score>self.score_alpha and score>self.score_beta and score<self.score_delta:
                    self.score_delta = score.copy()
                    self.X_delta = self.X[i, :].copy()
            
            a = 2 - 2*self._iter/self.max_iter # (8)
            
            for i in range(self.num_particle):
                r1 = np.random.uniform(size=[self.num_dim])
                A1 = 2*a*r1 - a # (3)
                D_alpha = np.abs(self.C1*self.X_alpha - self.w*self.X[i, :]) #(19)
                cstep_alpha = self.sigmoid(-1*A1*D_alpha) # (18)
                bstep_alpha = ( cstep_alpha>=np.random.uniform(size=[self.num_dim]) )*1.0 # (17)
                X1 = ((self.X_alpha+bstep_alpha)>=1)*1.0 # (16)
                
                r1 = np.random.uniform(size=[self.num_dim])
                A2 = 2*a*r1 - a # (3)
                D_beta = np.abs(self.C2*self.X_beta - self.w*self.X[i, :]) #(19)
                cstep_beta = self.sigmoid(-1*A2*D_beta) # (18)
                bstep_beta = ( cstep_beta>=np.random.uniform(size=[self.num_dim]) )*1.0 # (17)
                X2 = ((self.X_beta+bstep_beta)>=1)*1.0 # (16)
                
                r1 = np.random.uniform(size=[self.num_dim])
                A3 = 2*a*r1 - a # (3)
                D_delta = np.abs(self.C3*self.X_delta - self.w*self.X[i, :]) #(19)
                cstep_delta = self.sigmoid(-1*A3*D_delta) # (18)
                bstep_delta = ( cstep_delta>=np.random.uniform(size=[self.num_dim]) )*1.0 # (17)
                X3 = ((self.X_delta+bstep_delta)>=1)*1.0 # (16)
                
                # with r
                r1 = np.random.uniform(size=[self.num_dim])
                r2 = np.random.uniform(size=[self.num_dim])
                r3 = np.random.uniform(size=[self.num_dim])
                self.V[i, :] = self.w*(self.V[i, :] \
                                       + self.C1*r1*(X1-self.X[i, :])
                                       + self.C2*r2*(X2-self.X[i, :])
                                       + self.C3*r3*(X3-self.X[i, :])) # (20)
                
                self.X[i, :] = self.sigmoid((X1+X2+X3)/3) + self.V[i, :] # (21) + (14)
                
                self.X[i, :] = ( self.X[i, :]>=np.random.uniform(size=[self.num_dim]) ) *1.0 # (14)
            
            # print('iter:', self._iter, ', score:', self.score_alpha)
            # print(self.X_alpha)
            # print('---')
            
            self._iter = self._iter + 1
            self.gBest_curve[self._iter-1] = self.score_alpha.copy()
            self.gBest_score = self.score_alpha.copy()
            self.gBest_X = self.X_alpha.copy()
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-10*(x-0.5))) # (15)
            