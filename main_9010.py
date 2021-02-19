# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from BHPSOGWO import BHPSOGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import functools
import time
import warnings
#------------------------------------------------------------------------------


warnings.filterwarnings("ignore")
np.random.seed(42)
#------------------------------------------------------------------------------


# 讀資料
Breastcancer = pd.read_csv('Breastcancer.csv', header=None).values
BreastEW = pd.read_csv('BreastEW.csv', header=None).values
Congress = pd.read_csv('Congress.csv', header=None).values
Exactly = pd.read_csv('Exactly.csv', header=None).values
Exactly2 = pd.read_csv('Exactly2.csv', header=None).values
HeartEW = pd.read_csv('HeartEW.csv', header=None).values
Ionosphere = pd.read_csv('Ionosphere.csv', header=None).values
KrVsKpEW = pd.read_csv('KrVsKpEW.csv', header=None).values
Lymphography = pd.read_csv('Lymphography.csv', header=None).values
M_of_n = pd.read_csv('M-of-n.csv', header=None).values
PenglungEW = pd.read_csv('PenglungEW.csv', header=None).values
Sonar = pd.read_csv('Sonar.csv', header=None).values
SpectEW = pd.read_csv('SpectEW.csv', header=None).values
Tic_tac_toe = pd.read_csv('Tic-tac-toe.csv', header=None).values
Vote = pd.read_csv('Vote.csv', header=None).values
WaveformEW = pd.read_csv('WaveformEW.csv', header=None).values
Wine = pd.read_csv('Wine.csv', header=None).values
Zoo = pd.read_csv('Zoo.csv', header=None).values

X1, y1 = Breastcancer[:, :-1], Breastcancer[:, -1]
X2, y2 = BreastEW[:, :-1], BreastEW[:, -1]
X3, y3 = Congress[:, :-1], Congress[:, -1]
X4, y4 = Exactly[:, :-1], Exactly[:, -1]
X5, y5 = Exactly2[:, :-1], Exactly2[:, -1]
X6, y6 = HeartEW[:, :-1], HeartEW[:, -1]
X7, y7 = Ionosphere[:, :-1], Ionosphere[:, -1]
X8, y8 = KrVsKpEW[:, :-1], KrVsKpEW[:, -1]
X9, y9 = Lymphography[:, :-1], Lymphography[:, -1]
X10, y10 = M_of_n[:, :-1], M_of_n[:, -1]
X11, y11 = PenglungEW[:, :-1], PenglungEW[:, -1]
X12, y12 = Sonar[:, :-1], Sonar[:, -1]
X13, y13 = SpectEW[:, :-1], SpectEW[:, -1]
X14, y14 = Tic_tac_toe[:, :-1], Tic_tac_toe[:, -1]
X15, y15 = Vote[:, :-1], Vote[:, -1]
X16, y16 = WaveformEW[:, :-1], WaveformEW[:, -1]
X17, y17 = Wine[:, :-1], Wine[:, -1]
X18, y18 = Zoo[:, :-1], Zoo[:, -1]
#------------------------------------------------------------------------------


def fitness(x, X, y):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, x[i, :].astype(bool)], y, cv=skf)
            loss[i] = 0.99*(1-score.mean()) + 0.01*(np.sum(x[i, :])/X.shape[1])
        else:
            loss[i] = np.inf
            # print(666)
    return loss
#------------------------------------------------------------------------------


d = -1
g = 70
p = 8
times = 20
table = np.zeros((7, 18)) # ['avg acc', '% selected', 'avg time', 'avg loss', 'worst loss', 'best loss', 'std loss']
table[4, :] = -np.ones(18)*np.inf # worst
table[5, :] = np.ones(18)*np.inf # best
all_for_std = np.zeros((times, 18))
skf = StratifiedKFold(n_splits=10, shuffle=True)
#------------------------------------------------------------------------------


for i in range(times):
    total_time = time.time()
    #------------------------------------------------------------------------------
    
    
    start1 = time.time()
    loss1 = functools.partial(fitness, X=X1, y=y1)
    optimizer = BHPSOGWO(fit_func=loss1, 
                         num_dim=X1.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 0]: table[4, 0] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 0]: table[5, 0] = optimizer.gBest_score
    table[3, 0] += optimizer.gBest_score
    table[2, 0] += time.time()-start1
    all_for_std[i, 0] = optimizer.gBest_score
    
    table[0, 0] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X1[:, optimizer.gBest_X.astype(bool)], y1, cv=skf).mean()
    table[1, 0] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    
    # score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X1, y1, cv=skf)
    # print(X1.shape[1])
    # print(score.mean())
    # print('==='*16)
    #------------------------------------------------------------------------------
    
    
    start2 = time.time()
    loss2 = functools.partial(fitness, X=X2, y=y2)
    optimizer = BHPSOGWO(fit_func=loss2, 
                         num_dim=X2.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 1]: table[4, 1] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 1]: table[5, 1] = optimizer.gBest_score
    table[3, 1] += optimizer.gBest_score
    table[2, 1] += time.time()-start2
    all_for_std[i, 1] = optimizer.gBest_score
    
    table[0, 1] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X2[:, optimizer.gBest_X.astype(bool)], y2, cv=skf).mean()
    table[1, 1] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start3 = time.time()
    loss3 = functools.partial(fitness, X=X3, y=y3)
    optimizer = BHPSOGWO(fit_func=loss3, 
                         num_dim=X3.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 2]: table[4, 2] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 2]: table[5, 2] = optimizer.gBest_score
    table[3, 2] += optimizer.gBest_score
    table[2, 2] += time.time()-start3
    all_for_std[i, 2] = optimizer.gBest_score
    
    table[0, 2] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X3[:, optimizer.gBest_X.astype(bool)], y3, cv=skf).mean()
    table[1, 2] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start4 = time.time()
    loss4 = functools.partial(fitness, X=X4, y=y4)
    optimizer = BHPSOGWO(fit_func=loss4, 
                         num_dim=X4.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 3]: table[4, 3] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 3]: table[5, 3] = optimizer.gBest_score
    table[3, 3] += optimizer.gBest_score
    table[2, 3] += time.time()-start4
    all_for_std[i, 3] = optimizer.gBest_score
    
    table[0, 3] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X4[:, optimizer.gBest_X.astype(bool)], y4, cv=skf).mean()
    table[1, 3] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start5 = time.time()
    loss5 = functools.partial(fitness, X=X5, y=y5)
    optimizer = BHPSOGWO(fit_func=loss5, 
                         num_dim=X5.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 4]: table[4, 4] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 4]: table[5, 4] = optimizer.gBest_score
    table[3, 4] += optimizer.gBest_score
    table[2, 4] += time.time()-start5
    all_for_std[i, 4] = optimizer.gBest_score
    
    table[0, 4] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X5[:, optimizer.gBest_X.astype(bool)], y5, cv=skf).mean()
    table[1, 4] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start6 = time.time()
    loss6 = functools.partial(fitness, X=X6, y=y6)
    optimizer = BHPSOGWO(fit_func=loss6, 
                         num_dim=X6.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 5]: table[4, 5] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 5]: table[5, 5] = optimizer.gBest_score
    table[3, 5] += optimizer.gBest_score
    table[2, 5] += time.time()-start6
    all_for_std[i, 5] = optimizer.gBest_score
    
    table[0, 5] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X6[:, optimizer.gBest_X.astype(bool)], y6, cv=skf).mean()
    table[1, 5] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start7 = time.time()
    loss7 = functools.partial(fitness, X=X7, y=y7)
    optimizer = BHPSOGWO(fit_func=loss7, 
                         num_dim=X7.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 6]: table[4, 6] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 6]: table[5, 6] = optimizer.gBest_score
    table[3, 6] += optimizer.gBest_score
    table[2, 6] += time.time()-start7
    all_for_std[i, 6] = optimizer.gBest_score
    
    table[0, 6] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X7[:, optimizer.gBest_X.astype(bool)], y7, cv=skf).mean()
    table[1, 6] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start8 = time.time()
    loss8 = functools.partial(fitness, X=X8, y=y8)
    optimizer = BHPSOGWO(fit_func=loss8, 
                         num_dim=X8.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 7]: table[4, 7] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 7]: table[5, 7] = optimizer.gBest_score
    table[3, 7] += optimizer.gBest_score
    table[2, 7] += time.time()-start8
    all_for_std[i, 7] = optimizer.gBest_score
    
    table[0, 7] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X8[:, optimizer.gBest_X.astype(bool)], y8, cv=skf).mean()
    table[1, 7] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start9 = time.time()
    loss9 = functools.partial(fitness, X=X9, y=y9)
    optimizer = BHPSOGWO(fit_func=loss9, 
                         num_dim=X9.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 8]: table[4, 8] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 8]: table[5, 8] = optimizer.gBest_score
    table[3, 8] += optimizer.gBest_score
    table[2, 8] += time.time()-start9
    all_for_std[i, 8] = optimizer.gBest_score
    
    table[0, 8] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X9[:, optimizer.gBest_X.astype(bool)], y9, cv=skf).mean()
    table[1, 8] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start10 = time.time()
    loss10 = functools.partial(fitness, X=X10, y=y10)
    optimizer = BHPSOGWO(fit_func=loss10, 
                         num_dim=X10.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 9]: table[4, 9] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 9]: table[5, 9] = optimizer.gBest_score
    table[3, 9] += optimizer.gBest_score
    table[2, 9] += time.time()-start10
    all_for_std[i, 9] = optimizer.gBest_score
    
    table[0, 9] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X10[:, optimizer.gBest_X.astype(bool)], y10, cv=skf).mean()
    table[1, 9] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start11 = time.time()
    loss11 = functools.partial(fitness, X=X11, y=y11)
    optimizer = BHPSOGWO(fit_func=loss11, 
                         num_dim=X11.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 10]: table[4, 10] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 10]: table[5, 10] = optimizer.gBest_score
    table[3, 10] += optimizer.gBest_score
    table[2, 10] += time.time()-start11
    all_for_std[i, 10] = optimizer.gBest_score
    
    table[0, 10] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X11[:, optimizer.gBest_X.astype(bool)], y11, cv=skf).mean()
    table[1, 10] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start12 = time.time()
    loss12 = functools.partial(fitness, X=X12, y=y12)
    optimizer = BHPSOGWO(fit_func=loss12, 
                         num_dim=X12.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 11]: table[4, 11] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 11]: table[5, 11] = optimizer.gBest_score
    table[3, 11] += optimizer.gBest_score
    table[2, 11] += time.time()-start12
    all_for_std[i, 11] = optimizer.gBest_score
    
    table[0, 11] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X12[:, optimizer.gBest_X.astype(bool)], y12, cv=skf).mean()
    table[1, 11] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start13 = time.time()
    loss13 = functools.partial(fitness, X=X13, y=y13)
    optimizer = BHPSOGWO(fit_func=loss13, 
                         num_dim=X13.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 12]: table[4, 12] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 12]: table[5, 12] = optimizer.gBest_score
    table[3, 12] += optimizer.gBest_score
    table[2, 12] += time.time()-start13
    all_for_std[i, 12] = optimizer.gBest_score
    
    table[0, 12] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X13[:, optimizer.gBest_X.astype(bool)], y13, cv=skf).mean()
    table[1, 12] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start14 = time.time()
    loss14 = functools.partial(fitness, X=X14, y=y14)
    optimizer = BHPSOGWO(fit_func=loss14, 
                         num_dim=X14.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 13]: table[4, 13] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 13]: table[5, 13] = optimizer.gBest_score
    table[3, 13] += optimizer.gBest_score
    table[2, 13] += time.time()-start14
    all_for_std[i, 13] = optimizer.gBest_score
    
    table[0, 13] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X14[:, optimizer.gBest_X.astype(bool)], y14, cv=skf).mean()
    table[1, 13] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start15 = time.time()
    loss15 = functools.partial(fitness, X=X15, y=y15)
    optimizer = BHPSOGWO(fit_func=loss15, 
                         num_dim=X15.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 14]: table[4, 14] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 14]: table[5, 14] = optimizer.gBest_score
    table[3, 14] += optimizer.gBest_score
    table[2, 14] += time.time()-start15
    all_for_std[i, 14] = optimizer.gBest_score
    
    table[0, 14] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X15[:, optimizer.gBest_X.astype(bool)], y15, cv=skf).mean()
    table[1, 14] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start16 = time.time()
    loss16 = functools.partial(fitness, X=X16, y=y16)
    optimizer = BHPSOGWO(fit_func=loss16, 
                         num_dim=X16.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 15]: table[4, 15] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 15]: table[5, 15] = optimizer.gBest_score
    table[3, 15] += optimizer.gBest_score
    table[2, 15] += time.time()-start16
    all_for_std[i, 15] = optimizer.gBest_score
    
    table[0, 15] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X16[:, optimizer.gBest_X.astype(bool)], y16, cv=skf).mean()
    table[1, 15] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start17 = time.time()
    loss17 = functools.partial(fitness, X=X17, y=y17)
    optimizer = BHPSOGWO(fit_func=loss17, 
                         num_dim=X17.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 16]: table[4, 16] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 16]: table[5, 16] = optimizer.gBest_score
    table[3, 16] += optimizer.gBest_score
    table[2, 16] += time.time()-start17
    all_for_std[i, 16] = optimizer.gBest_score
    
    
    table[0, 16] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X17[:, optimizer.gBest_X.astype(bool)], y17, cv=skf).mean()
    table[1, 16] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    start18 = time.time()
    loss18 = functools.partial(fitness, X=X18, y=y18)
    optimizer = BHPSOGWO(fit_func=loss18, 
                         num_dim=X18.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 17]: table[4, 17] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 17]: table[5, 17] = optimizer.gBest_score
    table[3, 17] += optimizer.gBest_score
    table[2, 17] += time.time()-start18
    all_for_std[i, 17] = optimizer.gBest_score
    
    table[0, 17] += cross_val_score(KNeighborsClassifier(n_neighbors=5), X18[:, optimizer.gBest_X.astype(bool)], y18, cv=skf).mean()
    table[1, 17] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    print(i+1, ' ', round(time.time()-total_time, 2), 'sec')
    #------------------------------------------------------------------------------


table[:4, :] = table[:4, :] / times
table[6, :] = np.std(all_for_std, axis=0)
table = pd.DataFrame(table)
table.columns=['Breastcancer', 'BreastEW', 'Congress', 'Exactly', 'Exactly2', 'HeartEW',
               'Ionosphere', 'KrVsKpEW', 'Lymphography', 'M-of-n', 'PenglungEW', 'Sonar', 
                'SpectEW', 'Tic-tac-toe', 'Vote', 'WaveformEW', 'Wine', 'Zoo']
table.index = ['avg acc', '% selected', 'avg time', 'avg loss', 'worst loss', 'best loss', 'std loss']

all_for_std = pd.DataFrame(all_for_std)
all_for_std.columns=['Breastcancer', 'BreastEW', 'Congress', 'Exactly', 'Exactly2', 'HeartEW',
                     'Ionosphere', 'KrVsKpEW', 'Lymphography', 'M-of-n', 'PenglungEW', 'Sonar', 
                     'SpectEW', 'Tic-tac-toe', 'Vote', 'WaveformEW', 'Wine', 'Zoo']