# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from BHPSOGWO import BHPSOGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
#------------------------------------------------------------------------------


def fitness(x, X_train, y_train, X_test, y_test):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, x[i, :].astype(bool)], y_train)
            score = accuracy_score(knn.predict(X_test[:, x[i, :].astype(bool)]), y_test)
            loss[i] = 0.99*(1-score) + 0.01*(np.sum(x[i, :])/X_train.shape[1])
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
#------------------------------------------------------------------------------


for i in range(times):
    total_time = time.time()
    #------------------------------------------------------------------------------
    

    X_train1, X_test1, y_train1, y_test1 = train_test_split(Breastcancer[:, :-1], Breastcancer[:, -1], stratify=Breastcancer[:, -1], test_size=0.5)
    start1 = time.time()
    loss1 = functools.partial(fitness, X_train=X_train1, y_train=y_train1, X_test=X_test1, y_test=y_test1)
    optimizer = BHPSOGWO(fit_func=loss1, 
                         num_dim=X_train1.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 0]: table[4, 0] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 0]: table[5, 0] = optimizer.gBest_score
    table[3, 0] += optimizer.gBest_score
    table[2, 0] += time.time()-start1
    all_for_std[i, 0] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train1[:, optimizer.gBest_X.astype(bool)], y_train1)
    table[0, 0] += accuracy_score(knn.predict(X_test1[:, optimizer.gBest_X.astype(bool)]), y_test1)
    table[1, 0] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    
    # knn = KNeighborsClassifier(n_neighbors=5).fit(X_train1, y_train1)
    # print(X_train1.shape[1])
    # print(round(accuracy_score(knn.predict(X_test1), y_test1), 5))
    # print('==='*16)
    #------------------------------------------------------------------------------
    

    X_train2, X_test2, y_train2, y_test2 = train_test_split(BreastEW[:, :-1], BreastEW[:, -1], stratify=BreastEW[:, -1], test_size=0.5)
    start2 = time.time()
    loss2 = functools.partial(fitness, X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2)
    optimizer = BHPSOGWO(fit_func=loss2, 
                         num_dim=X_train2.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 1]: table[4, 1] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 1]: table[5, 1] = optimizer.gBest_score
    table[3, 1] += optimizer.gBest_score
    table[2, 1] += time.time()-start2
    all_for_std[i, 1] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train2[:, optimizer.gBest_X.astype(bool)], y_train2)
    table[0, 1] += accuracy_score(knn.predict(X_test2[:, optimizer.gBest_X.astype(bool)]), y_test2)
    table[1, 1] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train3, X_test3, y_train3, y_test3 = train_test_split(Congress[:, :-1], Congress[:, -1], stratify=Congress[:, -1], test_size=0.5)
    start3 = time.time()
    loss3 = functools.partial(fitness, X_train=X_train3, y_train=y_train3, X_test=X_test3, y_test=y_test3)
    optimizer = BHPSOGWO(fit_func=loss3, 
                         num_dim=X_train3.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 2]: table[4, 2] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 2]: table[5, 2] = optimizer.gBest_score
    table[3, 2] += optimizer.gBest_score
    table[2, 2] += time.time()-start3
    all_for_std[i, 2] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train3[:, optimizer.gBest_X.astype(bool)], y_train3)
    table[0, 2] += accuracy_score(knn.predict(X_test3[:, optimizer.gBest_X.astype(bool)]), y_test3)
    table[1, 2] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train4, X_test4, y_train4, y_test4 = train_test_split(Exactly[:, :-1], Exactly[:, -1], stratify=Exactly[:, -1], test_size=0.5)
    start4 = time.time()
    loss4 = functools.partial(fitness, X_train=X_train4, y_train=y_train4, X_test=X_test4, y_test=y_test4)
    optimizer = BHPSOGWO(fit_func=loss4, 
                         num_dim=X_train4.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 3]: table[4, 3] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 3]: table[5, 3] = optimizer.gBest_score
    table[3, 3] += optimizer.gBest_score
    table[2, 3] += time.time()-start4
    all_for_std[i, 3] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train4[:, optimizer.gBest_X.astype(bool)], y_train4)
    table[0, 3] += accuracy_score(knn.predict(X_test4[:, optimizer.gBest_X.astype(bool)]), y_test4)
    table[1, 3] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train5, X_test5, y_train5, y_test5 = train_test_split(Exactly2[:, :-1], Exactly2[:, -1], stratify=Exactly2[:, -1], test_size=0.5)
    start5 = time.time()
    loss5 = functools.partial(fitness, X_train=X_train5, y_train=y_train5, X_test=X_test5, y_test=y_test5)
    optimizer = BHPSOGWO(fit_func=loss5, 
                         num_dim=X_train5.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 4]: table[4, 4] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 4]: table[5, 4] = optimizer.gBest_score
    table[3, 4] += optimizer.gBest_score
    table[2, 4] += time.time()-start5
    all_for_std[i, 4] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train5[:, optimizer.gBest_X.astype(bool)], y_train5)
    table[0, 4] += accuracy_score(knn.predict(X_test5[:, optimizer.gBest_X.astype(bool)]), y_test5)
    table[1, 4] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train6, X_test6, y_train6, y_test6 = train_test_split(HeartEW[:, :-1], HeartEW[:, -1], stratify=HeartEW[:, -1], test_size=0.5)
    start6 = time.time()
    loss6 = functools.partial(fitness, X_train=X_train6, y_train=y_train6, X_test=X_test6, y_test=y_test6)
    optimizer = BHPSOGWO(fit_func=loss6, 
                         num_dim=X_train6.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 5]: table[4, 5] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 5]: table[5, 5] = optimizer.gBest_score
    table[3, 5] += optimizer.gBest_score
    table[2, 5] += time.time()-start6
    all_for_std[i, 5] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train6[:, optimizer.gBest_X.astype(bool)], y_train6)
    table[0, 5] += accuracy_score(knn.predict(X_test6[:, optimizer.gBest_X.astype(bool)]), y_test6)
    table[1, 5] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train7, X_test7, y_train7, y_test7 = train_test_split(Ionosphere[:, :-1], Ionosphere[:, -1], stratify=Ionosphere[:, -1], test_size=0.5)
    start7 = time.time()
    loss7 = functools.partial(fitness, X_train=X_train7, y_train=y_train7, X_test=X_test7, y_test=y_test7)
    optimizer = BHPSOGWO(fit_func=loss7, 
                         num_dim=X_train7.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 6]: table[4, 6] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 6]: table[5, 6] = optimizer.gBest_score
    table[3, 6] += optimizer.gBest_score
    table[2, 6] += time.time()-start7
    all_for_std[i, 6] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train7[:, optimizer.gBest_X.astype(bool)], y_train7)
    table[0, 6] += accuracy_score(knn.predict(X_test7[:, optimizer.gBest_X.astype(bool)]), y_test7)
    table[1, 6] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train8, X_test8, y_train8, y_test8 = train_test_split(KrVsKpEW[:, :-1], KrVsKpEW[:, -1], stratify=KrVsKpEW[:, -1], test_size=0.5)
    start8 = time.time()
    loss8 = functools.partial(fitness, X_train=X_train8, y_train=y_train8, X_test=X_test8, y_test=y_test8)
    optimizer = BHPSOGWO(fit_func=loss8, 
                         num_dim=X_train8.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 7]: table[4, 7] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 7]: table[5, 7] = optimizer.gBest_score
    table[3, 7] += optimizer.gBest_score
    table[2, 7] += time.time()-start8
    all_for_std[i, 7] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train8[:, optimizer.gBest_X.astype(bool)], y_train8)
    table[0, 7] += accuracy_score(knn.predict(X_test8[:, optimizer.gBest_X.astype(bool)]), y_test8)
    table[1, 7] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train9, X_test9, y_train9, y_test9 = train_test_split(Lymphography[:, :-1], Lymphography[:, -1], stratify=Lymphography[:, -1], test_size=0.5)
    start9 = time.time()
    loss9 = functools.partial(fitness, X_train=X_train9, y_train=y_train9, X_test=X_test9, y_test=y_test9)
    optimizer = BHPSOGWO(fit_func=loss9, 
                         num_dim=X_train9.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 8]: table[4, 8] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 8]: table[5, 8] = optimizer.gBest_score
    table[3, 8] += optimizer.gBest_score
    table[2, 8] += time.time()-start9
    all_for_std[i, 8] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train9[:, optimizer.gBest_X.astype(bool)], y_train9)
    table[0, 8] += accuracy_score(knn.predict(X_test9[:, optimizer.gBest_X.astype(bool)]), y_test9)
    table[1, 8] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train10, X_test10, y_train10, y_test10 = train_test_split(M_of_n[:, :-1], M_of_n[:, -1], stratify=M_of_n[:, -1], test_size=0.5)
    start10 = time.time()
    loss10 = functools.partial(fitness, X_train=X_train10, y_train=y_train10, X_test=X_test10, y_test=y_test10)
    optimizer = BHPSOGWO(fit_func=loss10, 
                         num_dim=X_train10.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 9]: table[4, 9] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 9]: table[5, 9] = optimizer.gBest_score
    table[3, 9] += optimizer.gBest_score
    table[2, 9] += time.time()-start10
    all_for_std[i, 9] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train10[:, optimizer.gBest_X.astype(bool)], y_train10)
    table[0, 9] += accuracy_score(knn.predict(X_test10[:, optimizer.gBest_X.astype(bool)]), y_test10)
    table[1, 9] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train11, X_test11, y_train11, y_test11 = train_test_split(PenglungEW[:, :-1], PenglungEW[:, -1], stratify=PenglungEW[:, -1], test_size=0.5)
    start11 = time.time()
    loss11 = functools.partial(fitness, X_train=X_train11, y_train=y_train11, X_test=X_test11, y_test=y_test11)
    optimizer = BHPSOGWO(fit_func=loss11, 
                         num_dim=X_train11.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 10]: table[4, 10] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 10]: table[5, 10] = optimizer.gBest_score
    table[3, 10] += optimizer.gBest_score
    table[2, 10] += time.time()-start11
    all_for_std[i, 10] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train11[:, optimizer.gBest_X.astype(bool)], y_train11)
    table[0, 10] += accuracy_score(knn.predict(X_test11[:, optimizer.gBest_X.astype(bool)]), y_test11)
    table[1, 10] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train12, X_test12, y_train12, y_test12 = train_test_split(Sonar[:, :-1], Sonar[:, -1], stratify=Sonar[:, -1], test_size=0.5)
    start12 = time.time()
    loss12 = functools.partial(fitness, X_train=X_train12, y_train=y_train12, X_test=X_test12, y_test=y_test12)
    optimizer = BHPSOGWO(fit_func=loss12, 
                         num_dim=X_train12.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 11]: table[4, 11] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 11]: table[5, 11] = optimizer.gBest_score
    table[3, 11] += optimizer.gBest_score
    table[2, 11] += time.time()-start12
    all_for_std[i, 11] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train12[:, optimizer.gBest_X.astype(bool)], y_train12)
    table[0, 11] += accuracy_score(knn.predict(X_test12[:, optimizer.gBest_X.astype(bool)]), y_test12)
    table[1, 11] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    

    X_train13, X_test13, y_train13, y_test13 = train_test_split(SpectEW[:, :-1], SpectEW[:, -1], stratify=SpectEW[:, -1], test_size=0.5)
    start13 = time.time()
    loss13 = functools.partial(fitness, X_train=X_train13, y_train=y_train13, X_test=X_test13, y_test=y_test13)
    optimizer = BHPSOGWO(fit_func=loss13, 
                         num_dim=X_train13.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 12]: table[4, 12] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 12]: table[5, 12] = optimizer.gBest_score
    table[3, 12] += optimizer.gBest_score
    table[2, 12] += time.time()-start13
    all_for_std[i, 12] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train13[:, optimizer.gBest_X.astype(bool)], y_train13)
    table[0, 12] += accuracy_score(knn.predict(X_test13[:, optimizer.gBest_X.astype(bool)]), y_test13)
    table[1, 12] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train14, X_test14, y_train14, y_test14 = train_test_split(Tic_tac_toe[:, :-1], Tic_tac_toe[:, -1], stratify=Tic_tac_toe[:, -1], test_size=0.5)
    start14 = time.time()
    loss14 = functools.partial(fitness, X_train=X_train14, y_train=y_train14, X_test=X_test14, y_test=y_test14)
    optimizer = BHPSOGWO(fit_func=loss14, 
                         num_dim=X_train14.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 13]: table[4, 13] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 13]: table[5, 13] = optimizer.gBest_score
    table[3, 13] += optimizer.gBest_score
    table[2, 13] += time.time()-start14
    all_for_std[i, 13] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train14[:, optimizer.gBest_X.astype(bool)], y_train14)
    table[0, 13] += accuracy_score(knn.predict(X_test14[:, optimizer.gBest_X.astype(bool)]), y_test14)
    table[1, 13] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train15, X_test15, y_train15, y_test15 = train_test_split(Vote[:, :-1], Vote[:, -1], stratify=Vote[:, -1], test_size=0.5)
    start15 = time.time()
    loss15 = functools.partial(fitness, X_train=X_train15, y_train=y_train15, X_test=X_test15, y_test=y_test15)
    optimizer = BHPSOGWO(fit_func=loss15, 
                         num_dim=X_train15.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 14]: table[4, 14] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 14]: table[5, 14] = optimizer.gBest_score
    table[3, 14] += optimizer.gBest_score
    table[2, 14] += time.time()-start15
    all_for_std[i, 14] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train15[:, optimizer.gBest_X.astype(bool)], y_train15)
    table[0, 14] += accuracy_score(knn.predict(X_test15[:, optimizer.gBest_X.astype(bool)]), y_test15)
    table[1, 14] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train16, X_test16, y_train16, y_test16 = train_test_split(WaveformEW[:, :-1], WaveformEW[:, -1], stratify=WaveformEW[:, -1], test_size=0.5)
    start16 = time.time()
    loss16 = functools.partial(fitness, X_train=X_train16, y_train=y_train16, X_test=X_test16, y_test=y_test16)
    optimizer = BHPSOGWO(fit_func=loss16, 
                         num_dim=X_train16.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 15]: table[4, 15] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 15]: table[5, 15] = optimizer.gBest_score
    table[3, 15] += optimizer.gBest_score
    table[2, 15] += time.time()-start16
    all_for_std[i, 15] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train16[:, optimizer.gBest_X.astype(bool)], y_train16)
    table[0, 15] += accuracy_score(knn.predict(X_test16[:, optimizer.gBest_X.astype(bool)]), y_test16)
    table[1, 15] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train17, X_test17, y_train17, y_test17 = train_test_split(Wine[:, :-1], Wine[:, -1], stratify=Wine[:, -1], test_size=0.5)
    start17 = time.time()
    loss17 = functools.partial(fitness, X_train=X_train17, y_train=y_train17, X_test=X_test17, y_test=y_test17)
    optimizer = BHPSOGWO(fit_func=loss17, 
                         num_dim=X_train17.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 16]: table[4, 16] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 16]: table[5, 16] = optimizer.gBest_score
    table[3, 16] += optimizer.gBest_score
    table[2, 16] += time.time()-start17
    all_for_std[i, 16] = optimizer.gBest_score
    
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train17[:, optimizer.gBest_X.astype(bool)], y_train17)
    table[0, 16] += accuracy_score(knn.predict(X_test17[:, optimizer.gBest_X.astype(bool)]), y_test17)
    table[1, 16] += np.sum(optimizer.gBest_X)/len(optimizer.gBest_X)
    #------------------------------------------------------------------------------
    
    
    X_train18, X_test18, y_train18, y_test18 = train_test_split(Zoo[:, :-1], Zoo[:, -1], stratify=Zoo[:, -1], test_size=0.5)
    start18 = time.time()
    loss18 = functools.partial(fitness, X_train=X_train18, y_train=y_train18, X_test=X_test18, y_test=y_test18)
    optimizer = BHPSOGWO(fit_func=loss18, 
                         num_dim=X_train18.shape[1], num_particle=p, max_iter=g)
    optimizer.opt()
    
    if optimizer.gBest_score>table[4, 17]: table[4, 17] = optimizer.gBest_score
    if optimizer.gBest_score<table[5, 17]: table[5, 17] = optimizer.gBest_score
    table[3, 17] += optimizer.gBest_score
    table[2, 17] += time.time()-start18
    all_for_std[i, 17] = optimizer.gBest_score
    
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train18[:, optimizer.gBest_X.astype(bool)], y_train18)
    table[0, 17] += accuracy_score(knn.predict(X_test18[:, optimizer.gBest_X.astype(bool)]), y_test18)
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