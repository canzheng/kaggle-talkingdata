#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_util import DataUtil
from model_util import coarse_to_fine_gs, xgb_estimator_fit
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint as sp_randint

import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
import pickle

du = DataUtil()
du.load_data(sample_rate=0.05)
x_train, x_test = du.vectorize_x(['brand_code', 'model_code', 'label_id_bag'])

# xgb seems have issue detecting number of columns with sparse matrix
#x_train = sp.hstack((x_train, sp.csr_matrix(np.ones((x_train.shape[0],1)))))




print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
y_train = du.get_y_train()
print('y_train shape: ', y_train.shape)

n_cv = 5

rf1 = RandomForestClassifier(n_estimators=500, n_jobs=4, max_depth=15, criterion='gini', random_state=1)



param_dist = {"max_depth": sp_randint(5, 50),
              "max_features": [0.1, 0.01, 0.001, 'auto', 'log2'],
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False]              
              }

# run randomized search
# 
start = time.time()
n_iter_search = 20
random_search = RandomizedSearchCV(rf1, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring='log_loss', cv=n_cv,
                                   verbose=10, random_state=1)
random_search.fit(x_train, y_train)
print('Best param: ', random_search.best_params_)
print('Best score:', random_search.best_score_)
with open('rf_cv.pickle', 'wb') as f: 
	pickle.dump(random_search, f)
print('Time: ', (time.time() -start) / 60)


