#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_util import DataUtil
from model_util import coarse_to_fine_gs, xgb_estimator_fit
from xgboost.sklearn import XGBClassifier
import scipy.sparse as sp
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import pickle

du = DataUtil()
du.load_data(sample_rate=0.05)
x_train, x_test = du.vectorize_x(['brand_code', 'model_code', 'label_id_bag'])

# xgb seems have issue detecting number of columns with sparse matrix
x_train = sp.hstack((x_train, sp.csr_matrix(np.ones((x_train.shape[0],1)))))


print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
y_train = du.get_y_train()
print('y_train shape: ', y_train.shape)

n_cv = 5

xgb1 = XGBClassifier(
    objective= 'multi:softprob',
    learning_rate =0.1,
    n_estimators=500,
    max_depth=5,
    min_child_weight=10,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    nthread=4,
    scale_pos_weight=1)

xgb_estimator_fit(xgb1, x_train, y_train, 'mlogloss', useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)




param_dist = {"max_depth": sp_randint(10, 40),
              "min_child_weight": sp_randint(1, 20),
              "subsample": sp_uniform(0,1),
              "colsample_bytree": sp_uniform(0,1),
              "gamma":[i/10.0 for i in range(0,5)]
              }

# run randomized search
# 
start = time.time()
n_iter_search = 20
random_search = RandomizedSearchCV(xgb1, param_distributions=param_dist, cv=n_cv,
                                   n_iter=n_iter_search, scoring='log_loss',
                                   verbose=10, random_state=1)
random_search.fit(x_train, y_train)
print('Best param: ', random_search.best_params_)
print('Best score:', random_search.best_score_)
with open('xgb_cv.pickle', 'wb') as f: 
	pickle.dump(random_search, f)

xgb1 = random_search.best_estimator_
xgb1.set_params({'learning_rate': 0.005, 'n_estimators': 10000})
print(xgb1)

xgb_estimator_fit(xgb1, x_train, y_train, 'mlogloss', useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)

print('final model: ', xgb1)

print('Time: ', (time.time() -start) / 60)
