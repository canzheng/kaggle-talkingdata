#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from model_util import coarse_to_fine_gs, xgb_estimator_fit
from xgboost.sklearn import XGBClassifier
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import pickle
from model_info import assemble_data, load_y, gen_folds, ModelInfo, save_model, load_evind
import numpy as np
n_cv = 5
#
x_train, x_test = assemble_data(['brand', 'model', 'label', 'appid'])
y = load_y()
evind = load_evind()

x_train = x_train[evind].astype(np.float32).toarray()
y = y[evind]
print(x_train.shape, y.shape)
#
#
kfsplit = gen_folds(
            x_train, y, n_folds=n_cv, random_state=0)
#
#
xgb1 = XGBClassifier(
    objective= 'multi:softprob',
    learning_rate =0.1,
    n_estimators=300,
    max_depth=14,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=0.05,
    nthread=4,
    scale_pos_weight=1)


xgb_estimator_fit(xgb1, x_train, y, 'mlogloss', useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)
print(xgb1.get_params())

#data_parts=['brand', 'model', 'label', 'appid']
    
#m = ModelInfo(clf=xgb1,
#                  data_parts=data_parts, label='xgb_{}'.format('-'.join(data_parts)), mean_func=np.mean)
#m.run(ignore_no_events=True)
#save_model(m)

#param_dist = {"max_depth": sp_randint(10, 30),
#              "min_child_weight": sp_randint(1, 10),
#              "subsample": sp_uniform(0.4,0.6),
#              "colsample_bytree": sp_uniform(0.4,0.6),
#              "gamma":[0, 0.1, 0.2]
#              }
#
## run randomized search
# 
#start = time.time()
#n_iter_search = 20
#random_search = RandomizedSearchCV(xgb1, param_distributions=param_dist, cv=kfsplit,
#                                   n_iter=n_iter_search, scoring='log_loss',
#                                   verbose=10, random_state=1)
#random_search.fit(x_train, y)
#print('Best param: ', random_search.best_params_)
#print('Best score:', random_search.best_score_)
#
#
#with open('xgb_cv.pickle', 'wb') as f: 
#	pickle.dump(random_search, f)
#
#xgb1 = random_search.best_estimator_
#xgb1.set_params({'learning_rate': 0.005, 'n_estimators': 10000})
#print(xgb1)
#
#xgb_estimator_fit(xgb1, x_train, y_train, 'mlogloss', useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)
#
#print('final model: ', xgb1)
#
#print('Time: ', (time.time() -start) / 60)
#