#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint as sp_randint
from model_info import assemble_data, load_y, gen_folds, ModelInfo, save_model
import numpy as np

import pandas as pd
import numpy as np
import time

n_cv = 5

x_train, x_test = assemble_data(['brand', 'model'], base_path='../data/')
y = load_y(base_path='../data/')
print(x_train.shape, y.shape)


kfsplit = gen_folds(
            x_train, y, n_folds=n_cv, random_state=0)


ext = ExtraTreesClassifier(n_estimators=500, n_jobs=4, max_depth=15, criterion='entropy', random_state=1)



param_dist = {"max_depth": sp_randint(5, 30),
              "max_features": [0.1, 0.01, 0.001, 'auto', 'log2'],
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11)
              }

# run randomized search
# 
start = time.time()
n_iter_search = 30
random_search = RandomizedSearchCV(ext, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring='log_loss', cv=n_cv,
                                   verbose=10, random_state=1)
random_search.fit(x_train, y)
print('Best param: ', random_search.best_params_)
print('Best score:', random_search.best_score_)


