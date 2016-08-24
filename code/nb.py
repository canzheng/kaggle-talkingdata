#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_util import DataUtil
from model_util import coarse_to_fine_gs, xgb_estimator_fit
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score

from scipy.stats import randint as sp_randint

import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
import pickle

from scipy import io


#du = DataUtil()
#du.load_data(sub_sample=False)
#x_train = du.vectorize_x(['brand_code', 'model_code'], train_only=True)


x_train = io.mmread("brand_model_train.mtx")
print('train set shape: ', x_train.shape)


#y_train = du.get_y_train()

with open('ytrain.pickle', 'rb') as f:
    y_train = pickle.load(f, fix_imports=False)
print('y_train shape: ', y_train.shape)


n_cv = 5


nb = BernoulliNB(alpha=1, binarize=None, fit_prior=True, class_prior=None)

scores = cross_val_score(nb, x_train, y_train, scoring='log_loss', cv=n_cv, n_jobs=-1)
print(scores)
