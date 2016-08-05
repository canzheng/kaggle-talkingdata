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


du = DataUtil()
du.load_data(sub_sample=False)
x_train, x_test = du.vectorize_x(['brand_code', 'model_code'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("brand_model_train.mtx", x_train)
io.mmwrite("brand_model_test.mtx", x_test)

x_train, x_test = du.vectorize_x(['brand_code', 'model_code', 'label_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("brand_model_label_train.mtx", x_train)
io.mmwrite("brand_model_label_test.mtx", x_test)

x_train, x_test = du.vectorize_x(['brand_code', 'model_code', 'app_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("brand_model_app_train.mtx", x_train)
io.mmwrite("brand_model_app_test.mtx", x_test)

x_train, x_test = du.vectorize_x(['brand_code', 'model_code', 'label_id_bag' 'app_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("brand_model_label_app_train.mtx", x_train)
io.mmwrite("brand_model_label_app_test.mtx", x_test)

y_train = du.get_y_train()
print('y_train shape: ', y_train.shape)

with open('ytrain.pickle', 'wb') as f:
    pickle.dump(y_train, f, protocol=-1, fix_imports=False)
print("done")
