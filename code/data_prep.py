#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_load import DataLoader

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


du = DataLoader()
du.load_data(sub_sample=False)
x_train, x_test = du.vectorize_x(['brand_code'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("train_brand.mtx", x_train)
io.mmwrite("test_brand.mtx", x_test)

x_train, x_test = du.vectorize_x(['model_code'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("train_model.mtx", x_train)
io.mmwrite("test_model.mtx", x_test)


x_train, x_test = du.vectorize_x(['label_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("train_label.mtx", x_train)
io.mmwrite("test_label.mtx", x_test)

x_train, x_test = du.vectorize_x(['app_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite("train_appid.mtx", x_train)
io.mmwrite("test_appid.mtx", x_test)


#y_train = du.get_y_train()
#print('y_train shape: ', y_train.shape)

#with open('ytrain.pickle', 'wb') as f:
#    pickle.dump(y_train, f, protocol=-1, fix_imports=False)
print("done")
