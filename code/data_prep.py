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
base_dir='data/'

du = DataLoader()
du.load_data(sub_sample=False)
x_train, x_test = du.vectorize_x(['brand_code'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite(base_dir + "train_brand.mtx", x_train)
io.mmwrite(base_dir + "test_brand.mtx", x_test)

x_train, x_test = du.vectorize_x(['model_code'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite(base_dir + "train_model.mtx", x_train)
io.mmwrite(base_dir + "test_model.mtx", x_test)


x_train, x_test = du.vectorize_x(['label_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite(base_dir + "train_label.mtx", x_train)
io.mmwrite(base_dir + "test_label.mtx", x_test)

x_train, x_test = du.vectorize_x(['app_id_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite(base_dir + "train_appid.mtx", x_train)
io.mmwrite(base_dir + "test_appid.mtx", x_test)

x_train, x_test = du.vectorize_EX(['category_bag'])
print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)
io.mmwrite(base_dir + "train_term.mtx", x_train)
io.mmwrite(base_dir + "test_term.mtx", x_test)


y_train = du.get_y_train()
print('y_train shape: ', y_train.shape)

with open(base_dir + 'ytrain.pickle', 'wb') as f:
    pickle.dump(y_train, f, protocol=-1, fix_imports=False)


ga_train = pd.read_csv('gender_age_train.csv')
ga_test = pd.read_csv('gender_age_test.csv')
ga_ev = pd.read_csv('events.csv')
ga_did = ga_ev.device_id.unique()
ind_withev = ga_train['device_id'].isin(ga_did).values
ind_withev_test = ga_test['device_id'].isin(ga_did).values

with open(base_dir + 'ind_withev_train.pickle', 'wb') as f:
    pickle.dump(ind_withev, f, protocol=-1, fix_imports=False)
with open(base_dir + 'ind_withev_test.pickle', 'wb') as f:
    pickle.dump(ind_withev_test, f, protocol=-1, fix_imports=False)


print("done")
