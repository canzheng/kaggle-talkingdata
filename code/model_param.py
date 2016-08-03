#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_util import DataUtil
from model_util import coarse_to_fine_gs, xgb_estimator_fit
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform as sp_uniform

import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
import pickle
import argparse


def model_param_search(estimator, X, y, param_dist, scoring,
                       n_iter=1, n_cv=5, verbose=10, random_state=1, model_id='model', save_search=True):
    start = time.time()

    random_search = RandomizedSearchCV(estimator, param_distributions=param_dist,
                                       n_iter=n_iter, scoring=scoring, cv=n_cv,
                                       verbose=verbose, random_state=random_state)
    random_search.fit(X, y)
    print('Best param: ', random_search.best_params_)
    print('Best score: ', random_search.best_score_)
    print('Best model: ', random_search.best_estimator_)
    if save_search:
        with open(model_id+'.pickle', 'wb') as f:
            pickle.dump(random_search, f)
    print('Time searching param for {}: {}'.format(
        model_id, (time.time() - start) / 60))

    return random_search.best_estimator_


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', help='number of cv folds', type=int, default=5)
    parser.add_argument(
        '--iter', help='number of iterations', type=int, default=10)
    parser.add_argument(
        '--sample', help='subsample (default -1 means no subsampling)', type=float, default=-1)
    parser.add_argument(
        '--varthresh', help='variance thresh (default 0 means take all)', type=float, default=0)

    args = parser.parse_args()

    rand_state = 1

    n_cv = args.cv
    n_iter_search = args.iter
    sample_rate = args.sample
    sub_sample = (False if sample_rate < 0 else True)
    var_thresh = args.varthresh
    scoring = 'log_loss'
    verbose = 10

    du = DataUtil()
    du.load_data(sub_sample=sub_sample, sample_rate=sample_rate)

    x_train, x_test = du.vectorize_x(
        ['brand_code', 'model_code', 'label_id_bag'], variance_thresh=var_thresh)
    print('train set shape: ', x_train.shape)
    print('test set shape: ', x_test.shape)

    # xgb seems have issue detecting number of columns with sparse matrix
    x_train_xgb = sp.hstack(
        (x_train, sp.csr_matrix(np.ones((x_train.shape[0], 1)))))
    print(
        'patching train data with non-zero column to get around xgb sparse issue')

    y_train = du.get_y_train()
    print('y_train shape: ', y_train.shape)

    rf1 = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, max_depth=15, criterion='gini', random_state=rand_state)
    rf2 = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, max_depth=15, criterion='entropy', random_state=rand_state)
    xgb1 = XGBClassifier(
        objective='multi:softprob',
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=10,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        nthread=-1,
        scale_pos_weight=1)
    ext1 = ExtraTreesClassifier(n_estimators=500, max_depth=15, criterion='gini', max_features=0.6,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                n_jobs=-1, random_state=1999)
    ext2 = ExtraTreesClassifier(n_estimators=500, max_depth=15, criterion='entropy', max_features=0.6,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                n_jobs=-1, random_state=1999)

    param_dist = {"max_depth": sp_randint(5, 50),
                  "max_features": [0.1, 0.01, 0.001, 'auto', 'log2'],
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False]
                  }
    rf1 = model_param_search(
        rf1, x_train, y_train, param_dist, scoring, n_iter_search, n_cv, verbose, model_id='rf1')
    rf2 = model_param_search(
        rf2, x_train, y_train, param_dist, scoring, n_iter_search, n_cv, verbose, model_id='rf2')
    ext1 = model_param_search(
        rf1, x_train, y_train, param_dist, scoring, n_iter_search, n_cv, verbose, model_id='ext1')
    ext2 = model_param_search(
        rf2, x_train, y_train, param_dist, scoring, n_iter_search, n_cv, verbose, model_id='ext2')

    xgb_estimator_fit(xgb1, x_train_xgb, y_train, 'mlogloss',
                      useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)

    param_dist = {"max_depth": sp_randint(10, 40),
                  "min_child_weight": sp_randint(1, 20),
                  "subsample": sp_uniform(0, 1),
                  "colsample_bytree": sp_uniform(0, 1),
                  "gamma": [i/10.0 for i in range(0, 5)]
                  }

    xgb1 = model_param_search(
        xgb1, x_train_xgb, y_train, param_dist, scoring, n_iter_search, n_cv, verbose, model_id='xgb1')

    xgb_estimator_fit(xgb1, x_train_xgb, y_train, 'mlogloss',
                      useTrainCV=True, cv_folds=n_cv, early_stopping_rounds=50)
