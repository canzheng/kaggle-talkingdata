#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)
#
#

import argparse
import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import pickle

base_dir = 'nnet/'


def cv_by_epoch(X, y, model_params, cv, batch_size=64, nb_epoch=15, patience=3, label=None, verbose=0):
    from keras.utils import np_utils
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    from time import strftime

    y_keras = np_utils.to_categorical(y)
    evind = load_evind()
    n_folds = len(cv)
    val_score = np.zeros(n_folds)
    # monitor scores for devices with / without events
    val_score_ev = np.zeros(n_folds)
    val_score_noev = np.zeros(n_folds)
    hist_list = []
    clfs = [get_model(**model_params) for i in range(n_folds)]
    # iteratively train epochs

    i_fold = 0
    for itrain, itest in cv:
        clf = clfs[i_fold]
        test_ind_ev = evind[itest]
        test_ind_noev = np.logical_not(test_ind_ev)

        print('Fold ', i_fold)
        train, test = X[itrain, :], X[itest, :]
        ytrain_keras, ytest_keras = y_keras[itrain, :], y_keras[itest, :]

        ytest = y[itest]

        if label is None:
            label = strftime('%Y%m%d%H%M%S')
        model_path = '{}{}_{}.h5'.format(base_dir, label, i_fold)

        early_stopping = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(
            model_path,
            save_best_only=True)

        hist = clf.fit(train, ytrain_keras,
                       validation_data=(test, ytest_keras),
                       nb_epoch=nb_epoch,
                       batch_size=batch_size,
                       verbose=verbose,
                       callbacks=[early_stopping, checkpoint])
        hist_list.append(hist.history)
        # predict on oof
        # loading model for best epoch
        clf = load_model(model_path)
        pred = clf.predict_proba(test, verbose=verbose)

        val_score[i_fold] = log_loss(ytest, pred)
        val_score_ev[i_fold] = log_loss(ytest[test_ind_ev], pred[test_ind_ev])
        val_score_noev[i_fold] = log_loss(
            ytest[test_ind_noev], pred[test_ind_noev])
        print('Validation score: ', val_score[i_fold])
        print(
            'Val-score w/ ev: {}, w/o ev: {}'.format(val_score_ev[i_fold], val_score_noev[i_fold]))
        i_fold += 1

    print("Val score", val_score)
    print("Val score w/ ev", val_score_ev)
    print("Val score w/o ev average", val_score_noev)

    return val_score, val_score_ev, val_score_noev, hist_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='random seed', type=int, default=1337)
    parser.add_argument(
        '--epochs', help='epochs to train', type=int, default=5)
    parser.add_argument('--cv', help='number of folds', type=int, default=5)

    args = parser.parse_args()

    # to control seed for theano
    np.random.seed(args.seed)

    from keras.utils import np_utils
    from keras_wrapper import get_model, FlexInputKerasClassifier
    from model_info import load_evind, assemble_data, gen_folds, assemble_stack, load_y

#   data = [
#       ['brand', 'model'],
#       ['brand', 'model', 'appid'],
#       ['brand', 'model', 'appfreq'],
#       ['brand', 'model', 'appcount'],
#       ['brand', 'model', 'apppopcount'],
#       ['brand', 'model', 'appid', 'appcount'],
#       ['brand', 'model', 'appid', 'apppopcount'],
#       ['brand', 'model', 'appfreq', 'appcount'],
#       ['brand', 'model', 'appfreq', 'apppopcount'],
#       ['brand', 'model', 'label'],
#       ['brand', 'model', 'label', 'appid'],
#       ['brand', 'model', 'label', 'appfreq'],
#       ['brand', 'model', 'label', 'appcount'],
#       ['brand', 'model', 'label', 'apppopcount'],
#       ['brand', 'model', 'label', 'appid', 'appcount'],
#       ['brand', 'model', 'label', 'appid', 'apppopcount'],
#       ['brand', 'model', 'label', 'appfreq', 'appcount'],
#       ['brand', 'model', 'label', 'appfreq', 'apppopcount'],
#       ['brand', 'model', 'term'],
#       ['brand', 'model', 'term', 'appid'],
#       ['brand', 'model', 'term', 'appfreq'],
#       ['brand', 'model', 'term', 'appcount'],
#       ['brand', 'model', 'term', 'apppopcount'],
#       ['brand', 'model', 'term', 'appid', 'appcount'],
#       ['brand', 'model', 'term', 'appid', 'apppopcount'],
#       ['brand', 'model', 'term', 'appfreq', 'appcount'],
#       ['brand', 'model', 'term', 'appfreq', 'apppopcount']
#   ]


#   data = [
#       ['brand', 'model', 'label', 'appid', 'appcount'],
#       ['brand', 'model', 'label', 'appid', 'appcount', 'apppopcount'],
#       ['brand', 'model', 'term', 'appid'],
#       ['brand', 'model', 'label', 'appfreq'],
#       ['brand', 'model', 'term', 'appfreq']
#   ]
    data = [['brand', 'model', 'label', 'appid']]
    n_neurons = 512
    reg = 0
    drop = 0.5
    cv_result = []
    for parts in data:
        # load X based on input
        x_train, x_test = assemble_data(parts)

        # load y and transform to categorical
        y = load_y()
        # stratify by y + evind
        evind = load_evind()
        y_strat = np.core.defchararray.add(y.astype(str), evind.astype(str))
        kfsplit = gen_folds(
            x_train, y_strat, n_folds=args.cv, random_state=args.seed)

        # prepare params
        params = {'l1_neurons': n_neurons,
                  'l2_neurons': n_neurons,
                  'reg': reg,
                  'dropout': drop,
                  'input_dim': x_train.shape[1]}

        # cv by epoch
        val_score, val_score_ev, val_score_noev, hist_list = cv_by_epoch(x_train.toarray(),
                                                                         y, model_params=params, cv=kfsplit, nb_epoch=args.epochs)


        best_ep = [h['val_loss'].argmin() + 1 for h in hist_list]

        r = {
            'parts': parts,
            'neurons': n_neurons,
            'l2_reg': reg,
            'dropout': drop,
            'val_score': val_score,
            'val_score_ev': val_score_ev,
            'val_score_noev': val_score_noev,
            'best_ep': best_ep
        }
        cv_result.append(r)
        import pprint
        pprint.pprint(r)

        pprint.pprint(hist_list)


#    with open('model/cv_nn_{}_{:.4f}_{:.2f}.pickle'.format(int(neurons), reg, drop), 'wb') as f:
#        pickle.dump(cv_result, f, protocol=-1, fix_imports=False)#

#   import sms
#   tpl_value = {'#name#': 'nn_cv'}
#   data = sms.tpl_send_sms(tpl_value=tpl_value)
#   print(data.decode("utf-8", "ignore"))
