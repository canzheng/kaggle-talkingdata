#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)
#

from scipy import io
import os
import pickle
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from keras.utils import np_utils


# In[53]:

def get_model(input_dim):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(
        Dense(512, input_dim=input_dim, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(12, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


# In[60]:

class KerasWrapper(object):

    def __init__(self, num_features, label='nn_model'):
        self.clf = get_model(num_features)
        self.label = label

    def fit(self, X, y, nb_epoch=10, validation_split=None, validation_data=None, verbose=2, batch_size=32,
            patience=5, save_best=True, return_best=True):
        # update input
        callbacks = []
        if validation_data is not None or validation_split is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=patience)
            callbacks.append(early_stopping)
        if save_best:
            checkpointer = ModelCheckpoint(filepath='{}.hdf5'.format(self.label), verbose=1,
                                           save_best_only=True, save_weights_only=True)
            callbacks.append(checkpointer)
        self.hist = self.clf.fit(X, y, nb_epoch=nb_epoch, validation_split=validation_split,
                                 validation_data=validation_data,
                                 verbose=verbose,
                                 batch_size=batch_size,
                                 callbacks=callbacks)
        val_loss = np.array(self.hist.history['val_loss'])
        if return_best:
            return self, val_loss.min(), val_loss.argmin() + 1
        else:
            return self, val_loss[-1], val_loss.shape[0]

    def predict_proba(self, X):
        pred = self.clf.predict_proba(X, verbose=1)
        return pred


# In[61]:

def cv_score(X, y, n_folds=10, random_state=1999):
    kf = KFold(n_folds, shuffle=True, random_state=random_state)
    scores = np.zeros(n_folds)
    best_epochs = np.zeros(n_folds)
    i_fold = 0
    for itrain, itest in kf.split(X):

        train = X[itrain, :]
        test = X[itest, :]
        ytrain, ytest = y[itrain], y[itest]
        clf = KerasWrapper(
            num_features=train.shape[1], label='keras_{}'.format(i_fold))
        clf, val_loss, num_epoch = clf.fit(train, ytrain, nb_epoch=40,
                                           validation_split=0.2, batch_size=32,
                                           patience=5)

        print('Best Epoch at {}, score: {}'.format(num_epoch, val_loss))
        scores[i_fold] = val_loss
        best_epochs[i_fold] = num_epoch
        i_fold += 1
    return scores, best_epochs


x_train = io.mmread('code/brand_model_label_app_train.mtx')
x_train = x_train.astype(np.byte).toarray()
with open('code/ytrain.pickle', 'rb') as f:
    y_train = pickle.load(f, fix_imports=False)
print('Data loaded')

print('train set shape: ', x_train.shape)
print('y shape: ', y_train.shape)


# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(0.0005)
# x_train = selector.fit_transform(x_train).toarray()
y = np_utils.to_categorical(y_train, y_train.nunique())


scores, best_epochs = cv_score(x_train, y, n_folds=5)

print("Score: mean = {}, std = {}".format(scores.mean(), scores.std()))
print("Num Epochs: mean = {}".format(best_epochs.mean()))
print("Num of epochs: ", best_epochs)


import sms
tpl_value = {'#name#': 'testing model'}
data = sms.tpl_send_sms(tpl_value=tpl_value)
print(data.decode("utf-8", "ignore"))
