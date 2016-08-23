#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)

from keras.wrappers.scikit_learn import KerasClassifier
from scipy.sparse import issparse
import numpy as np

def get_model(input_dim=100, l1_neurons=512, l2_neurons=512,
              reg=0.001,
              dropout=0.4):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.regularizers import l2, activity_l2

    model = Sequential()
    model.add(
        Dense(l1_neurons, input_dim=input_dim, activation='relu', W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Dense(l2_neurons, activation='relu', W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


class FlexInputKerasClassifier(KerasClassifier):

    '''Extends official Keras scikit-learn wrapper
    Only changes:
    1. adds classes_ field to adhere to scikit-learn convention
    2. automatically adds a param named input_dim to sk_params, which can be used to build model
    3. converts sparse matrix to dense, which can be problematic if input data is huge.
    a better option is to use a generator in the future

    '''

    def fit(self, X, y, **kwargs):
        self.sk_params['input_dim'] = X.shape[1]
        if issparse(X):
            X = X.toarray()
        self.classes_, y = np.unique(y, return_inverse=True)
        return super().fit(X, y, **kwargs)
    def predict_proba(self, X, **kwargs):
        if issparse(X):
            X = X.toarray()
        return super().predict_proba(X, **kwargs)
    def predict(self, X, **kwargs):
        return self.classes_[super().predict(X, **kwargs)]
    
    def set_params(self, **kwargs):
        
        return super().set_params(**kwargs)


if __name__ == '__main__':
    from model_info import ModelInfo

    clf = FlexInputKerasClassifier(build_fn=get_model, **{
            'l1_neurons': 128,
            'l2_neurons': 128,
            'nb_epoch': 2,
            'verbose': 2,
            'batch_size': 64,
            'dropout':0.4,
            'reg':0.001
        })

    m_net = ModelInfo(clf=clf,
                  data_parts=['brand', 'model', 'label', 'aid'], label='ann')


    m_net.run()
