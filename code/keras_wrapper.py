#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)

import inspect
from time import strftime
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Sequential
from scipy.sparse import issparse
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np


def get_model(input_dim=100, l1_neurons=512, l2_neurons=512,
              reg=0.001,
              dropout=0.4, output_dim=12):
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
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def get_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[int(batch_size*counter) : int(batch_size*(counter+1))]
        counter += 1
        X_batch = X[batch_index,:].toarray()
        if y is None:
            yield X_batch
        else:
            y_batch = y[batch_index, :]
            yield X_batch, y_batch
        
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

class FlexInputKerasBase(BaseEstimator):

    '''Extends official Keras scikit-learn wrapper
    Only changes:
    1. adds classes_ field to adhere to scikit-learn convention
    2. automatically adds a param named input_dim to sk_params, which can be used to build model
    3. converts sparse matrix to dense, which can be problematic if input data is huge.
    4. require validation_data for earlystopping, optionally model_path and patience
    a better option is to use a generator in the future
    5. add generaters
    '''

    def __init__(self, build_fn=get_model, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params

    


    def get_params(self, deep=True):
        '''Get parameters for this estimator.
        # Arguments
            deep: boolean, optional
                If True, will return the parameters for this estimator and
                contained sub-objects that are estimators.
        # Returns
            params : dict
                Dictionary of parameter names mapped to their values.
        '''
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        '''Set the parameters of this estimator.
        # Arguments
        params: dict
            Dictionary of parameter names mapped to their values.
        # Returns
            self
        '''
        self.build_fn = params.pop('build_fn', get_model)
        self.sk_params.update(params)
        return self



    def fit(self, X, y, **kwargs):
        # require params:
        # validation_data
        #
        # optional params:
        # model_path,
        # patience
        # 
        self.classes_, y = np.unique(y, return_inverse=True)


        shuffle = kwargs.get('shuffle', self.sk_params.get('shuffle', True))
        batch_size = kwargs.get('batch_size', self.sk_params.get('batch_size', 64))
        
        
        self.sk_params['input_dim'] = X.shape[1]
        self.sk_params['output_dim'] = len(self.classes_)

        self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_method = self.model.fit

        if issparse(X):
            # if X is not sparse ignore fit_generator
            if self.sk_params.get('fit_generator', False):
                kwargs['generator'] = get_generator(X, y, batch_size, shuffle)
                fit_method = self.model.fit_generator
                kwargs['samples_per_epoch'] = X.shape[0]
                #kwargs['pickle_safe'] = True
                if 'shuffle' in kwargs.keys():
                    del kwargs['shuffle']
            else:
                kwargs['x'] = X.toarray()
                kwargs['y'] = y
        else: 
            kwargs['x'] = X
            kwargs['y'] = y



        # require validation_data is provided, and transform to categorical 
        validation_data = kwargs.pop('validation_data', None)
        if validation_data is not None:
            if loss_name == 'categorical_crossentropy' and len(validation_data[1].shape) != 2:
                y_val = to_categorical(validation_data[1])
            else:
                y_val = validation_data[1]
            x_val = validation_data[0]
            if issparse(x_val):
                if self.sk_params.get('val_generator', False):
                    kwargs['validation_data'] = get_generator(x_val, y_val, batch_size / 2, False)
                    kwargs['nb_val_samples'] = x_val.shape[0]
                else:
                    kwargs['validation_data'] = (x_val.toarray(), y_val)
            else:
                kwargs['validation_data'] = (x_val, y_val)
            
            # automatically enable early-stopping and model checkpoint
            patience = self.sk_params.get('patience', 5)
            model_path = self.sk_params.get(
                    'model_path', 'model/{}.h5'.format(strftime('%Y%m%d%H%M%S')))

            earlystopping = EarlyStopping(patience=patience)
            checkpoint = ModelCheckpoint(
                model_path,
                save_best_only=True)

            kwargs['callbacks'] = [earlystopping, checkpoint]

        fit_args = copy.deepcopy(self.filter_sk_params(fit_method))
        fit_args.update(kwargs)
        #if fit_args.get('verbose', 1) > 1:
        #    print('Fitting model: ', fit_args)
        

        self.hist_ = fit_method(**fit_args)
        if fit_args.get('verbose', 1) > 1:
            print('Hist: ', self.hist_)
            
        if validation_data is not None:
            print('best epoch: {}'.format(np.array(self.hist_.history['val_loss']).argmin() + 1))
            
            self.model = load_model(model_path)
        return self


    def filter_sk_params(self, fn, override={}):
        '''Filter sk_params and return those in fn's arguments
        # Arguments
            fn : arbitrary function
            override: dictionary, values to override sk_params
        # Returns
            res : dictionary dictionary containing variables
                in both sk_params and fn's arguments.
        '''
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in self.sk_params.items():
            if name in fn_args:
                res.update({name: value})
        res.update(override)
        return res

class FlexInputKerasClassifier(FlexInputKerasBase):
    def predict_proba(self, X, **kwargs):
        batch_size = kwargs.get('batch_size', self.sk_params.get('batch_size', 64))
        

        predict_method = self.model.predict_proba

        if issparse(X):
            # if X is not sparse ignore fit_generator
            if self.sk_params.get('pred_generator', False):
                kwargs['generator'] = get_generator(X, None, batch_size, False)
                predict_method = self.model.predict_generator
                kwargs['val_samples'] = X.shape[0]
                #kwargs['pickle_safe'] = True
            else:
                kwargs['x'] = X.toarray()
        else: 
            kwargs['x'] = X



        kwargs = self.filter_sk_params(predict_method, kwargs)
        probs = predict_method(**kwargs)
        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X, **kwargs):
        D = self.predict_proba(X, **kwargs)
        return self.classes_[np.argmax(D, axis=1)]


if __name__ == '__main__':
    from model_info import ModelInfo

    clf = FlexInputKerasClassifier(build_fn=get_model, **{
        'l1_neurons': 128,
        'l2_neurons': 128,
        'nb_epoch': 4,
        'verbose': 2,
        'batch_size': 64,
        'dropout': 0.4,
        'reg': 0.001,
        'fit_generator': True,
        'val_generator': True,
        'pred_generator': True
    })

    m_net = ModelInfo(clf=clf,
                      data_parts=['brand', 'model'], n_folds=2, label='ann')

    m_net.run()
