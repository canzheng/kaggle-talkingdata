#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # @Author  : Can Zheng(can.zheng@gmail.com)
from model_info import ModelInfo, save_model
from keras_wrapper import FlexInputKerasClassifier, get_model
import gc

#data_file = {
#    'brand': 'brand.mtx',
#    'model': 'model.mtx',
#    'label': 'label.mtx',
#    'appid': 'appid.mtx',
#    'term': 'term.mtx',
#    'appcount': 'appcount.mtx',
#    'appfreq': 'appfreq.mtx',
#    'apppopcount': 'apppopcount.mtx'
#}


if __name__ == '__main__':
    data_parts=['brand', 'model', 'label', 'appid']
#    print("===========Model 1===============")
#    params = {
#        'nb_epoch':20,
#        'l1_neurons':512,
#        'l2_neurons':512,
#        'reg':0.001,
#        'dropout':0.4,
#        'verbose':2,
#        'fit_generator': True,
#        'val_generator': True,
#        'pred_generator': True
#        }
#
#    m = ModelInfo(clf=FlexInputKerasClassifier(build_fn=get_model, **params),
#                  data_parts=data_parts, label='nnet_512_001_04_{}'.format('-'.join(data_parts)))
#
#    m.run()
#    save_model(m)
    
#    print("===========Model 2===============")
#    params = {
#        'nb_epoch':20,
#        'l1_neurons':512,
#        'l2_neurons':512,
#        'reg':0.001,
#        'dropout':0.5,
#        'verbose':0,
#        'fit_generator': True,
#        'val_generator': True,
#        'pred_generator': True
#        }
#
#    m = ModelInfo(clf=FlexInputKerasClassifier(build_fn=get_model, **params),
#                  data_parts=data_parts, label='nnet_512_001_05_{}'.format('-'.join(data_parts)))
#
#    m.run()
#    save_model(m)

#    print("===========Model 3===============")
#    params = {
#        'nb_epoch':20,
#        'l1_neurons':512,
#        'l2_neurons':512,
#        'reg':0,
#        'dropout':0.5,
#        'verbose':0,
#        'fit_generator': True,
#        'val_generator': True,
#        'pred_generator': True
#        }
#
#    m = ModelInfo(clf=FlexInputKerasClassifier(build_fn=get_model, **params),
#                  data_parts=data_parts, label='nnet_512_noreg_05_{}'.format('-'.join(data_parts)))
#
#    m.run()
#    save_model(m)
#    del m
#    gc.collect()

    print("===========Model 4===============")
    params = {
        'nb_epoch':20,
        'l1_neurons':1024,
        'l2_neurons':1024,
        'reg':0.0,
        'dropout':0.5,
        'verbose':0,
        'fit_generator': True,
        'val_generator': True,
        'pred_generator': True
        }

    m = ModelInfo(clf=FlexInputKerasClassifier(build_fn=get_model, **params),
                  data_parts=data_parts, label='nnet_1024_noreg_05_{}'.format('-'.join(data_parts)))

    m.run()
    save_model(m)

    del m
    gc.collect()
    print("===========Model 5===============")
    params = {
        'nb_epoch':20,
        'l1_neurons':1024,
        'l2_neurons':1024,
        'reg':0.001,
        'dropout':0.5,
        'verbose':0,
        'fit_generator': True,
        'val_generator': True,
        'pred_generator': True
        }

    m = ModelInfo(clf=FlexInputKerasClassifier(build_fn=get_model, **params),
                  data_parts=data_parts, label='nnet_1024_001_05_{}'.format('-'.join(data_parts)))

    m.run()
    save_model(m)


