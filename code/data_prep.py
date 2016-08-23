#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


from data_load import DataLoader
import pandas as pd
import numpy as np
import time
import pickle
from scipy import io
import argparse


def gen_vectorized_data(du, columns, label, base_dir='data/'):
    print('generating {} data'.format(label))
    x_train, x_test = du.vectorize_x(columns)
    print('train set shape: ', x_train.shape)
    print('test set shape: ', x_test.shape)
    io.mmwrite('{}train_{}.mtx'.format(base_dir, label), x_train)
    io.mmwrite('{}test_{}.mtx'.format(base_dir, label), x_test)
    print('completed generating {} data'.format(label))


def gen_term_data(du, base_dir='/data'):
    x_train, x_test = du.vectorize_EX(['category_bag'])
    print('train set shape: ', x_train.shape)
    print('test set shape: ', x_test.shape)
    io.mmwrite(base_dir + "train_term.mtx", x_train)
    io.mmwrite(base_dir + "test_term.mtx", x_test)


def gen_y(du, base_dir='/data'):
    y_train = du.get_y_train()
    print('y_train shape: ', y_train.shape)

    with open(base_dir + 'ytrain.pickle', 'wb') as f:
        pickle.dump(y_train, f, protocol=-1, fix_imports=False)


def gen_evind(base_dir='/data'):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--brand', action='store_true')
    parser.add_argument('--model', action='store_true')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--app', action='store_true')
    parser.add_argument('--term', action='store_true')
    parser.add_argument('--y', action='store_true')
    parser.add_argument('--evind', action='store_true')
    parser.add_argument('--all', action='store_true')

    parser.add_argument('vars', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    base_dir = 'data/'

    if len(args.vars) == 0:
        print('no base dir provided, use ./data as default')
    else:
        base_dir = args.vars[0]

    du = DataLoader()
    du.load_data(sub_sample=False)

    if args.brand or args.all:
        gen_vectorized_data(du, ['brand_code'], 'brand', base_dir)
    if args.model or args.all:
        gen_vectorized_data(du, ['model_code'], 'model', base_dir)
    if args.label or args.all:
        gen_vectorized_data(du, ['label_id_bag'], 'label', base_dir)
    if args.app or args.all:
        gen_vectorized_data(du, ['app_id_bag'], 'appid', base_dir)
    if args.term or args.all:
        gen_term_data(du, base_dir)
    if args.y or args.all:
        gen_y(du, base_dir)
    if args.evind or args.all:
        gen_evind(base_dir)

    print('Done')
