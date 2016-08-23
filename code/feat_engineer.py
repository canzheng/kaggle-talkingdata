#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


import numpy as np
import time
from scipy.sparse import hstack, csr_matrix
from scipy import io
import pandas as pd
import argparse


def gen_app_count(dev_app, ga_train, ga_test, base_dir='/data'):
    start_time = time.time()

    print('generating app count per device')

    app_count = dev_app.groupby(['device_id'])['app_id'].agg(
        {'app_count': lambda x: x.nunique()})
    app_count_train = ga_train['device_id'].map(app_count['app_count']).fillna(0)
    app_count_train = app_count_train / app_count_train.max()

    app_count_train = csr_matrix(app_count_train.values).transpose()

    app_count_test = ga_test['device_id'].map(app_count['app_count']).fillna(0)
    app_count_test = app_count_test / app_count_test.max()

    app_count_test = csr_matrix(app_count_test.values).transpose()


    print('train set shape: ', app_count_train.shape)
    io.mmwrite(base_dir + "train_appcount.mtx", app_count_train)

    print('test set shape: ', app_count_test.shape)
    io.mmwrite(base_dir + "test_appcount.mtx", app_count_test)
    print('Time generating app count: ', (time.time() - start_time) / 60)


def gen_app_pop_count(dev_app, ga_train, ga_test, base_dir='/data'):
    start_time = time.time()

    print('generating popularity weighted app count per device')

    app_popularity = dev_app.groupby(['app_id'])['device_id'].agg(
        {'popularity': lambda x: x.nunique()})
    app_pop_count = dev_app.groupby(['device_id'])['app_id'].agg(
        {'app_pop_count': lambda x: app_popularity.loc[x.unique(), 'popularity'].sum()})


    app_count_train = ga_train['device_id'].map(
        app_pop_count['app_pop_count']).fillna(0)
    app_count_train = app_count_train / app_count_train.max()

    app_count_train = csr_matrix(app_count_train.values).transpose()

    app_count_test = ga_test['device_id'].map(app_pop_count['app_pop_count']).fillna(0)
    app_count_test = app_count_test / app_count_test.max()

    app_count_test = csr_matrix(app_count_test.values).transpose()

    print('train set shape: ', app_count_train.shape)
    io.mmwrite(base_dir + "train_apppopcount.mtx", app_count_train)

    print('test set shape: ', app_count_test.shape)
    io.mmwrite(base_dir + "test_apppopcount.mtx", app_count_test)
    print('Time generating app pop count: ', (time.time() - start_time) / 60)


def gen_app_freq(dev_app, ga_train, ga_test, base_dir='/data'):
    start_time = time.time()

    print('generating app freq data')
    app_ev_count = dev_app.groupby(
        ['device_id', 'app_id']).size().unstack(fill_value=0)

    ev_count = dev_app.groupby(['device_id'])['event_id'].agg(
        {'ev_count': lambda x: x.nunique()})

    # freq of app appearing in events for each device (across all event data)
    app_freq = pd.DataFrame(np.divide(
        app_ev_count.values, ev_count.values), index=app_ev_count.index, columns=app_ev_count.columns)



    del app_ev_count, ev_count



    # get freq for train data only
    app_freq_train = app_freq.loc[ga_train.device_id.values, :].fillna(0)

    # get apps not appearing in train devices, drop them in both train and test
    # because test can not utilize app info that is noshow in train
    app_freq_sum_train = app_freq_train.sum(axis=0)
    noshow_apps = app_freq_sum_train[app_freq_sum_train == 0].index

    app_freq_train.drop(noshow_apps, axis=1, inplace=True)
    app_freq_train = csr_matrix(app_freq_train.values)

    print('train set shape: ', app_freq_train.shape)
    print('train sparsity: ', app_freq_train.getnnz() /
          (app_freq_train.shape[0] * app_freq_train.shape[1]))
    io.mmwrite(base_dir + "train_appfreq.mtx", app_freq_train)
    del app_freq_train

    app_freq_test = app_freq.loc[ga_test.device_id.values, :].fillna(0)
    app_freq_test.drop(noshow_apps, axis=1, inplace=True)
    app_freq_test = csr_matrix(app_freq_test.values)

    print('test set shape: ', app_freq_test.shape)
    print('test sparsity: ', app_freq_test.getnnz() /
          (app_freq_test.shape[0] * app_freq_test.shape[1]))
    io.mmwrite(base_dir + "test_appfreq.mtx", app_freq_test)
    print('Time generating app frequency: ', (time.time() - start_time) / 60)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', action='store_true')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--popcount', action='store_true')
    parser.add_argument('--all', action='store_true')

    parser.add_argument('vars', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    base_dir = 'data/'

    if len(args.vars) == 0:
        print('no base dir provided, use ./data as default')
    else:
        base_dir = args.vars[0]

    ga_train = pd.read_csv('gender_age_train.csv')
    ga_test = pd.read_csv('gender_age_test.csv')
    ev = pd.read_csv('events.csv')
    app_ev = pd.read_csv('app_events.csv')
    dev_app = app_ev.merge(ev[['event_id', 'device_id']], how='inner', on='event_id')
    del ev, app_ev

    if args.count or args.all:
        gen_app_count(dev_app, ga_train, ga_test, base_dir)
    if args.popcount or args.all:
        gen_app_pop_count(dev_app, ga_train, ga_test, base_dir)
    if args.freq or args.all:
        gen_app_freq(dev_app, ga_train, ga_test, base_dir)

    print('Done')

