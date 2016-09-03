#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)

import pandas as pd
import numpy as np
from scipy import io
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from numpy import mean
from scipy.stats.mstats import gmean


class GenderAgeGroupProb(object):

    def __init__(self, by=['brand_code', 'model_code'], prior_weight=40.):
        self.prior_weight = prior_weight
        self.by = by
        self.coef_ = {}

    def get_params(self, deep=True):
        return {'prior_weight': self.prior_weight, 'by': self.by}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def __get_df(self, X, y=None):
        df = pd.DataFrame(X.toarray(), columns=['device_id','brand_code', 'model_code'])
        if y is not None:
            df['y'] = y
        return df

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        dtrain = self.__get_df(X, y)
        for by in self.by:
            prior, prob = self.__fit(dtrain, by)
            self.coef_[by] = (prior, prob)
        return self

    def predict_proba(self, X):
        dtest = self.__get_df(X)
        pred_list = []

        for by in self.by:
            (prior, prob) = self.coef_[by]
            pred = self.__predict_proba(dtest, by, prior, prob)
            pred_list.append(pred)

        p = mean(np.stack(pred_list), axis=0)
        return p

    def __fit(self, df, by):

        prior = df['y'].value_counts().sort_index()/df.shape[0]
        # fit gender probs by grouping column
        c = df.groupby([by, 'y']).size().unstack().fillna(0)
        total = c.sum(axis=1)
        prob = (c.add(self.prior_weight*prior)
                ).div(c.sum(axis=1)+self.prior_weight, axis=0)
        return prior, prob

    def __predict_proba(self, df, by, prior, prob):
        pred = df[[by]].merge(prob, how='left',
                              left_on=by, right_index=True).fillna(prior)[prob.columns]
        pred.loc[pred.iloc[:, 0].isnull(), :] = prior
        return pred.values


def load_pbm(base_dir='./'):
    ga_train = pd.read_csv(base_dir + 'gender_age_train.csv')
    bm_dev = pd.read_csv(
        base_dir + 'phone_brand_device_model.csv', encoding='utf-8')
    ga_test = pd.read_csv(base_dir + 'gender_age_test.csv')

    print('train set shape: ',  ga_train.shape)
    print('brand device set shape: ', bm_dev.shape)
    print('test set shape: ', ga_test.shape)

    # 1. there are duplicated device_ids, mostly with same brand / model, few with different brand / model
    # 2. some different brands have same models, use brand + model to make
    # model unique across brands

    print('Start cleaning device models')
    bm_dev_cleaned = bm_dev.drop_duplicates(subset='device_id')

    bm_dev_cleaned.loc[
        :, 'device_model'] = bm_dev_cleaned.phone_brand.str.cat(bm_dev_cleaned.device_model)

    brand_le = LabelEncoder()
    brand_code = brand_le.fit_transform(bm_dev_cleaned.phone_brand)
    bm_dev_cleaned.insert(
        len(bm_dev_cleaned.columns), 'brand_code', brand_code)


    model_le = LabelEncoder()
    model_code = model_le.fit_transform(bm_dev_cleaned.device_model)
    bm_dev_cleaned.insert(
        len(bm_dev_cleaned.columns), 'model_code', model_code)
    dtrain = ga_train[['device_id']].merge(
        bm_dev_cleaned[['device_id', 'brand_code', 'model_code']], on=['device_id'])
    dtrain = csr_matrix(dtrain.values)
    dtest = ga_test[['device_id']].merge(
        bm_dev_cleaned[['device_id', 'brand_code', 'model_code']], on=['device_id'])
    dtest = csr_matrix(dtest.values)

    io.mmwrite("data/train_pbm.mtx", dtrain)
    io.mmwrite("data/test_pbm.mtx", dtest)


if __name__ == '__main__':
    from model_info import ModelInfo, save_model
    #load_pbm()
    #print('load_pbm completed')

    clf = GenderAgeGroupProb()

    model = ModelInfo(clf=clf,
                      data_parts=['pbm'], n_folds=5, label='pbm_amean', mean_func=mean)

    model.run()
    save_model(model)