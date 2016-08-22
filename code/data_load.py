#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # @Author  : Can Zheng(can.zheng@gmail.com)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from numpy import random
import time
from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer

cat_stop = ['nan', 'app', '),', 'application', 'based', 'category', 'unknown']

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc) if (len(self.wnl.lemmatize(t)) > 1 and (self.wnl.lemmatize(t) not in cat_stop))]

class DataLoader:

    def __init__(self, base_dir='../'):
        self.base_dir = base_dir

    def load_data(self, sub_sample=True, sample_rate=0.1, rand_seed=1999):

        print('Start loading data')
        random.seed(rand_seed)
        ga_train = pd.read_csv(self.base_dir + 'gender_age_train.csv')
        lab_cat = pd.read_csv(self.base_dir + 'label_categories.csv')
        lab_app = pd.read_csv(self.base_dir + 'app_labels.csv')
        ev = pd.read_csv(self.base_dir + 'events.csv')
        app_ev = pd.read_csv(self.base_dir + 'app_events.csv')
        bm_dev = pd.read_csv(
            self.base_dir + 'phone_brand_device_model.csv', encoding='utf-8')
        ga_test = pd.read_csv(self.base_dir + 'gender_age_test.csv')

        if (sub_sample):
            print('Sub-sampling with sample rate :', sample_rate)
            num_train = int(ga_train.shape[0] * sample_rate)
            num_test = int(ga_test.shape[0] * sample_rate)
            train_did_sample = random.choice(
                ga_train.device_id.values, size=num_train, replace=False)
            test_did_sample = random.choice(
                ga_test.device_id.values, size=num_test, replace=False)
            ga_train = ga_train.loc[
                ga_train.device_id.isin(train_did_sample), :]
            ga_test = ga_test.loc[ga_test.device_id.isin(test_did_sample), :]

            sample_did = np.concatenate((train_did_sample, test_did_sample))
            ev = ev.loc[ev.device_id.isin(sample_did), :]
            app_ev = app_ev.loc[app_ev.event_id.isin(ev.event_id), :]

        print('Complete loading data')

        print('train set shape: ',  ga_train.shape)
        print('label category set shape: ', lab_cat.shape)
        print('label app set shape: ', lab_app.shape)
        print('event set shape: ', ev.shape)
        print('app event set shape: ', app_ev.shape)
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
        brand_code = pd.Series(
            brand_le.fit_transform(bm_dev_cleaned.phone_brand).astype(np.str)) + 'b'
        bm_dev_cleaned.insert(
            len(bm_dev_cleaned.columns), 'brand_code', brand_code.values)

        model_le = LabelEncoder()
        model_code = pd.Series(
            model_le.fit_transform(bm_dev_cleaned.device_model).astype(np.str)) + 'm'
        bm_dev_cleaned.insert(
            len(bm_dev_cleaned.columns), 'model_code', model_code.values)

        print('Complete cleaning device models')

        # ## Generating bags
        #
        # dev_gender_age -> events -> app_event -> app_label -> app_category

        print('Start generating bags')

        start_time = time.time()

        # label_id in label_cat is unque, so use category instead of label
        app_cat = lab_app.merge(lab_cat, how='left', on=['label_id'])

        # generate category bag for app
        app_cat_bag = app_cat.groupby(['app_id'], sort=False).agg(
            {'category': lambda x: ','.join(x.astype(np.str)),
             'label_id': lambda x: ','.join(x.astype(np.str))})

        # join app category bag with app_event table
        app_ev['category_bag'] = app_ev['app_id'].map(app_cat_bag['category'])
        app_ev['label_id_bag'] = app_ev['app_id'].map(app_cat_bag['label_id'])

        # generate app bags for events
        app_ev_grouped = app_ev.groupby(['event_id'], sort=False)

        app_ev_bag = app_ev_grouped.agg({
            'category_bag': lambda x: ','.join(x.astype(np.str)),
            'label_id_bag': lambda x: ','.join(x.astype(np.str)),
            'app_id': lambda x: ','.join(x.astype(np.str))
        })

        ev['app_id_bag'] = ev['event_id'].map(app_ev_bag['app_id'])
        ev['category_bag'] = ev['event_id'].map(app_ev_bag['category_bag'])
        ev['label_id_bag'] = ev['event_id'].map(app_ev_bag['label_id_bag'])

        ev_bag = ev.groupby(['device_id'], sort=False).agg({
            'category_bag': lambda x: ','.join(x.astype(np.str)),
            'app_id_bag': lambda x: ','.join(x.astype(np.str)),
            'label_id_bag': lambda x: ','.join(x.astype(np.str))
        })

        self.ga_bm_train = ga_train.merge(bm_dev_cleaned, on=['device_id'])
        self.ga_bm_train['app_id_bag'] = self.ga_bm_train[
            'device_id'].map(ev_bag['app_id_bag'])
        self.ga_bm_train['category_bag'] = self.ga_bm_train[
            'device_id'].map(ev_bag['category_bag'])
        self.ga_bm_train['label_id_bag'] = self.ga_bm_train[
            'device_id'].map(ev_bag['label_id_bag'])
        self.ga_bm_train['app_id_bag'].fillna('nan', inplace=True)
        self.ga_bm_train['category_bag'].fillna('nan', inplace=True)
        self.ga_bm_train['label_id_bag'].fillna('nan', inplace=True)

        self.ga_bm_test = ga_test.merge(bm_dev_cleaned, on=['device_id'])
        self.ga_bm_test['app_id_bag'] = self.ga_bm_test[
            'device_id'].map(ev_bag['app_id_bag'])
        self.ga_bm_test['category_bag'] = self.ga_bm_test[
            'device_id'].map(ev_bag['category_bag'])
        self.ga_bm_test['label_id_bag'] = self.ga_bm_test[
            'device_id'].map(ev_bag['label_id_bag'])
        self.ga_bm_test['app_id_bag'].fillna('nan', inplace=True)
        self.ga_bm_test['category_bag'].fillna('nan', inplace=True)
        self.ga_bm_test['label_id_bag'].fillna('nan', inplace=True)

        print("Time: ", round(((time.time() - start_time) / 60), 2))

    def vectorize_x(self, columns, variance_thresh=0, train_only=False):

        print('Start vectorizing')
        start_time = time.time()

        hasher = CountVectorizer(binary=True, stop_words=['nan'])

        train_dtm = hasher.fit_transform(
            self.ga_bm_train[columns].apply(lambda x: ','.join(x), axis=1))

        print('dtm train shape: ', train_dtm.shape)

        selector = VarianceThreshold(variance_thresh)
        train_dtm = selector.fit_transform(train_dtm)
        print('dtm train shape after variance thresh: ', train_dtm.shape)

        if not train_only:
            test_dtm = hasher.transform(
                self.ga_bm_test[columns].apply(lambda x: ','.join(x), axis=1))

            print('dtm test shape: ', test_dtm.shape)
            test_dtm = selector.transform(test_dtm)
            print('dtm test shape after variance thresh: ', test_dtm.shape)

        print("Time: ", round(((time.time() - start_time)/60), 2))
        print('Complete vectorizing')
        if train_only:
            return train_dtm
        else:
            return (train_dtm, test_dtm)

    def vectorize_EX(self, columns, variance_thresh=0, train_only=False):

        print('Start vectorizing')
        start_time = time.time()
        hasher = CountVectorizer(binary=True, tokenizer=LemmaTokenizer(), stop_words='english')

        train_dtm = hasher.fit_transform(
            self.ga_bm_train[columns].apply(lambda x: ','.join(x), axis=1))
        print(hasher.get_feature_names())
        print('dtm train shape: ', train_dtm.shape)

        selector = VarianceThreshold(variance_thresh)
        train_dtm = selector.fit_transform(train_dtm)
        print('dtm train shape after variance thresh: ', train_dtm.shape)

        if not train_only:
            test_dtm = hasher.transform(
                self.ga_bm_test[columns].apply(lambda x: ','.join(x), axis=1))

            print('dtm test shape: ', test_dtm.shape)
            test_dtm = selector.transform(test_dtm)
            print('dtm test shape after variance thresh: ', test_dtm.shape)

        print("Time: ", round(((time.time() - start_time)/60), 2))
        print('Complete vectorizing')
        if train_only:
            return train_dtm
        else:
            return (train_dtm, test_dtm)
            

    def get_y_train(self):

        # prior to feed into XGB, encode the labels of the groups

        self.group_le = LabelEncoder()
        return pd.Series(self.group_le.fit_transform(self.ga_bm_train.group))

    def export_prediction(self, pred, file_name):
        pred_df = pd.DataFrame(
            pred, columns=self.group_le.inverse_transform(np.arange(12)))
        pred_df.insert(0, 'device_id', self.ga_bm_test.device_id)
        pred_df.to_csv(self.base_dir + file_name, index=False)


if __name__ == '__main__':
    du = DataLoader()
    du.load_data(sample_rate=0.05)
    train_dtm, test_dtm = du.vectorize_x(['brand_code', 'model_code', 'label_id_bag'])
    print('train set shape: ', train_dtm.shape)
    print('test set shape: ', test_dtm.shape)
    y_train = du.get_y_train()
    print('y_train shape: ', y_train.shape)
