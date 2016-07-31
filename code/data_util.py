#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # @Author  : Can Zheng(can.zheng@gmail.com)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import time

class DataUtil:
    def __init__(self, base_dir='../'):
        self.base_dir = base_dir
        

    def load_data(self):

        print('Start loading data')
        ga_train = pd.read_csv(self.base_dir + 'gender_age_train.csv')
        lab_cat = pd.read_csv(self.base_dir +  'label_categories.csv')
        lab_app = pd.read_csv(self.base_dir +  'app_labels.csv')
        ev = pd.read_csv(self.base_dir +  'events.csv')
        app_ev = pd.read_csv(self.base_dir +  'app_events.csv')
        bm_dev = pd.read_csv(self.base_dir +  'phone_brand_device_model.csv', encoding='utf-8')
        ga_test = pd.read_csv(self.base_dir +  'gender_age_test.csv')
        print('Complete loading data')


        print('train set shape: ',  ga_train.shape)
        print('label category set shape: ', lab_cat.shape)
        print('label app set shape: ', lab_app.shape)
        print('event set shape: ', ev.shape)
        print('app event set shape: ', app_ev.shape)
        print('brand device set shape: ', bm_dev.shape)
        print('test set shape: ', ga_test.shape)

        # 1. there are duplicated device_ids, mostly with same brand / model, few with different brand / model
        # 2. some different brands have same models, use brand + model to make model unique across brands

        print('Start cleaning device models')
        bm_dev_cleaned = bm_dev.drop_duplicates(subset='device_id')

        bm_dev_cleaned.loc[:, 'device_model'] = bm_dev_cleaned.phone_brand + bm_dev_cleaned.device_model

        brand_le = LabelEncoder()
        brand_code =  pd.Series(brand_le.fit_transform(bm_dev_cleaned.phone_brand).astype(np.str)) + 'b'
        bm_dev_cleaned.insert(len(bm_dev_cleaned.columns), 'brand_code', brand_code.values)


        model_le = LabelEncoder()
        model_code = pd.Series(model_le.fit_transform(bm_dev_cleaned.device_model).astype(np.str)) + 'm'
        bm_dev_cleaned.insert(len(bm_dev_cleaned.columns), 'model_code', model_code.values)

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
        self.ga_bm_train['app_id_bag'] = ga_bm_train['device_id'].map(ev_bag['app_id_bag'])
        self.ga_bm_train['category_bag'] = ga_bm_train['device_id'].map(ev_bag['category_bag'])
        self.ga_bm_train['label_id_bag'] = ga_bm_train['device_id'].map(ev_bag['label_id_bag'])
        self.ga_bm_train['app_id_bag'].fillna('nan', inplace=True)
        self.ga_bm_train['category_bag'].fillna('nan', inplace=True)
        self.ga_bm_train['label_id_bag'].fillna('nan', inplace=True)


        self.ga_bm_test = ga_test.merge(bm_dev_cleaned, on=['device_id'])
        self.ga_bm_test['app_id_bag'] = ga_bm_test['device_id'].map(ev_bag['app_id_bag'])
        self.ga_bm_test['category_bag'] = ga_bm_test['device_id'].map(ev_bag['category_bag'])
        self.ga_bm_test['label_id_bag'] = ga_bm_test['device_id'].map(ev_bag['label_id_bag'])
        self.ga_bm_test['app_id_bag'].fillna('nan', inplace=True)
        self.ga_bm_test['category_bag'].fillna('nan', inplace=True)
        self.ga_bm_test['label_id_bag'].fillna('nan', inplace=True)

        print("Time: ", round(((time.time() - start_time)/60),2))

        
        
    def vectorize_x(self, columns, train_only=False)

        print('Start vectorizing')
        start_time = time.time()

        hasher = CountVectorizer(binary=True, stop_words=['nan'])


        train_dtm = hasher.fit_transform(
            ga_bm_train[columns].apply(lambda x: ','.join(x), axis = 1))

        print('dtm train shape: ', train_dtm.shape)

        if not train_only:
            test_dtm=hasher.transform(
                ga_bm_test[columns].apply(lambda x: ','.join(x), axis = 1))
            print('dtm test shape: ', test_dtm.shape)

        print("Time: ", round(((time.time() - start_time)/60),2))
        print('Complete vectorizing')
        return (train_dtm, test_dtm) if train_only else test_dtm

    def get_y_train(self):

        # prior to feed into XGB, encode the labels of the groups
        
        self.group_le = LabelEncoder()
        self.group_le.fit(ga_bm_train.group)
        return pd.Series(self.group_le.transform(ga_bm_train.group))

    def export_prediction(self, pred, file_name):
        pred_df = pd.DataFrame(pred, columns = self.group_le.inverse_transform(np.arange(12)))
        pred_df.insert(0, 'device_id', ga_bm_test.device_id)
        pred_df.to_csv(self.base_dir + file_name, index=False)

