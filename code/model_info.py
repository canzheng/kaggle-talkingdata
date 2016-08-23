#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # @Author  : Can Zheng(can.zheng@gmail.com)
from scipy import io
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import log_loss
from scipy.stats.mstats import gmean
from numpy import mean
import pickle
import numpy as np

data_file = {
    'brand': 'brand.mtx',
    'model': 'model.mtx',
    'label': 'label.mtx',
    'aid': 'appid.mtx',
    'term': 'term.mtx',
    #'acount': 'appcount.mtx',
    #'afreq': 'appfreq.mtx',
}


def gen_folds(X, y, n_folds=5, random_state=0):
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)

    folds = kf.split(X, y)
    # iteratively train epochs
    kfsplit = [(itrain, itest) for itrain, itest in folds]
    return kfsplit


def load_model(label, base_path='code/'):
    with open(base_path + label + '.pickle', 'rb') as f:
        model_info = pickle.load(f)
    return model_info


def save_model(model_info, base_path='code/'):
    with open(base_path + model_info.label + '.pickle', 'wb') as f:
        y = pickle.dump(model_info, f, protocol=-1, fix_imports=False)
    return


def load_y(base_path='code/'):
    with open(base_path + 'ytrain.pickle', 'rb') as f:
        y = pickle.load(f).values
    return y


def assemble_data(parts):
    print('assemble data for ', parts)

    dtrain_parts = [load_data(p, prefix='train_') for p in parts]
    dtrain = hstack(dtrain_parts, format='csr')

    dtest_parts = [load_data(p, prefix='test_') for p in parts]
    dtest = hstack(dtest_parts, format='csr')
    print('train set data shape: ', dtrain.shape)
    print('test set data shape: ', dtest.shape)
    return dtrain, dtest


def assemble_stack(model_labels):
    print('assemble stack for ', model_labels)
    stack_models = [load_model(label) for label in model_labels]
    pred_oof = np.concatenate([m.pred_oof for m in stack_models], axis=1)
    pred_test = np.concatenate([m.pred_test for m in stack_models], axis=1)
    print('train set stack shape: ', pred_oof.shape)
    print('test set stack shape: ', pred_test.shape)

    return csr_matrix(pred_oof), csr_matrix(pred_test)


def load_data(label, prefix='train_', base_path='code/'):
    d = io.mmread(base_path + prefix + data_file[label]).tocsr()
    return d


def cv_predict(estimator, dtrain, dtest, kfsplit, y_train, mean_func, score_func, predict=True):
    # mean_func needs to support a 2nd parameters axis
    # mean_func(data, axis = 1)
    assert dtrain.shape[1] == dtest.shape[1], 'train and test has different number of features: {}'.format(
        (dtrain.shape[1], dtest.shape[1]))
    n_folds = len(kfsplit)
    print('Stacking for {} folds'.format(n_folds))
    pred_test = []
    pred_oof = None
    cv_score = np.zeros(n_folds)
    for f_idx, (itrain, itest) in enumerate(kfsplit):
        x_train_fold = dtrain[itrain, :]
        x_test_fold = dtrain[itest, :]
        y_train_fold = y_train[itrain]
        y_test_fold = y_train[itest]
        estimator.fit(x_train_fold, y_train_fold)

        pred = estimator.predict_proba(x_test_fold)
        cv_score[f_idx] = score_func(y_test_fold, pred)
        if predict:
            # initialize oof prediction array as (n_samples, n_classes)
            if pred_oof is None:
                pred_oof = np.zeros((dtrain.shape[0], len(estimator.classes_)))
            pred_oof[itest, :] = pred
            pred_test.append(estimator.predict_proba(dtest))

    if predict:
        # taking mean of n_folds prediction for test set
        mean_pred_test = mean_func(np.stack(pred_test), axis=0)

    print('cv score: {}, mean={}, std={}'.format(
        cv_score, cv_score.mean(), cv_score.std()))

    if predict:
        return cv_score, pred_oof, mean_pred_test
    else:
        return cv_score


class ModelInfo(object):

    def __init__(self, clf=None, clf_class=None, params=None, data_parts=[], stack=[],
                 n_folds=5, kf_seed=0, mean_func=gmean, score_func=log_loss,
                 pred_oof=None, pred_test=None, cv_score=None, label=None):
        if clf is not None:
            self.clf_class = type(clf)
            self.params = clf.get_params()
        else:
            if clf_class is None:
                raise ValueError('clf and clf_class can not both be None')
            else:
                self.clf_class = clf_class
                self.params = params
        self.data_parts = data_parts
        self.stack = stack
        self.pred_oof = pred_oof
        self.pred_test = pred_test
        self.cv_score = cv_score
        self.n_folds = n_folds
        self.kf_seed = kf_seed
        self.mean_func = mean_func
        self.score_func = score_func
        if label is None:
            self.label = self.__make_label()
        else:
            self.label = label

    def _get_estimator(self):
        return self.clf_class(**self.params)

    def __make_label():
        from time import strftime
        # label is in below format:
        # <estimator>_<data_parts>_<S<num_of_stacked_models>>_postfix
        label = '{}_{}_S{}_{}'.format(self.clf_class.__name__[:4],
                                      '-'.join([p[:2]
                                                for p in self.data_parts]),
                                      len(self.stack),
                                      strftime('%Y%m%d%H%M%S')
                                      )

    def run(self, verify_only=False):
        # prepare data
        train_list = []
        test_list = []
        if len(self.data_parts) > 0:
            data_train, data_test = assemble_data(self.data_parts)
            train_list.append(data_train)
            test_list.append(data_test)

        if len(self.stack) > 0:
            stack_train, stack_test = assemble_stack(self.stack)
            train_list.append(stack_train)
            test_list.append(stack_test)

        x_train = hstack(train_list, format='csr')
        x_test = hstack(test_list, format='csr')

        print('Final train set shape: ', x_train.shape)
        print('Final test set shape: ', x_test.shape)

        y_train = load_y()
        # get kfolds
        kfsplit = gen_folds(
            x_train, y_train, n_folds=self.n_folds, random_state=self.kf_seed)
        cv_score, pred_oof, pred_test = cv_predict(self._get_estimator(), x_train, x_test, kfsplit, y_train,
                                                   mean_func=self.mean_func, score_func=self.score_func)
        if verify_only:
            assert np.array_equal(
                pred_oof, self.pred_oof), 'Verification Warning: calculated different pred_oof'
            assert np.array_equal(
                pred_test, self.pred_test), 'Verification Warning: calculated different pred_test'
            assert np.array_equal(
                cv_score, self.cv_score), 'Verification Warning: calculated different cv score'

        else:
            if self.pred_oof is not None or self.pred_test is not None or self.cv_score is not None:
                print(
                    'Warning: model {} already has prediction data, will be overwriten')
            self.pred_oof = pred_oof
            self.pred_test = pred_test
            self.cv_score = cv_score
        return pred_oof, pred_test, cv_score


if __name__ == '__main__':
    # test model_info class
    from sklearn.linear_model import LogisticRegression

    m = ModelInfo(clf=LogisticRegression(C=0.02, solver='lbfgs', multi_class='multinomial'),
                  data_parts=['brand', 'model'], label='logreg_basic')
    pred_oof, pred_test, cv_score = m.run()
    pred_oof.shape, pred_test.shape, cv_score.shape

    print('save test model')
    save_model(m)
    print('load test model')

    m1 = load_model('logreg_basic')
    pred_oof, pred_test, cv_score = m1.run(verify_only=True)
    try:
        m1.params['C'] = 0.01
        pred_oof, pred_test, cv_score = m1.run(verify_only=True)
    except AssertionError:
        print('Caught different predictions as expected')
