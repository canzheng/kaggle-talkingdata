#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:32:10
# @Author  : Can Zheng (can.zheng@gmail.com)


def xgb_estimator_fit(alg, dtrain, dtarget, metrics, useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=dtarget.values)
        xgb_param['num_class'] = dtarget.nunique()

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=[metrics], early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        # print(cvresult.shape)
        # print(xgb.best_ntree_limit)

    #alg.fit(dtrain, dtarget,eval_metric=metrics)

    #feat_imp = pd.Series(alg.booster().get_fscore())[:50].sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

    return {'n_estimators': cvresult.shape[0]}


def get_val_right(from_val, method, step):
    if method == 'linear':
        return from_val + step
    elif method == 'log':
        return from_val * step
    else:
        print('Error: unsupported stepping method ', method)
        return None


def get_val_left(from_val, method, step):
    if method == 'linear':
        return from_val - step
    elif method == 'log':
        return from_val / step
    else:
        print('Error: unsupported stepping method ', method)
        return None


def coarse_to_fine_gs(estimator, X, y, scoring, stages, n_jobs=-1, iid=False, cv=10):
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone
    from pprint import pprint
    import time
    start_time = time.time()

    within = lambda x, scope: x >= scope[0] and x <= scope[1]
    base_params = estimator.get_params()

    for stage in stages:

        if 'precompute' in stage.keys():
            precomp_result = stage['precompute'](
                estimator, X, y, **stage['precompute_params'])
            print(precomp_result)
            base_params.update(precomp_result)
        gs_param = {}
        gs_step_idx = {}
        gs_searched = {}
        # set initial params
        param_info = stage['params']
        for p_name, p_data in param_info.items():
            init_val = p_data['initial_value']
            p_values = [
                v for v in
                [
                    get_val_left(
                        init_val, p_data['method'], p_data['stepping'][0]),
                    init_val,
                    get_val_right(
                        init_val, p_data['method'], p_data['stepping'][0])
                ] if within(v, param_info[p_name]['valid_scope'])]

            gs_param[p_name] = p_values
            gs_step_idx[p_name] = 0
            gs_searched[p_name] = []

        while (len(gs_param) > 0):
            base_estimator = clone(estimator)

            base_estimator.set_params(**base_params)
            base_estimator.set_params(**gs_param)
            print(base_estimator)

            gs = GridSearchCV(
                estimator=base_estimator, param_grid=gs_param, scoring=scoring, n_jobs=n_jobs, iid=iid, cv=cv)
            print(X.shape)

            gs.fit(X, y)
            print("Grid search done for params: ", gs_param)
            print("Time: ", round(((time.time() - start_time)/60), 2))
            pprint(gs.grid_scores_)
            pprint(gs.best_params_)
            pprint(gs.best_score_)
            new_gs_param = {}
            for p_name, p_value in gs.best_params_.items():
                new_param = []
                if p_value == min(gs_param[p_name]):
                    print(
                        p_name, " - best param is at left end of grid. searching smaller values")
                    v3 = p_value
                    v2 = get_val_left(
                        v3, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                    v1 = get_val_left(
                        v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                    new_param = [
                        v for v in [v1, v2, v3]
                        if within(v, param_info[p_name]['valid_scope']) and v not in gs_searched[p_name]]
                    if len(new_param) <= 0 and gs_step_idx[p_name] < len(param_info[p_name]['stepping']) - 1:
                        print(
                            p_name, " - no valid values, searching in smaller steps")
                        gs_step_idx[p_name] += 1
                        print(
                            "step size = ", param_info[p_name]['stepping'][gs_step_idx[p_name]])

                        v3 = p_value
                        v2 = get_val_left(
                            v3, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        v1 = get_val_left(
                            v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        new_param = [
                            v for v in [v1, v2, v3]
                            if within(v, param_info[p_name]['valid_scope']) and v not in gs_searched[p_name]]

                elif p_value == max(gs_param[p_name]):
                    print(
                        p_name, " - best param is at right end of grid. searching larger values")
                    v1 = p_value
                    v2 = get_val_right(
                        v1, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                    v3 = get_val_right(
                        v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                    new_param = [
                        v for v in [v1, v2, v3]
                        if within(v, param_info[p_name]['valid_scope']) and v not in gs_searched[p_name]]
                    if len(new_param) <= 1 and gs_step_idx[p_name] < len(param_info[p_name]['stepping']) - 1:
                        print(
                            p_name, " - no valid values, searching in smaller steps")
                        gs_step_idx[p_name] += 1
                        print(
                            "step size = ", param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        v1 = p_value
                        v2 = get_val_right(
                            v1, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        v3 = get_val_right(
                            v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        new_param = [
                            v for v in [v1, v2, v3]
                            if within(v, param_info[p_name]['valid_scope']) and v not in gs_searched[p_name]]

                else:
                    if len(new_param) <= 1 and gs_step_idx[p_name] < len(param_info[p_name]['stepping']) - 1:
                        print(
                            p_name, " - best param is at center of grid. searching in smaller steps")
                        gs_step_idx[p_name] += 1
                        param_info[p_name]['valid_scope'] = [
                            min(gs_param[p_name]), max(gs_param[p_name])]
                        print(
                            "step size = ", param_info[p_name]['stepping'][gs_step_idx[p_name]])

                        v2 = p_value
                        v1 = get_val_left(
                            v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        v3 = get_val_right(
                            v2, param_info[p_name]['method'], param_info[p_name]['stepping'][gs_step_idx[p_name]])
                        new_param = [
                            v for v in [v1, v2, v3]
                            if within(v, param_info[p_name]['valid_scope']) and v not in gs_searched[p_name]]
                print(p_name, " - tentative new param values: ", new_param)
                if(len(new_param) > 1):
                    new_gs_param[p_name] = new_param
                    for v in gs_param[p_name]:
                        if v != p_value:
                            gs_searched[p_name].append(v)
                else:
                    print(p_name, " - found optimal value: ", p_value)
                    base_params[p_name] = p_value

            # use the new params
            gs_param = new_gs_param
            print("=" * 70)

    return base_params
