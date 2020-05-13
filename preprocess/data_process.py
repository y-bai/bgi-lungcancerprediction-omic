#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_process.py
    Description:
    
Created by YongBai on 2020/1/20 10:02 PM.
"""
from .imports import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFwe, SelectFromModel, SelectKBest, RFECV
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
import joblib


def data_split_transform(x, y, test_ratio=0.2, has_stage=False):

    if has_stage:
        x_normal_stage = x[y == 0]
        x_i_stage = x[y == 1]
        x_ii_stage = x[y == 2]
        x_iii_stage = x[y == 3]
        x_iv_stage = x[y == 4]

        y_normal_stage = y[y == 0]
        y_i_stage = y[y == 1]
        y_ii_stage = y[y == 2]
        y_iii_stage = y[y == 3]
        y_iv_stage = y[y == 4]

        x_normal_train, x_normal_test, y_normal_train, y_normal_test = train_test_split(
            x_normal_stage, y_normal_stage, test_size=test_ratio, random_state=123)

        x_i_train, x_i_test, y_i_train, y_i_test = train_test_split(
            x_i_stage, y_i_stage, test_size=test_ratio, random_state=123)

        x_ii_train, x_ii_test, y_ii_train, y_ii_test = train_test_split(
            x_ii_stage, y_ii_stage, test_size=test_ratio, random_state=123)

        x_iii_train, x_iii_test, y_iii_train, y_iii_test = train_test_split(
            x_iii_stage, y_iii_stage, test_size=test_ratio, random_state=123)

        x_iv_train, x_iv_test, y_iv_train, y_iv_test = train_test_split(
            x_iv_stage, y_iv_stage, test_size=test_ratio, random_state=123)

        x_train = np.vstack((x_normal_train, x_i_train, x_ii_train, x_iii_train, x_iv_train))
        y_train = np.hstack((y_normal_train, y_i_train, y_ii_train, y_iii_train, y_iv_train))
        x_test = np.vstack((x_normal_test, x_i_test, x_ii_test, x_iii_test, x_iv_test))
        y_test = np.hstack((y_normal_test, y_i_test, y_ii_test, y_iii_test, y_iv_test))

    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_ratio,
                                                        random_state=123,
                                                        stratify=y)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    # ss = StandardScaler()
    ss = MinMaxScaler()
    x_train_sf = ss.fit_transform(x_train)
    x_test_sf = ss.transform(x_test)
    model_path_conf = get_config()['model_path']
    scaler_file = os.path.join(model_path_conf['out_model_dir'], model_path_conf['model_split_scaler_file'])
    if os.path.exists(scaler_file):
        os.remove(scaler_file)
    joblib.dump(ss, scaler_file)

    return x_train_sf, y_train, x_test_sf, y_test


def data_whole_transform(x, y):
    ss = MinMaxScaler()
    x_whole_sf = ss.fit_transform(x)
    model_path_conf = get_config()['model_path']
    scaler_file = os.path.join(model_path_conf['out_model_dir'], model_path_conf['model_whole_scaler_file'])
    if os.path.exists(scaler_file):
        os.remove(scaler_file)
    joblib.dump(ss, scaler_file)
    return x_whole_sf, y


def feature_sel(x, y, sel_method='estimator', k=None,  estimator=None, score_func=chi2):
    """

    :param x:
    :param y:
    :param k:
    :param sel_method: kbest, fdr, fpr, fwe, estimator, rfecv
    :param estimator:
    :param score_func:
    :return:
    """

    if sel_method == 'kbest':
        assert k is not None
        selector = SelectKBest(score_func, k)
    elif sel_method == 'fdr':
        selector = SelectFdr(score_func, alpha=0.05)
    elif sel_method == 'fpr':
        selector = SelectFpr(score_func, alpha=0.05)
    elif sel_method == 'fwe':
        selector = SelectFwe(score_func, alpha=0.05)
    elif sel_method == 'estimator':
        assert estimator is not None
        if k is None:
            selector = SelectFromModel(estimator=estimator)
        else:
            selector = SelectFromModel(estimator=estimator, max_features=k, threshold=-np.inf)
    elif sel_method == 'rfecv':
        assert estimator is not None
        selector = RFECV(estimator, step=1, cv=5)
    else:
        raise Exception('unknown input parameters.')

    assert selector is not None
    x_new = selector.fit_transform(x, y)
    return selector.get_support(), x_new, y




