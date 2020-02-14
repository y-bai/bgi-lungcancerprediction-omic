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


def data_split_transform(x, y, test_ratio=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_ratio,
                                                        random_state=123,
                                                        stratify=y)
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




