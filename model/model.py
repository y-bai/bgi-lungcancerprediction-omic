#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: model.py
    Description:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
Created by YongBai on 2020/1/20 11:20 PM.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold


def get_class_weight(y_train):
    cls = np.unique(y_train)
    cls_weight = compute_class_weight('balanced', cls, y_train)
    class_weight_dict = dict(zip(cls, cls_weight))
    return class_weight_dict


def model_train_run(x_train, y_train, cv_score=True, n_jobs=8, cls_weight=None):
    # if cls_weight is None:
    #     cls = np.unique(y_train)
    #     cls_wei = np.ones(len(cls)) * 0.5
    #     cls_weight = dict(zip(cls, cls_wei))
    clfs = [
        ('rbfsvm', SVC(gamma='scale', C=1, class_weight=cls_weight)),
        ('nb', GaussianNB()),
        ('gp', GaussianProcessClassifier(1.0 * RBF(1.0))),
        ('etc', ExtraTreesClassifier(n_estimators=40, max_depth=5, max_features='auto', class_weight=cls_weight)),
        ('rf', RandomForestClassifier(n_estimators=40, max_depth=5, max_features='auto', class_weight=cls_weight)),
        ('gbm', GradientBoostingClassifier(n_estimators=40, max_depth=5,
                                           max_features='auto', learning_rate=0.01)),
        ('xgb', XGBClassifier(n_estimators=40, max_depth=5, learning_rate=0.01))
    ]

    # clf = VotingClassifier(estimators=clfs, voting='soft')
    scores = None
    if cv_score:
        clf = StackingClassifier(estimators=clfs,
                                 final_estimator=LogisticRegression(
                                     class_weight=cls_weight,
                                     multi_class='multinomial',
                                     max_iter=1000))  # 0.5
        scores = cross_validate(clf, x_train, y_train,
                                cv=3, n_jobs=n_jobs,
                                verbose=1, scoring=['f1_macro'])
                                # return_train_score=True)

    f_clf = StackingClassifier(estimators=clfs, final_estimator=LogisticRegression(class_weight=cls_weight), cv=3)
    f_clf.fit(x_train, y_train)
    return f_clf, scores


def single_model_train_run(x_train, y_train, cv_score=True, cls_weight=None, n_cv=3):
    scores = None
    if cv_score:
        # clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01)
        # preformance greater than XGBClassifier
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, max_features='auto', class_weight=cls_weight)

        x_normal_stage = x_train[y_train == 0]
        x_i_stage = x_train[y_train == 1]
        x_ii_stage = x_train[y_train == 2]
        x_iii_stage = x_train[y_train == 3]
        x_iv_stage = x_train[y_train == 4]

        y_normal_stage = y_train[y_train == 0]
        y_i_stage = y_train[y_train == 1]
        y_ii_stage = y_train[y_train == 2]
        y_iii_stage = y_train[y_train == 3]
        y_iv_stage = y_train[y_train == 4]

        x_norm_i_stage = np.vstack((x_normal_stage, x_i_stage))
        y_norm_i_stage = np.hstack((y_normal_stage, y_i_stage))

        skf = StratifiedKFold(n_splits=n_cv)
        auc = []
        f1_score_val = []
        for train_index, test_index in skf.split(x_norm_i_stage, y_norm_i_stage):
            i_x_train = np.vstack((x_norm_i_stage[train_index], x_ii_stage, x_iii_stage, x_iv_stage))
            i_y_train = np.hstack((y_norm_i_stage[train_index], y_ii_stage, y_iii_stage, y_iv_stage))

            i_x_val = np.vstack((x_norm_i_stage[test_index], x_ii_stage, x_iii_stage, x_iv_stage))
            i_y_val = np.hstack((y_norm_i_stage[test_index], y_ii_stage, y_iii_stage, y_iv_stage))

            clf.fit(i_x_train, i_y_train)
            y_pred = clf.predict_proba(i_x_val)
            y_pred_lab = clf.predict(i_x_val)
            auc.append(roc_auc_score(i_y_val, y_pred, average='macro', multi_class='ovo', labels=[0, 1, 2, 3, 4]))
            f1_score_val.append(f1_score(i_y_val, y_pred_lab, average='macro'))

        scores = {'auc': auc, 'f1_score': f1_score_val}

    # clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01)
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, max_features='auto', class_weight=cls_weight)
    clf.fit(x_train, y_train)

    return clf, scores


