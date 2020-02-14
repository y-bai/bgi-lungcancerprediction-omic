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
                                 final_estimator=LogisticRegression(class_weight=cls_weight))  # 0.5
        scores = cross_validate(clf, x_train, y_train,
                                cv=5, n_jobs=n_jobs,
                                verbose=1, scoring=['f1', 'recall', 'precision'],
                                return_train_score=True)

    f_clf = StackingClassifier(estimators=clfs, final_estimator=LogisticRegression(class_weight=cls_weight), cv=5)
    f_clf.fit(x_train, y_train)
    return f_clf, scores


