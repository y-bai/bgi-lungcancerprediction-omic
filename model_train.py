#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: model_train.py
    Description:
    
Created by YongBai on 2020/2/13 10:33 PM.
"""
import os
import numpy as np
import pandas as pd
import argparse
import logging
from general_utils.helper import get_config
from preprocess import load_data, data_split_transform, data_whole_transform, feature_sel
from model import model_train_run, get_class_weight, single_model_train_run
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def model_split_train():

    conf_outmodelpath = get_config()['model_path']
    out_model_dir = conf_outmodelpath['out_model_dir']

    logger.info('loading training data...')
    model_split_train_test_data = os.path.join(out_model_dir, conf_outmodelpath['model_split_train_test_data'])
    if os.path.exists(model_split_train_test_data):
        tt_data = np.load(model_split_train_test_data)
        x_train, y_train, x_test, y_test, feat_name = \
            tt_data['x_train'], tt_data['y_train'], tt_data['x_test'], tt_data['y_test'], tt_data['feat_name']
    else:
        x, y_label, _, y_stage_val, feat_name = load_data('train_data')  # HARD CODE here

        # end = OneHotEncoder()
        # y_stage_val_end = end.fit_transform(y_stage_val.reshape(-1, 1))

        # tarining data splitting and transformation
        x_train, y_train, x_test, y_test = data_split_transform(x, y_stage_val, has_stage=True)
        np.savez(model_split_train_test_data,
                 x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, feat_name=feat_name)

    logger.info('selecting features by model...')
    supp, x_train_new, y_train_new = feature_sel(x_train,
                                                 y_train,
                                                 k=15,
                                                 estimator=ExtraTreesClassifier(n_estimators=40, max_depth=5,
                                                                                max_features='auto'),
                                                 sel_method='estimator')

    model_sel_feature_re_fname = os.path.join(out_model_dir,
                                              conf_outmodelpath['model_split_select_feature_results'])
    logger.info('saving the selected features...')
    if os.path.exists(model_sel_feature_re_fname):
        os.remove(model_sel_feature_re_fname)
    np.savez(model_sel_feature_re_fname, original_feat=feat_name, supp=supp)

    logger.info('training split_model...')
    cls_weight = get_class_weight(y_train_new)
    # clf, scores = model_train_run(x_train_new, y_train_new, cls_weight=cls_weight)

    clf, scores = single_model_train_run(x_train_new, y_train_new, cv_score=True, cls_weight=cls_weight)

    logger.info('saving split_model...')
    model_split_model_dump_file = os.path.join(out_model_dir,
                                               conf_outmodelpath['model_split_model_dump_file'])
    if os.path.exists(model_split_model_dump_file):
        os.remove(model_split_model_dump_file)
    joblib.dump(clf, model_split_model_dump_file)
    logger.info('saving the cross-validation score...')
    if scores is not None:
        model_split_cv_score_results = os.path.join(out_model_dir,
                                                    conf_outmodelpath["model_split_cv_score_results"])
        pd.DataFrame.from_dict(scores).to_csv(model_split_cv_score_results, sep='\t', index=False)

    logger.info('for model evalution, see corresponding .ipynb file')
    logger.info('DONE--training model based on the split data(training data and testing data)')


def model_whole_train():
    conf_outmodelpath = get_config()['model_path']
    out_model_dir = conf_outmodelpath['out_model_dir']

    logger.info('loading whole training data (without split)...')
    model_whole_train_test_data = os.path.join(out_model_dir, conf_outmodelpath['model_whole_train_test_data'])
    if os.path.exists(model_whole_train_test_data):
        tt_data = np.load(model_whole_train_test_data)
        x_train, y_train, feat_name = tt_data['x_train'], tt_data['y_train'], tt_data['feat_name']
    else:
        x, y_label, _, y_stage_val, feat_name = load_data('train_data')  # HARD CODE here

        # tarining data splitting and transformation
        x_train, y_train = data_whole_transform(x, y_stage_val)
        np.savez(model_whole_train_test_data,
                 x_train=x_train, y_train=y_train, feat_name=feat_name)

    logger.info('selecting features by model...')
    supp, x_train_new, y_train_new = feature_sel(x_train,
                                                 y_train,
                                                 k=15,
                                                 estimator=ExtraTreesClassifier(n_estimators=40, max_depth=5,
                                                                                max_features='auto'),
                                                 sel_method='estimator')

    model_sel_feature_re_fname = os.path.join(out_model_dir,
                                              conf_outmodelpath['model_whole_select_feature_results'])
    logger.info('saving the selected features...')
    if os.path.exists(model_sel_feature_re_fname):
        os.remove(model_sel_feature_re_fname)
    np.savez(model_sel_feature_re_fname, original_feat=feat_name, supp=supp)

    logger.info('training whole model...')
    cls_weight = get_class_weight(y_train_new)
    # clf, _ = model_train_run(x_train_new, y_train_new, cls_weight=cls_weight, cv_score=False)

    clf, _ = single_model_train_run(x_train_new, y_train_new, cv_score=False, cls_weight=cls_weight)
    y_pred = clf.predict_proba(x_train_new)

    risk = np.dot(y_pred, np.array([[0], [1], [4], [8], [10]]))

    logger.info(clf.classes_)
    logger.info(list(zip(y_train_new, y_pred, risk.flatten())))

    logger.info('saving whole model...')
    model_whole_model_dump_file = os.path.join(out_model_dir,
                                               conf_outmodelpath['model_whole_model_dump_file'])
    if os.path.exists(model_whole_model_dump_file):
        os.remove(model_whole_model_dump_file)
    joblib.dump(clf, model_whole_model_dump_file)

    logger.info('for model evalution, see corresponding .ipynb file')
    logger.info('DONE--training model based on the whole data')


def main(args):
    train_type = args.train_type
    if train_type == 'split_train':
        model_split_train()
    else:
        model_whole_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cancer data preparation')
    parser.add_argument(
        "-t",
        "--train_type",
        type=str,
        default="split_train"  # could be split_train or whole_train
    )

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)

    