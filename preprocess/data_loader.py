#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_loader.py
    Description:
    
Created by YongBai on 2020/1/20 2:24 PM.
"""
from .imports import *
import numpy as np
import pandas as pd


def get_selected_tss_fromfile(in_tss_featues_data_file, save_feature_file=None):
    """
    This is the first step to get the selected TSS features.
    This is tmp function and will be replaced in the future if necessary.
    :param in_tss_featues_data_file:
    :param save_feature_file:
    :return:
    """
    tss_cov_df = pd.read_csv(in_tss_featues_data_file, sep='\t', index_col=False)
    cols = list(tss_cov_df.columns)
    sel_col = cols[2:]
    if save_feature_file is not None:
        feats_str = ','.join(sel_col)
        with open(save_feature_file, 'w') as f:
            f.write(feats_str)
    return sel_col


def get_sampleids_fromfile(in_csv_file, save_sampleid_file=None):
    tss_cov_df = pd.read_csv(in_csv_file, sep='\t', index_col=False)
    sampleids = tss_cov_df.iloc[:, 0].values.tolist()
    if save_sampleid_file is not None:
        samples_str = ','.join(sampleids)
        with open(save_sampleid_file, 'w') as f:
            f.write(samples_str)
    return sampleids


def get_tss_cov(config_section, labels=['LUAD', 'benign'],
                tss_sel_feature_lst=None,
                sample_sel_lst=None):
    """

    :param config_section:
    :param labels: sample type, return all sample types if None.
    :param tss_sel_feature_lst: TSS selected features. return all features if None.
    :param sample_sel_lst: selected sample ids, return all samples if None
    :return: selected TSS gene feature for the given type of samples
    """

    tss_cov_fname = get_config()['paths'][config_section]
    common_cols = get_config()['common_cols']['comm_cols'].split(',')

    tss_cov_df = pd.read_csv(tss_cov_fname, sep='\t', index_col=False)
    old_col_names = list(tss_cov_df.columns)
    new_col_names = old_col_names.copy()
    new_col_names[0] = common_cols[0]
    tss_cov_df.rename(columns=dict(zip(old_col_names, new_col_names)), inplace=True)
    tss_cov_df[common_cols[0]] = tss_cov_df[common_cols[0]].astype(str)

    if tss_sel_feature_lst is not None:
        use_cols = common_cols + tss_sel_feature_lst
        tss_cov_df = tss_cov_df[use_cols]

    if sample_sel_lst is not None:
        tss_cov_df = tss_cov_df.loc[tss_cov_df[common_cols[0]].isin(sample_sel_lst)]

    if labels is not None:
        tss_cov_df = tss_cov_df.loc[tss_cov_df[common_cols[1]].isin(labels)]

    logging.info(tss_cov_df.shape)

    return tss_cov_df.reset_index(drop=True)


def get_extra_featues(labels=['LUAD', 'benign']):

    extra_feat_fanme = get_config()['paths']['extra_feature_path']
    extra_feat_names = get_config()['extra_feat_list']['extra_feats'].split(',')
    common_cols = get_config()['common_cols']['comm_cols'].split(',')
    extra_feat_df = pd.read_csv(extra_feat_fanme, sep='\t', index_col=False)
    if labels is not None:
        extra_feat_df = extra_feat_df.loc[extra_feat_df[common_cols[1]].isin(labels)].reset_index(drop=True)

    extra_feat_df = extra_feat_df[common_cols + extra_feat_names]
    return extra_feat_df


def get_features(in_tss_conf_section, tss_sel_feature_lst=None,
                 sample_sel_lst=None, labels=['LUAD', 'benign'], save_path=None):
    """
    157 * 40，
    this is for model train
    :param in_tss_conf_section:
    :param tss_sel_feature_lst:
    :param sample_sel_lst:
    :param labels:
    :return:
    """

    common_cols = get_config()['common_cols']['comm_cols'].split(',')

    tss_cov_df = get_tss_cov(in_tss_conf_section, labels=labels,
                             tss_sel_feature_lst=tss_sel_feature_lst,
                             sample_sel_lst=sample_sel_lst)

    extra_feat_df = get_extra_featues(labels=labels)

    tmp_re_df = pd.merge(tss_cov_df, extra_feat_df, on=common_cols)
    tmp_re_df['label'] = 0
    pos_labels = get_config()['common_cols']['pos_labels'].split(',')

    tmp_re_df.loc[tmp_re_df[common_cols[1]].isin(pos_labels), 'label'] = 1

    stage_df = pd.read_csv(get_config()['paths']['sample_stage_path'], sep="\s+", usecols=['sample', 'stage'])

    stage_df['stage_val'] = 0
    stage_df.loc[stage_df['stage'].isin(['IA', 'IB']), 'stage_val'] = 1
    stage_df.loc[stage_df['stage'].isin(['IIA', 'IIB']), 'stage_val'] = 2
    stage_df.loc[stage_df['stage'].isin(['IIIA', 'IIIB']), 'stage_val'] = 3
    stage_df.loc[stage_df['stage'].isin(['IVA', 'IVB']), 'stage_val'] = 4

    tmp_re_stage_df = pd.merge(tmp_re_df, stage_df, on=common_cols[0])

    cols_names = [x for x in tmp_re_stage_df.columns if x not in common_cols]
    re = tmp_re_stage_df[cols_names]

    if save_path is not None:
        if os.path.exists(save_path):
            os.remove(save_path)
        re.to_csv(save_path, index=False, sep='\t')

    return re


def load_data(data_type):

    conf = get_config()
    output_conf = conf['out_dir']
    neg_labels = conf['common_cols']['neg_labels'].split(',')
    pos_labels = conf['common_cols']['pos_labels'].split(',')
    labels = pos_labels + neg_labels

    if data_type == 'train_data':
        out_data_conf_section = 'train_test_data_fname'
        in_tss_conf_section = 'tss_cov_2k_30_path'
        tss_sel_feature_lst = None
        sample_sel_lst = None
        labels = [pos_labels[0], neg_labels[0]]  # NOTE: this following the previous version of the GROUPs
        # comment out is would have 284 samples
    else:
        out_data_conf_section = 'unseen_pred_data_fname'
        in_tss_conf_section = 'tss_cov_2k_all_path'

        # first: get the tss feature lists
        logging.info('loading tss selected features/columns from the given data file...')
        feat_name_f = os.path.join(output_conf['out_root_dir'], output_conf['tss_feature_fname'])
        if not os.path.exists(feat_name_f):
            tss_cov_fname = conf['paths']['tss_cov_2k_30_path']
            tss_sel_feature_lst = get_selected_tss_fromfile(tss_cov_fname, save_feature_file=feat_name_f)
        else:
            with open(feat_name_f, 'r') as f:
                tss_sel_feats_str = f.read()
                tss_sel_feature_lst = tss_sel_feats_str.split(',')
        logging.info('Total number of the tss selected features/columns: {}'.format(len(tss_sel_feature_lst)))

        # second: get unseen prediction sample id
        logging.info('loading unseen prediction sample ids...')
        unseen_sampleids_f = os.path.join(output_conf['out_root_dir'], output_conf['unseen_pred_sampleids'])
        if not os.path.exists(unseen_sampleids_f):
            sample_sel_lst = get_sampleids_fromfile(conf['paths']['unseen_pred_tss_path2_sampleid_75'],
                                                    save_sampleid_file=unseen_sampleids_f)
        else:
            with open(unseen_sampleids_f, 'r') as f:
                sampleids_str = f.read()
                sample_sel_lst = sampleids_str.split(',')

        logging.info('Total number of the unseen samples: {}'.format(len(sample_sel_lst)))

    logging.info('generating data...')
    feat_fn = os.path.join(output_conf['out_root_dir'], output_conf[out_data_conf_section])
    if not os.path.exists(feat_fn):
        feat_df = get_features(in_tss_conf_section, tss_sel_feature_lst=tss_sel_feature_lst,
                               sample_sel_lst=sample_sel_lst, labels=labels, save_path=feat_fn)
    else:
        feat_df = pd.read_csv(feat_fn, sep='\t')

    assert feat_df is not None

    y_stage_str = feat_df.pop('stage').values
    y_stage_val = feat_df.pop('stage_val').values
    y_label = feat_df.pop('label').values

    feat_name = feat_df.columns
    x = feat_df.values

    return x, y_label, y_stage_str, y_stage_val, np.array(feat_name)


