#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: loaddata.py
    Description:
    
Created by YongBai on 2020/1/20 2:35 PM.
"""
import argparse
import logging
from preprocess import load_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    process_data_type = args.process_data_type
    x, y, feat_name = load_data(process_data_type)
    logger.info('data generated, x.shape={0},y.shape={1}, n_pos={2}, n_neg={3}, len(features)={4}'.format(
        x.shape, y.shape, len(y[y == 1]), len(y[y == 0]), len(feat_name)
    ))
    logger.info('feat_name :\n{0}'.format(feat_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cancer data preparation')
    parser.add_argument(
        "-t",
        "--process_data_type",
        type=str,
        default="train_data"  # could be train_data or pred_data
    )

    args = parser.parse_args()
    logger.info('args: {}'.format(args))
    main(args)
