#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: helper.py
    Description:
    
Created by YongBai on 2020/1/20 2:20 PM.
"""

import configparser


def get_config(config_fname=None):
    """

    :param config_fname:
    :return:
    """
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        allow_no_value=True,
        inline_comment_prefixes=('#')
    )

    if config_fname is None:
        import os
        curr_dir = os.path.dirname(__file__)
        config_f = os.path.join(curr_dir, 'config.ini')
    else:
        config_f = config_fname

    with open(config_f, 'r') as config_f:
        config.read_file(config_f)

    return config