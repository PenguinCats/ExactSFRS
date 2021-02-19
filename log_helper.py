# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  log_helper.py
@Time    :  2021/2/19 0019 14:18
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import re
import os
import logging


def ensure_dir_exist(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def create_log_id(dir_path):
    ensure_dir_exist(dir_path)
    dir_list = os.listdir(dir_path)
    log_list = [item for item in dir_list if (re.match(r"log(.*)\.log", item))]
    log_save_id = len(log_list)
    return log_save_id


def log_tool_init(prefix='train',
                  folder=None,
                  level=logging.DEBUG,
                  console_level=logging.DEBUG,
                  no_console=True):

    # clear handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    # define log dir path & log file path
    log_dir_path = os.path.join(folder, 'log/')
    ensure_dir_exist(log_dir_path)
    log_id = create_log_id(log_dir_path)
    log_path = os.path.join(log_dir_path, "log_{}_{}.log".format(log_id, prefix))

    # make log handler
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)

    logging.getLogger('matplotlib.font_manager').disabled = True
