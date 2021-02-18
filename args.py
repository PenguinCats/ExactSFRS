# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  args.py
@Time    :  2021/2/6 0006 11:40
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""


class Args(object):
    def __init__(self):

        # system environment
        self.seed = 19981125

        # path
        self.raw_data_dir = "data/dataset/raw/"
        self.preprocessed_data_dir = "data/dataset/preprocessed/"

        # train data setting
        self.city_for_generate_train = "nanjing"
        self.city_range = [118.668107, 32.16532, 118.970461, 31.872734]
        self.grid_step = [0.0001, 0.0001]  # City size via lon&lat step, 大约对应南京的 10m * 10m
        self.city_dianping_index_order = [1, 14, 17, 18]

        self.train_n_tuples = 10000
        self.train_area_size_range = [60, 300]  # region 大约对应多少个格子的范围
        self.positive_noise_rate = 0.1
        self.positive_shift_rate = 0.3
        self.hard_example_rate = 0.1
        self.negative_noise_rate = 0.5
        self.negative_shift_rate = 0.7

        self.training_data_generation_batch = 64

        # test data setting
        self.test_n_region = 2000

        # model setting
        self.filter_size = [11, 9, 7]
        self.feature_dim = [16, 128, 64, 32]
        self.stride = [2, 2, 2]
        self.dropout_rate = 0.1
        self.delta = 0.3
        self.weight_decay = 0.00005


args = Args()
