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
        self.seed = 981125

        # path
        self.raw_data_dir = "data/dataset/raw/"
        self.preprocessed_data_dir = "data/dataset/preprocessed/"
        self.trained_model_dir = "trained_model/"
        self.train_log_dir = "trained_model/"
        self.test_effective_dir = "test/effective_test/"
        self.test_efficient_dir = "test/efficient_test/"
        self.test_search_dir = "test/search_test/"

        # train data setting
        self.city_for_generate_train = "nanjing"
        self.city_range = [118.740693, 32.091769, 118.830456, 31.965555]
        self.tencent_city_gps_range = [118.740693, 118.830456, 31.965555, 32.091769]
        self.grid_step = [0.0001, 0.0001]  # City size via lon&lat step, 大约对应南京的 10m * 10m
        self.city_dianping_index_order = [1, 2, 14, 17, 18, 11]

        self.train_n_tuples = 20000
        self.train_area_size_range = [60, 200]  # region 大约对应多少个格子的范围
        self.shift_n_ratio = 0.5
        self.positive_noise_rate = 0.1
        self.positive_shift_grid = 5
        self.hard_example_rate = 0.1
        self.negative_noise_rate = 0.5
        # self.negative_shift_rate = 0.5
        self.negative_shift_grid = 20

        self.training_data_generation_batch = 16

        # metric setting
        self.K = 10

        # evaluate setting
        self.evaluate_gap = 100
        self.evaluate_n_region = 50
        self.evaluate_n_comparison = 2048

        # effective test setting
        self.test_effective_region = 2000
        self.test_effective_model_name = "04-10_20-53"

        # efficient test setting
        self.test_efficient_region = 100
        self.test_n_comparison = 2048
        self.test_efficient_model_name = "04-10_20-53"

        # search test setting
        self.test_search_regions = [
            [118.781819, 32.043878, 118.787291, 32.037639],
            [118.915071, 32.117184, 118.934727, 32.105479],
            [118.829498, 32.056973, 118.863144, 32.042059],
            [118.750770, 32.070938, 118.755727, 32.065719],
            [118.816559, 31.934518, 118.821937, 31.928436]
        ]
        self.test_search_model_name = "04-10_20-53"

        # search setting
        self.N = 10

        # model setting
        self.filter_size = [11, 9, 7]
        self.feature_dim = [16, 128, 64, 32]
        self.stride = [2, 2, 2]
        self.dropout_rate = 0.1
        self.delta = 0.3
        self.weight_decay = 0.00005

        # downstream shop placement recommendation setting
        self.downstream_test_model_name = '04-10_20-53'
        self.downstream_test_city = 'nanjing'
        self.downstream_area_step = 40
        self.downstream_train_shops = ["肯德基", "汉堡王"]
        self.downstream_test_shops = ["麦当劳"]
        self.downstream_train_batch_n = 5000
        self.downstream_train_batch_size = 8
        self.downstream_evaluate_frequency = 200
        self.downstream_evaluate_k = 30
        # --------------------直接找出几个区域--------------------
        self.downstream_area_q_coordinate_lef_top = [118.781699, 32.042347]


args = Args()
