# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  args.py
@Time    :  2021/2/6 0006 11:40
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  args.py
@Time    :  2021/2/6 0006 11:19
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""


class Args(object):
    def __init__(self):

        # system environment
        self.seed = 19981125

        # path
        self.raw_data_dir = "./dataset/raw/"
        self.preprocessed_data_dir = "./dataset/preprocessed/"

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


args = Args()
