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
        self.raw_data_dir = "../data/raw_data"
        self.preprocessed_data_dir = "../data/preprocessed_data"
        self.category_dir = "../data/category"
        self.save_dir = "trained_model/"

        # data setting
        self.city_list = ["Nanjing", "Hangzhou"]
        self.city_range = [[118.715099, 32.148505, 118.818182, 31.986015],
                           [120.063268, 30.336489, 120.282994, 30.192375]]
        self.city_size = [0.005, 0.005]  # City size via lon&lat step
        self.city_dianping_index_order = [[0, 1, 2, 7, 14, 15, 17, 18, 23], [0, 1, 2, 9, 16, 17, 19, 20, 26]]

        # model info
        self.target_city = "Hangzhou"
        self.target_city_id = self.city_list.index(self.target_city)

        # self.intentionally_ignored_cate_list = ["火锅", "四川火锅", "重庆火锅"]
        # self.intentionally_ignored_shop_name = []
        # self.small_cate_for_fitting_list = ["川菜家常菜", "串串香"]
        self.intentionally_ignored_cate_list = []
        self.intentionally_ignored_shop_name = ["盒马鲜生"]
        self.small_cate_for_fitting_list = ["生鲜", "超市/便利店"]
        self.level_plus = [3, 3]

        self.lr = 0.001
        self.weight_decay = 0.01
        self.n_epoch = 50

        self.use_category_ontology_diagram = True
        self.use_multi_level_category = True

        # test setting
        """
        0: 单类型数据中的指定类型作为 test data
        1: 多类型数据中的指定店名部分 test data
        2: 单类型数据中的指定店名部分 test data
        """
        self.test_data_mode = 2
        self.n_neg_grid = 60

        # self.test_target_type_list = ["火锅", "四川火锅", "重庆火锅"]  # 指定类型

        # self.test_target_type_list = []  # 指定类型
        # self.test_file_target_shop_name = ["盒马鲜生"]  # 多类型数据中，哪些店名测试数据

        # self.test_origin_target_shop_name = ["海底捞火锅"]  # 单类型数据中，哪些店名测试数据
        self.test_origin_target_shop_name = ["盒马鲜生"]  # 单类型数据中，哪些店名测试数据

        self.K_list = [6, 10, 15]

        # Graph and propagation layers setting
        self.embedding_size = 64
        self.city_bias_size = 4
        self.relation_dim = 64
        self.conv_dim_list = [64, 16, 8]
        # self.dropout_for_conv = [0.1, 0.1, 0.1]
        self.dropout_for_conv = [0.5, 0.5, 0.5]
        self.ncf_dim_list = [32, 16, 1]
        self.dropout_for_ncf = [0.5, 0.5, 0.5]
        self.kg_transR_batch_size = 64
        self.NCF_batch_size = 64


args = Args()