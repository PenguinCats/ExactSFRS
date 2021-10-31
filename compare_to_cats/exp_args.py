#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : exp_args.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/8
# @Desc  : None

class ExperimentArgs:
    def __init__(self):
        self.seed = 19971007
        self.lr = 5e-4
        self.weight_decay = 0.00005
        self.delta_for_loss = 0.0
        self.lambda_for_loss_fuse = 0.3
        self.lambda_for_test_fuse = 0.3

        self.train_epoch = 20
        self.batch_per_epoch = 1000
        self.batch_size = 10
        self.print_iter_frequency = 200

        self.n_test_samples = 50
        self.n_test_negative_size = 100
        self.n_test_K = 10

        self.data_path = 'data/data_file'

        # the following configs are for data generation
        # data generation --> dg
        self.dg_length_and_width_range = [10, 20]  # the number of gird
        self.dg_noise_range = [0, 1]
        self.dg_positive_modify_ratio = 0.15
        self.dg_negative_modify_ratio = 0.35
        self.dg_unselected_modify_ratio = 0.5
        self.dg_negative_choice_probability = 0.4
        self.dg_empty_area_reserved_probability = 0
        self.dg_modify_unselected_feature_ratio_rate = 1.0
        self.dg_cos_sim_threshold = 1.0
        self.dg_modify_inner_circle_times = 1

        # the following configs are for real world test
        self.rwt_selected_area_coordinate = [
            [-73.99614949927026, 40.68384432116034],
            [-73.94832596036136, 40.65413946909123],
            [-73.91635682499665, 40.615842047192146],
            [-73.99623167386585, 40.744621740014885]]
        # self.rwt_selected_feature = [['餐饮'], ['餐饮'], ['餐饮'], ['餐饮'], ['餐饮'], ['餐饮', '购物'], ['购物']]
        # self.rwt_concern_checkin = [True, True, True, True, True, True, True]
        self.rwt_selected_feature = [['文娱', '购物'], ['便民服务', '住宅区'], ['文娱'], ['便民服务', '住宅区'], ['医疗', '购物'], ['餐饮'],
                                     ['便民服务', '住宅区'], ['医疗', '购物'], ['住宅区'], ['餐饮'], ['餐饮'], ['餐饮'], ['餐饮'],
                                     ['餐饮', '购物'], ['购物']]
        self.rwt_concern_checkin = [True, False, True, False, True, True, False, True, False, True, True, True, True,
                                    True, True]
        self.rwt_dg_length_and_width = [5, 5]
        self.rwt_model_start_time = [2, 2]
        self.rwt_model_start_time = "2021-07-18_23-13"

        # exp args
        self.trained_result = 'experiment/v1_NYC'


exp_args = ExperimentArgs()
