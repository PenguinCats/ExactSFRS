# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  effective_test.py
@Time    :  2021/2/20 0020 11:17
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tool.metrics import hit_ratio_at_K, MRR
from args import args
from tool.log_helper import log_tool_init, logging
from data.city_data import CityData
from data.test_effective_data_generator import TestEffectiveDataGenerator
from Triplet.triplet import global_max_pooling

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # init
    random.seed(args.seed)
    log_tool_init(folder=args.test_effective_dir, no_console=False)
    logging.info(' -- '.join(['%s:%s' % item for item in args.__dict__.items()]))

    # load data
    logging.info("loading city data...")
    city_data = CityData()

    # test generator
    logging.info("building test data generator...")
    evaluate_data_generator = TestEffectiveDataGenerator(city_data, args.test_effective_region)

    # reload model
    logging.info("reloading model...")
    model = torch.load(os.path.join(args.trained_model_dir, "model_{}.pkl".format(args.test_model_name)))

    # test
    logging.info("testing effectiveness...")
    model.eval()
    with torch.no_grad():
        hr_item = []
        mrr_item = []

        v_region = torch.stack([global_max_pooling(model(torch.Tensor(r))) for r in evaluate_data_generator.regions])
        v_pos = torch.stack([global_max_pooling(model(r)) for r in evaluate_data_generator.pos_set])

        dis_pos = torch.sum(nn.functional.mse_loss(v_region, v_pos, reduction='none'), dim=1)

        for idx, v_rq in enumerate(v_region):
            dis_neg = torch.sum(nn.functional.mse_loss(v_rq, v_region[evaluate_data_generator.neg_set_idx[idx]],
                                                       reduction='none'), dim=1)
            distances_total = torch.cat((dis_pos[idx].unsqueeze(0), dis_neg))
            _, sorted_indices = torch.sort(distances_total, descending=False)
            hr_item.append(hit_ratio_at_K(0, sorted_indices, k=args.K))
            mrr_item.append(MRR(0, sorted_indices))

    logging.info("HR@{}: {} | MRR: {}".format(args.K, np.mean(hr_item), np.mean(mrr_item)))

    # draw result
    name_list = ['HR@'.format(args.K), 'MRR']
    num_list = [np.mean(hr_item), np.mean(mrr_item)]
    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
    plt.tight_layout()
    plt.savefig(os.path.join(args.test_effective_dir,
                             'test_effective_result_{}.png'.format(args.test_model_name)))
    plt.show()
