# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  efficient_test.py
@Time    :  2021/2/21 0021 18:19
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from args import args
from log_helper import log_tool_init, logging
from data.city_data import CityData
from data.test_efficient_data_generator import TestEfficientDataGenerator
from ExactSFRS.exact_sfrs import ExactSFRS

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
    evaluate_data_generator = TestEfficientDataGenerator(city_data, args.test_n_region)

    # reload model
    logging.info("reloading model...")
    model = torch.load(os.path.join(args.trained_model_dir, "model_{}.pkl".format(args.test_model_name)))

    # test
    logging.info("testing...")
    model.eval()
    with torch.no_grad():
        search_space_feature = model(torch.Tensor(city_data.grid_feature))

        time_consume = []
        sfrs = ExactSFRS(search_space_feature)

        for idx, region in tqdm(enumerate(evaluate_data_generator.regions)):
            region_feature = model(torch.Tensor(region))
            single_search_start_time = time()
            search_result = sfrs.search(region_feature)
            single_search_finish_time = time()
            single_search_time_consume = single_search_finish_time - single_search_start_time
            time_consume.append(single_search_time_consume)

        # draw train result
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xlabel("query ID")
        plt.ylabel("single query runtime")
        plt.plot(range(len(time_consume)), time_consume)
        plt.subplot(1, 2, 2)
        plt.xlabel("Exact SFRS")
        plt.ylabel("average runtime")
        plt.bar(["Exact SFRS"], np.mean(time_consume))
        plt.savefig(os.path.join(args.test_efficient_dir,
                                 'test_efficient_result_{}.png'.format(args.test_model_name)))
        plt.show()
