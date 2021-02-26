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
from tool.log_helper import log_tool_init, logging
from data.city_data import CityData
from data.test_efficient_data_generator import TestEfficientDataGenerator
from ExactSFRS.exact_sfrs import ExactSFRS
from tool.draw_map import draw_search_result_by_search_result

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # init
    random.seed(args.seed)
    log_tool_init(folder=args.test_efficient_dir, no_console=False, prefix='test_efficient')
    logging.info(' -- '.join(['%s:%s' % item for item in args.__dict__.items()]))

    # load data
    logging.info("loading city data...")
    city_data = CityData()

    # test generator
    logging.info("building test data generator...")
    evaluate_data_generator = TestEfficientDataGenerator(city_data, args.test_efficient_region)

    # reload model
    logging.info("reloading model...")
    model = torch.load(os.path.join(args.trained_model_dir, "model_{}.pkl".format(args.test_efficient_model_name)))

    # test
    logging.info("testing efficiency...")
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

            # draw map
            if idx < 20:
                rc = evaluate_data_generator.coordinates[idx]
                region_coordinate = city_data.get_coordinate_by_index(rc[0], rc[0] + rc[2], rc[1], rc[1] + rc[3])
                search_result_coordinates = [city_data.get_coordinate_by_index(v[0], v[1], v[2], v[3])
                                             for v in search_result]
                html = draw_search_result_by_search_result(region_coordinate, search_result_coordinates)
                with open(os.path.join(args.test_efficient_dir,
                                       'test_efficient_result_{}_{}.html'.format(args.test_model_name, idx)), 'w') as f:
                    f.write(html)

            logging.info(search_result)

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
    plt.tight_layout()
    plt.savefig(os.path.join(args.test_efficient_dir,
                             'test_efficient_result_{}.png'.format(args.test_model_name)))
    plt.show()
