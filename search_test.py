# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  search_test.py
@Time    :  2021/2/25 0025 12:05
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import random
import torch
from tqdm import tqdm
from args import args
from tool.log_helper import log_tool_init, logging
from data.city_data import CityData
from data.test_search_data_generator import TestSearchDataGenerator
from ExactSFRS.exact_sfrs import ExactSFRS
from tool.draw_map import draw_search_result_by_search_result

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # init
    random.seed(args.seed)
    log_tool_init(folder=args.test_search_dir, no_console=False, prefix='test_search')
    logging.info(' -- '.join(['%s:%s' % item for item in args.__dict__.items()]))

    # load data
    logging.info("loading city data...")
    city_data = CityData()

    # test generator
    logging.info("building test data generator...")
    search_data_generator = TestSearchDataGenerator(city_data)

    # reload model
    logging.info("reloading model...")
    model = torch.load(os.path.join(args.trained_model_dir, "model_{}.pkl".format(args.test_search_model_name)))

    # test
    logging.info("testing search...")
    model.eval()
    with torch.no_grad():
        search_space_feature = model(torch.Tensor(city_data.grid_feature))

        sfrs = ExactSFRS(search_space_feature)

        for idx, region in tqdm(enumerate(search_data_generator.regions)):
            region_feature = model(torch.Tensor(region))
            search_result = sfrs.search(region_feature)

            rc = search_data_generator.coordinate_idx[idx]
            region_coordinate = city_data.get_coordinate_by_index(rc[0], rc[1], rc[2], rc[3])
            search_result_coordinates = [city_data.get_coordinate_by_index(v[0], v[1], v[2], v[3])
                                         for v in search_result]
            html = draw_search_result_by_search_result(region_coordinate, search_result_coordinates)
            with open(os.path.join(args.test_search_dir,
                                   'test_search_result_{}_{}.html'.format(args.test_search_model_name, idx)), 'w') as f:
                f.write(html)
            logging.info(search_result)
