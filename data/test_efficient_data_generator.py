# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  test_efficient_data_generator.py
@Time    :  2021/2/20 0020 15:00
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

from args import args


class TestEfficientDataGenerator(object):
    def __init__(self, city_data, test_cnt=args.test_n_region):
        self.city_data = city_data
        self.regions, self.coordinates = self.generate_query_regions(test_cnt)

    def generate_query_regions(self, test_cnt):
        regions = []
        coordinates = []
        for _ in range(test_cnt):
            rq_feature, rq_coordinate = self.city_data.generate_region(copy=False)
            regions.append(rq_feature)
            coordinates.append(rq_coordinate)

        return regions, coordinates
