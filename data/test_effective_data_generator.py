# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  test_effective_data_generator.py
@Time    :  2021/2/9 0009 16:02
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

from tqdm import tqdm
from args import args
from data_helper import generate_r_positive, is_intersect


class TestEffectiveDataGenerator(object):
    def __init__(self, city_data):
        self.city_data = city_data
        self.regions, self.coordinates = self.generate_query_regions()
        self.test_set = self.generate_test_set()

    def generate_query_regions(self):
        regions = []
        coordinates = []
        for _ in tqdm(range(args.test_n_region)):
            rq_feature, rq_coordinate = self.city_data.generate_region()
            regions.append(rq_feature)
            coordinates.append(rq_coordinate)

        return regions, coordinates

    def generate_test_set(self):
        test_set = []

        for k1, v1 in enumerate(self.regions):
            test_item = []

            # positive sample
            r_pos = generate_r_positive(v1)
            test_item.append(r_pos)

            for k2, v2 in enumerate(self.regions):
                if k1 == k2:
                    continue
                if is_intersect(self.coordinates[k1], self.coordinates[k2]):
                    continue
                # negative samples
                test_item.append(v2)

            test_set.append(test_item)

        return test_set
