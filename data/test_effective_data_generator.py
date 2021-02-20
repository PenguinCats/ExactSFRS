# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  test_effective_data_generator.py
@Time    :  2021/2/9 0009 16:02
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import torch
from args import args
from data.data_helper import generate_r_positive, is_intersect


class TestEffectiveDataGenerator(object):
    def __init__(self, city_data, test_cnt=args.test_n_region):
        self.city_data = city_data
        self.regions, self.coordinates = self.generate_query_regions(test_cnt)
        self.pos_set, self.neg_set_idx = self.generate_test_set()

    def generate_query_regions(self, test_cnt):
        regions = []
        coordinates = []
        for _ in range(test_cnt):
            rq_feature, rq_coordinate = self.city_data.generate_region(copy=False)
            regions.append(rq_feature)
            coordinates.append(rq_coordinate)

        return regions, coordinates

    def generate_test_set(self):
        pos_set = []
        neg_set_idx = []

        for k1, v1 in enumerate(self.regions):
            # positive sample
            r_pos = generate_r_positive(v1.copy())
            pos_set.append(torch.Tensor(r_pos))

            # negative samples
            test_item_idx = []
            for k2, v2 in enumerate(self.regions):
                if k1 == k2:
                    continue
                if is_intersect(self.coordinates[k1], self.coordinates[k2]):
                    continue

                test_item_idx.append(k2)

            neg_set_idx.append(test_item_idx)

        return pos_set, neg_set_idx
