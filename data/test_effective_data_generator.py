# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  test_effective_data_generator.py
@Time    :  2021/2/9 0009 16:02
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import random
import torch
from args import args
from data.data_helper import generate_r_positive, is_intersect


class TestEffectiveDataGenerator(object):
    def __init__(self, city_data, test_cnt=args.test_effective_region, test_comparisons_cnt=args.test_n_comparison):
        self.test_cnt = test_cnt
        self.test_comparisons_cnt = test_comparisons_cnt
        self.city_data = city_data
        self.regions, self.coordinates = self.generate_query_regions()
        self.v_q_ids, self.pos_set, self.neg_set_idx = self.generate_test_set()

    def generate_query_regions(self):
        regions = []
        coordinates = []
        for _ in range(self.test_comparisons_cnt):
            rq_feature, rq_coordinate = self.city_data.generate_region(copy=False)
            regions.append(rq_feature)
            coordinates.append(rq_coordinate)

        return regions, coordinates

    def generate_test_set(self):
        candidate_ids = range(len(self.regions))
        v_q_ids = random.sample(candidate_ids, self.test_cnt)

        pos_set = []
        neg_set_idx = []

        for k1 in v_q_ids:
            v1 = self.regions[k1]
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

        return v_q_ids, pos_set, neg_set_idx
