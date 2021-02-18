# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  train_data_generator.py
@Time    :  2021/2/7 0007 12:02
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import random
import torch
import numpy as np
from data.data_helper import is_intersect, generate_r_positive
from args import args


class TrainDataGenerator(object):
    def __init__(self, city_data):
        self.city_data = city_data
        self.generate_tuples()

    def generate_tuples(self, batch_size=64):
        training_tuples = [[], [], []]
        training_coordinates = []

        # a region is represented by a point (x, y) which is the upper-left block index and the length/height
        for _ in range(batch_size):
            # 生成区域
            rq_feature, rq_coordinate = self.city_data.generate_region()

            # 生成正负样例
            r_pos = generate_r_positive(rq_feature.copy())
            if np.random.rand() > args.hard_example_rate:
                r_neg = self.generate_r_negative_simple(rq_coordinate)
            else:
                r_neg = self.generate_r_negative_hard(rq_feature.copy())

            training_tuples[0].append(torch.Tensor(rq_feature))
            training_tuples[1].append(torch.Tensor(r_pos))
            training_tuples[2].append(torch.Tensor(r_neg))
            training_coordinates.append(rq_coordinate)

        return training_tuples, training_coordinates

    @staticmethod
    def generate_r_negative_hard(rq_feature):
        total_objects = np.sum(rq_feature)
        n_noise_object = int(total_objects * args.negative_noise_rate)
        n_shift_object = int(total_objects * args.negative_shift_rate)

        a, b, c = np.where(rq_feature > 0)
        coordinates = [[a[idx], b[idx], c[idx]]
                       for idx in range(len(a))
                       for _ in range(rq_feature[a[idx]][b[idx]][c[idx]])]

        # random delete object
        objects_to_delete = random.sample(coordinates, n_noise_object)
        for obj in objects_to_delete:
            rq_feature[obj[0], obj[1], obj[2]] -= 1

        # random shift object
        objects_to_shift = random.sample(coordinates, n_shift_object)
        for obj in objects_to_shift:
            na = np.random.randint(-obj[0], rq_feature.shape[0] - obj[0] - 1)
            nb = np.random.randint(-obj[1], rq_feature.shape[1] - obj[1] - 1)
            nc = np.random.randint(-obj[2], rq_feature.shape[2] - obj[2] - 1)

            rq_feature[obj[0], obj[1], obj[2]] -= 1
            rq_feature[na, nb, nc] += 1

        # random add object
        a = np.random.choice(range(rq_feature.shape[0]), size=n_noise_object)
        b = np.random.choice(range(rq_feature.shape[1]), size=n_noise_object)
        c = np.random.choice(range(rq_feature.shape[2]), size=n_noise_object)
        rq_feature[a, b, c] += 1

        return rq_feature

    def generate_r_negative_simple(self, rq_coordinate):
        while True:
            r_neg, r_neg_coordinate = self.city_data.generate_region()

            if is_intersect(rq_coordinate, r_neg_coordinate):
                continue
            else:
                return r_neg
