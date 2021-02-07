# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  train_data_generator.py
@Time    :  2021/2/7 0007 12:02
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import random
import time

from tqdm import tqdm
import numpy as np
from data_helper import is_intersect
from args import args


class TrainDataGenerator(object):
    def __init__(self, city_data):
        self.city_data = city_data
        self.training_tuples, self.training_coordinates = self.generate_tuples()

    def generate_tuples(self):
        training_tuples = []
        training_coordinates = []

        # a region is represented by a point (x, y) which is the upper-left block index and the length/height
        for _ in tqdm(range(args.train_n_tuples)):
            # 生成区域
            rq_feature, rq_coordinate = self.generate_region()

            # 生成正负样例
            r_pos = self.generate_r_positive(rq_feature)
            if np.random.rand() > args.hard_example_rate:
                r_neg = self.generate_r_negative_simple(rq_coordinate)
            else:
                r_neg = self.generate_r_negative_hard(rq_feature)

            training_tuples.append([rq_feature, r_pos, r_neg])
            training_coordinates.append(rq_coordinate)

        return training_tuples, training_coordinates

    def generate_region(self):
        while True:
            # randint 是闭区间
            lon_idx = random.randint(0, self.city_data.grid_size[0] - 1)
            lat_idx = random.randint(0, self.city_data.grid_size[1] - 1)

            length_max = min(self.city_data.grid_size[0] - lon_idx, args.train_area_size_range[1])
            height_max = min(self.city_data.grid_size[1] - lat_idx, args.train_area_size_range[1])
            if length_max < args.train_area_size_range[1] or height_max < args.train_area_size_range[1]:
                continue

            # randint 是闭区间
            length = random.randint(args.train_area_size_range[0], length_max)
            height = random.randint(args.train_area_size_range[0], height_max)

            # 切片是左闭右开区间
            rq_feature = self.city_data.grid_feature[lon_idx:lon_idx+length, lat_idx:lat_idx+height]

            if np.sum(rq_feature) < 50:
                # 区域内 object 太少，重新选择区域
                continue
            else:
                return rq_feature, (lon_idx, lat_idx, length, height)

    @staticmethod
    def generate_r_positive(rq_feature):
        total_objects = np.sum(rq_feature)
        n_noise_object = int(total_objects * args.positive_noise_rate)
        n_shift_object = int(total_objects * args.positive_shift_rate)

        a, b, c = np.where(rq_feature > 0)
        coordinates = [[a[idx], b[idx], c[idx]]*rq_feature[a[idx]][b[idx]][c[idx]] for idx in range(len(a))]
        # random delete object
        objects_to_delete = random.sample(coordinates, n_noise_object)

        for obj in objects_to_delete:
            rq_feature[obj[0], obj[1], obj[2]] -= 1

        # random shift object
        objects_to_shift = random.sample(coordinates, n_shift_object)
        for obj in objects_to_shift:
            na = np.random.randint(-obj[0], rq_feature.shape[0]-obj[0]-1)
            nb = np.random.randint(-obj[1], rq_feature.shape[1]-obj[1]-1)
            nc = np.random.randint(-obj[2], rq_feature.shape[2]-obj[2]-1)

            rq_feature[obj[0], obj[1], obj[2]] -= 1
            rq_feature[na, nb, nc] += 1

        # random add object
        a = np.random.choice(range(rq_feature.shape[0]), size=n_noise_object)
        b = np.random.choice(range(rq_feature.shape[1]), size=n_noise_object)
        c = np.random.choice(range(rq_feature.shape[2]), size=n_noise_object)
        rq_feature[a, b, c] += 1

        return rq_feature

    @staticmethod
    def generate_r_negative_hard(rq_feature):
        total_objects = np.sum(rq_feature)
        n_noise_object = int(total_objects * args.negative_noise_rate)
        n_shift_object = int(total_objects * args.negative_shift_rate)

        a, b, c = np.where(rq_feature > 0)
        coordinates = [[a[idx], b[idx], c[idx]]*rq_feature[a[idx]][b[idx]][c[idx]] for idx in range(len(a))]

        # random delete object
        objects_to_delete = random.sample(coordinates, n_noise_object)
        for obj in objects_to_delete:
            rq_feature[obj[0], obj[1], obj[2]] -= 1

        # random shift object
        objects_to_shift = random.sample(coordinates, n_shift_object)
        for obj in objects_to_shift:
            na = np.random.randint(-obj[0], rq_feature.shape[0]-obj[0]-1)
            nb = np.random.randint(-obj[1], rq_feature.shape[1]-obj[1]-1)
            nc = np.random.randint(-obj[2], rq_feature.shape[2]-obj[2]-1)

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
            r_neg, r_neg_coordinate = self.generate_region()

            if is_intersect(rq_coordinate, r_neg_coordinate):
                continue
            else:
                return r_neg
