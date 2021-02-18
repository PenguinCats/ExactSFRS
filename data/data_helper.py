# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  data_helper.py
@Time    :  2021/2/6 0006 15:27
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import random
import numpy as np
import pandas as pd
from args import args


def load_category():
    big_category = pd.read_csv(os.path.join(args.raw_data_dir, 'big_category.csv'))
    big_category_dict = dict()
    for item in big_category.itertuples():
        big_category_dict[item.name] = item.ID
    n_category = len(big_category_dict)

    return big_category_dict, n_category


def is_intersect(rq_coordinate, r_neg_coordinate):
    lon_max = min((rq_coordinate[0] + rq_coordinate[2]), (r_neg_coordinate[0] + r_neg_coordinate[2]))
    lon_min = max(rq_coordinate[0], r_neg_coordinate[0])
    lat_max = min((rq_coordinate[1] + rq_coordinate[3]), (r_neg_coordinate[1] + r_neg_coordinate[3]))
    lat_min = max(rq_coordinate[1], r_neg_coordinate[1])

    if lon_max < lon_min or lat_max < lat_min:
        return False
    else:
        return True


def generate_r_positive(rq_feature):
    total_objects = np.sum(rq_feature)
    n_noise_object = int(total_objects * args.positive_noise_rate)
    n_shift_object = int(total_objects * args.positive_shift_rate)

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
