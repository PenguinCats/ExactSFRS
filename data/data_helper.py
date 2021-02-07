# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  data_helper.py
@Time    :  2021/2/6 0006 15:27
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
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
