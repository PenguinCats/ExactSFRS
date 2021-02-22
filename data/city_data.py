# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  city_data.py
@Time    :  2021/2/6 0006 11:38
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import random
import numpy as np
import pandas as pd
from args import args
from data.data_helper import load_category


class CityData(object):
    def __init__(self):
        _, self.grid_size = self.split_grid()
        data, self.n_category = self.load_data()
        self.grid_feature = self.extract_grid_info(data)

    @staticmethod
    def split_grid():
        area_longitude_boundary = np.arange(args.city_range[0], args.city_range[2], args.grid_step[0])
        area_latitude_boundary = np.arange(args.city_range[3], args.city_range[1], args.grid_step[1])

        grid_size = (len(area_longitude_boundary), len(area_latitude_boundary))

        return (area_longitude_boundary, area_latitude_boundary), grid_size

    @staticmethod
    def load_data():
        big_category_dict, n_category = load_category()
        data_path = os.path.join(args.raw_data_dir, args.city_for_generate_train + '.csv')
        data = pd.read_csv(data_path, usecols=args.city_dianping_index_order)

        # format data
        data = data[data['status'] == 0].drop(columns='status')
        data['category'] = data['big_category'].map(lambda x: big_category_dict[x])
        data = data.drop(columns='big_category')

        return data, n_category

    def extract_grid_info(self, data):
        grid_feature = np.zeros([self.n_category, self.grid_size[0], self.grid_size[1]], dtype=np.int32)
        for item in data.itertuples():
            lon_idx = int((item.longitude - args.city_range[0]) // args.grid_step[0])
            lat_idx = int((item.latitude - args.city_range[3]) // args.grid_step[1])
            if 0 <= lon_idx < self.grid_size[0] and 0 <= lat_idx < self.grid_size[1]:
                grid_feature[item.category][lon_idx][lat_idx] += 1

        return grid_feature

    def generate_region(self, copy=False):
        while True:
            # randint 是闭区间
            lon_idx = random.randint(0, self.grid_size[0] - 1)
            lat_idx = random.randint(0, self.grid_size[1] - 1)

            length_max = min(self.grid_size[0] - lon_idx, args.train_area_size_range[1])
            height_max = min(self.grid_size[1] - lat_idx, args.train_area_size_range[1])
            if length_max < args.train_area_size_range[1] or height_max < args.train_area_size_range[1]:
                continue

            # randint 是闭区间
            length = random.randint(args.train_area_size_range[0], length_max)
            height = random.randint(args.train_area_size_range[0], height_max)

            # 切片是左闭右开区间
            rq_feature = self.grid_feature[:, lon_idx:lon_idx+length, lat_idx:lat_idx+height]

            if np.sum(rq_feature) < 50:
                # 区域内 object 太少，重新选择区域
                continue
            else:
                if copy:
                    return rq_feature.copy(), (lon_idx, lat_idx, length, height)
                else:
                    return rq_feature, (lon_idx, lat_idx, length, height)

    @staticmethod
    def get_coordinate_by_index(lef, rig, bot, top):
        lef_lon = args.city_range[0] + lef * args.grid_step[0]
        rig_lon = args.city_range[0] + rig * args.grid_step[0]
        bot_lat = args.city_range[3] + bot * args.grid_step[1]
        top_lat = args.city_range[3] + top * args.grid_step[1]
        return [[top_lat, lef_lon], [bot_lat, lef_lon], [bot_lat, rig_lon], [top_lat, rig_lon]]
