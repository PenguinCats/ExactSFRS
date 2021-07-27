#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : feature_extractor.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from compare_to_cats.data.data_args import data_args


class FeatureExtractor:
    def __init__(self, n_grid, n_lon_len, n_lat_len):
        self.n_grid, self.n_lon_len, self.n_lat_len = n_grid, n_lon_len, n_lat_len
        # POI data
        self.POI_label_encoder = LabelEncoder()
        POI_data, n_POI_cate, self.POI_cate2ID = self.load_POI_data()
        self.n_POI_cate = n_POI_cate
        POI_feature_cate, POI_feature_cate_ratio = self.generate_POI_feature(POI_data, n_POI_cate)

        self.feature = np.concatenate((POI_feature_cate, POI_feature_cate_ratio), axis=1)

    def get_grid_id(self, lon, lat):
        lon_idx = (lon - data_args.area_coordinate[0]) // data_args.grid_size_longitude_degree
        lat_idx = (lat - data_args.area_coordinate[2]) // data_args.grid_size_latitude_degree
        if not(0 <= lon_idx < self.n_lon_len and 0 <= lat_idx < self.n_lat_len):
            return -1, -1
        return int(lon_idx), int(lat_idx)

    def load_POI_data(self):
        POI_data = pd.read_csv(data_args.POI_path, usecols=[0, 2, 3, 7, 12, 13, 14, 15, 16],
                               dtype={
                                   'category id 0': int,
                                   'category id 1': str,
                                   'category id 2': str,
                                   'category id 3': str,
                                   'category id 4': str,
                               })
        POI_data = POI_data[POI_data['category id 0'] > 0]
        POI_data.fillna('nan')

        category_list_1 = POI_data['category id 1'].values.tolist()
        category_list_2 = POI_data['category id 2'].values.tolist()
        category_list_3 = POI_data['category id 3'].values.tolist()
        category_list_4 = POI_data['category id 4'].values.tolist()
        cate_list = list(set(category_list_1 + category_list_2 + category_list_3 + category_list_4) - set(list(['nanan'])))
        cate_list.append('nanan')

        self.POI_label_encoder.fit(cate_list)
        cate_corresponding_id = self.POI_label_encoder.transform(cate_list)
        POI_cate2ID = {}
        for idx, v in enumerate(cate_list):
            POI_cate2ID[v] = cate_corresponding_id[idx]
        POI_cate2ID.pop('nanan')

        POI_data['category id 1'] = self.POI_label_encoder.transform(category_list_1)
        POI_data['category id 2'] = self.POI_label_encoder.transform(category_list_2)
        POI_data['category id 3'] = self.POI_label_encoder.transform(category_list_3)
        POI_data['category id 4'] = self.POI_label_encoder.transform(category_list_4)

        return POI_data, len(cate_list) - 1, POI_cate2ID

    def generate_POI_feature(self, POI_data, n_POI_cate):
        POI_feature_cate = np.zeros((self.n_grid, n_POI_cate))
        POI_feature_cate_ratio = np.zeros((self.n_grid, n_POI_cate))
        POI_feature_handed = np.zeros((self.n_grid, 4))

        for val in POI_data.itertuples():
            s_lon_idx, s_lat_idx = self.get_grid_id(val.longitude, val.latitude)
            if not (s_lon_idx == -1 or s_lat_idx == -1):
                grid_id = s_lon_idx * self.n_lat_len + s_lat_idx
                if val._6 < n_POI_cate:
                    POI_feature_cate[grid_id, val._6] += 1
                if val._7 < n_POI_cate:
                    POI_feature_cate[grid_id, val._7] += 1
                if val._8 < n_POI_cate:
                    POI_feature_cate[grid_id, val._8] += 1
                if val._9 < n_POI_cate:
                    POI_feature_cate[grid_id, val._9] += 1
                POI_feature_handed[grid_id, 3] += val.checkin

        POI_feature_handed[:, 0] = np.sum(POI_feature_cate, axis=1)
        POI_feature_handed[:, 1] = POI_feature_handed[:, 0] / (100 * 100)

        for idx in range(self.n_grid):
            if POI_feature_handed[idx, 0] > 0:
                diversity = -1 * np.sum([(v / (1.0 * POI_feature_handed[idx, 0])) *
                                         np.log(v / (1.0 * POI_feature_handed[idx, 0]))
                                         if v != 0 else 0 for v in POI_feature_cate[idx]])
                POI_feature_handed[idx, 2] = diversity
                POI_feature_cate_ratio[idx] = POI_feature_cate[idx, :] / POI_feature_handed[idx, 0]
            else:
                POI_feature_handed[idx, 2] = 0

        return POI_feature_cate, POI_feature_cate_ratio
