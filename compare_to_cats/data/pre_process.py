#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre_process.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None

import os
import numpy as np
from compare_to_cats.data.feature_extractor import FeatureExtractor
from compare_to_cats.data.data_args import data_args


def get_city_size():
    area_longitude_boundary = np.arange(data_args.area_coordinate[0], data_args.area_coordinate[1],
                                        data_args.grid_size_longitude_degree)
    area_latitude_boundary = np.arange(data_args.area_coordinate[2], data_args.area_coordinate[3],
                                       data_args.grid_size_latitude_degree)

    n_grid = (len(area_longitude_boundary) - 1) * (len(area_latitude_boundary) - 1)

    print('n_grid: {}'.format(n_grid))

    n_lon_len = (len(area_longitude_boundary) - 1)
    n_lat_len = (len(area_latitude_boundary) - 1)

    return n_grid, n_lon_len, n_lat_len


def get_choice_corresponding_features():
    choice = {
        '住宅区': {'POI': ['Apartment Building'], 'land_use': [0, 10]},
        '学校': {'POI': ['School', 'Bar', 'Coffee', 'Restaurant', 'Snack'], 'land_use': [13]},
        '餐饮': {'POI': ['Bakery&Dessert', 'Bar', 'Coffee', 'Restaurant', 'Snack'], 'land_use': [5, 10]},
        '医疗': {'POI': ['Hospital'], 'land_use': [7]},
        '旅宿': {'POI': ['Hotel'], 'land_use': [6]},
        '购物': {'POI': ['Clothing Store', 'Mall', 'Market', 'Store', 'Bike Shop'], 'land_use': [5]},
        '公园景点': {'POI': ['Park', 'Resort'], 'land_use': [8]},
        '墓地': {'POI': ['Cemetery'], 'land_use': [3]},
        '交通': {'POI': ['Airport', 'Boat or Ferry', 'Highway or Road', 'Public Transportation', 'Train Station'], 'land_use': [12]},
        '文娱': {'POI': ['Club', 'Concert', 'Gaming Cafe', 'Gym', 'Library', 'Movie Theater', 'Museum', 'Recreation', 'Other Great Outdoors', 'Other Nightlife'], 'land_use': [3, 5, 8]},
        '宗教': {'POI': ['Religion'], 'land_use': [3]},
        '办公区': {'POI': ['Office'], 'land_use': [9, 10]},
        '工业生产': {'POI': ['Farm', 'Winery'], 'land_use': [1]},
        '政府设施': {'POI': ['Fire Station', 'Government Building', 'Police Station'], 'land_use': [11]},
        '出行服务': {'POI': ['Gas Station or Garage', 'Parking Garage', 'Rest Area'], 'land_use': [2, 3, 4]},
        '便民服务': {'POI': ['Bank', 'Post Office'], 'land_use': [5]},
    }
    return choice


class PreProcessor:
    def __init__(self):
        n_grid, n_lon_len, n_lat_len = get_city_size()

        featureExtractor = FeatureExtractor(n_grid=n_grid, n_lon_len=n_lon_len, n_lat_len=n_lat_len)
        np.save(os.path.join(data_args.data_path, 'city_feature.npy'), featureExtractor.feature)
        np.save(os.path.join(data_args.data_path, 'POI_cater2ID_dict.npy'), featureExtractor.POI_cate2ID)

        choice_corresponding_features = get_choice_corresponding_features()
        for k in choice_corresponding_features.keys():
            choice_corresponding_features[k]['POI'] = \
                featureExtractor.POI_label_encoder.transform(choice_corresponding_features[k]['POI']).tolist()

        city_base_info = {
            'n_grid': n_grid,
            'n_lon_len': n_lon_len,
            'n_lat_len': n_lat_len,
            'n_POI_cate': featureExtractor.n_POI_cate,
            'area_coordinate': data_args.area_coordinate,
            'grid_size_longitude_degree': data_args.grid_size_longitude_degree,
            'grid_size_latitude_degree': data_args.grid_size_latitude_degree,
            'choice_corresponding_features': choice_corresponding_features,
        }

        print(city_base_info)
        np.save(os.path.join(data_args.data_path, 'city_base_info_dict.npy'), city_base_info)


if __name__ == "__main__":
    preProcess = PreProcessor()
