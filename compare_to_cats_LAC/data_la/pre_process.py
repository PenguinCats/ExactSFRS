#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre_process.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None

import os
import numpy as np
import dgl
from compare_to_cats_LAC.data_la.feature_extractor import FeatureExtractor
from compare_to_cats_LAC.data_la.data_args import data_args


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
    # TODO: 根据洛杉矶的店铺类型和用地类型重新进行对应
    choice = {
        '住宅区': {'POI': ['Apartment Building'], 'land_use': [3, 4, 6, 8, 9, 12, 13, 16, 17, 18]},
        '商业区': {'POI': ['Business'], 'land_use': [19, 20, 21, 23, 24, 25]},
        '教育研究': {'POI': ['Schools and Educational Facilities', 'Research Institute'], 'land_use': [19, 35]},
        '餐饮': {'POI': ['Bar', 'Coffee', 'Restaurant', 'Snack', 'Canteen'], 'land_use': [3, 4, 19, 20, 21]},
        '医疗': {'POI': ['Hospital&Pharmacy', 'Nursing Home', 'Animal Shelter'], 'land_use': [23, 32]},
        '旅宿': {'POI': ['Hotel'], 'land_use': [19]},
        '购物': {'POI': ['Clothing Store', 'Market', 'Store'], 'land_use': [20, 21, 24, 25, 33, 36]},
        '公园景点': {'POI': ['Park', 'Open Area', 'Events Venue'], 'land_use': [0, 10, 14]},
        '墓地': {'POI': ['Cemetery', 'Mausoleum', 'Crematorium'], 'land_use': [32]},
        '交通': {'POI': ['Boat or Ferry', 'Highway or Road', 'Dock'], 'land_use': [22]},
        '文娱': {'POI': ['Movie Theater', 'Library', 'Centre and Pavilion', 'Concert', 'Museum', 'Club', 'Sports Venue',
                       'Other Entertainment'], 'land_use': [19, 23, 32, 38]},
        '宗教': {'POI': ['Religion'], 'land_use': [19, 32]},
        '办公区': {'POI': ['Public Building', 'Conference Centre', 'Office'], 'land_use': [19, 38]},
        '工业生产': {'POI': ['Factory'], 'land_use': [1, 2, 26, 27, 28, 29, 30, 31]},
        '政府设施': {'POI': ['Police Station', 'Fire Station', 'Government Building', 'Prison', 'Social Facility',
                         'Waste Transfer Station'], 'land_use': [32]},
        '出行服务': {'POI': ['Parking Garage', 'Gas Station or Car Service'], 'land_use': [32, 37, 34]},
        '便民服务': {'POI': ['Bank', 'Toilets', 'Post Office', 'Public Bath', 'Salon and Spa', 'Laundry',
                         'Hire and Share and Repair'], 'land_use': [19, 20, 21, 23, 24, 25]},
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

        pass


if __name__ == "__main__":
    preProcess = PreProcessor()
