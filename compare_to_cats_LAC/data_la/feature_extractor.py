#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : feature_extractor.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from compare_to_cats_LAC.data_la.data_args import data_args


class FeatureExtractor:
    def __init__(self, n_grid, n_lon_len, n_lat_len):
        self.n_grid, self.n_lon_len, self.n_lat_len = n_grid, n_lon_len, n_lat_len
        # POI data
        self.POI_label_encoder = LabelEncoder()
        POI_data, n_POI_cate, self.POI_cate2ID = self.load_POI_data()
        self.n_POI_cate = n_POI_cate
        POI_feature_cate, POI_feature_cate_ratio, POI_feature_handed = self.generate_POI_feature(POI_data, n_POI_cate)

        self.feature = np.concatenate((POI_feature_cate, POI_feature_cate_ratio), axis=1)

    def get_grid_id(self, lon, lat):
        lon_idx = (lon - data_args.area_coordinate[0]) // data_args.grid_size_longitude_degree
        lat_idx = (lat - data_args.area_coordinate[2]) // data_args.grid_size_latitude_degree
        if not(0 <= lon_idx < self.n_lon_len and 0 <= lat_idx < self.n_lat_len):
            return -1, -1
        return int(lon_idx), int(lat_idx)

    def load_POI_data(self):
        POI_data = pd.read_csv(data_args.POI_path, usecols=[0, 1, 2, 3, 4],
                               dtype={
                                   'id': int,
                                   'name': str,
                                   'lat': float,
                                   'lon': float,
                                   'type': str,
                               })

        category_list = POI_data['type'].values.tolist()
        cate_list = list(set(category_list))

        self.POI_label_encoder.fit(cate_list)
        cate_corresponding_id = self.POI_label_encoder.transform(cate_list)
        POI_cate2ID = {}
        for idx, v in enumerate(cate_list):
            POI_cate2ID[v] = cate_corresponding_id[idx]

        POI_data['type'] = self.POI_label_encoder.transform(category_list)

        # print(POI_data)
        # print(cate_list)
        # print(len(cate_list))
        # print(POI_cate2ID)
        return POI_data, len(cate_list), POI_cate2ID

    def generate_POI_feature(self, POI_data, n_POI_cate):
        POI_feature_cate = np.zeros((self.n_grid, n_POI_cate))
        POI_feature_cate_ratio = np.zeros((self.n_grid, n_POI_cate))
        POI_feature_handed = np.zeros((self.n_grid, 4))

        for val in POI_data.itertuples():
            s_lon_idx, s_lat_idx = self.get_grid_id(val.lon, val.lat)
            if not (s_lon_idx == -1 or s_lat_idx == -1):
                grid_id = s_lon_idx * self.n_lat_len + s_lat_idx
                POI_feature_cate[grid_id, val.type] += 1

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

        return POI_feature_cate, POI_feature_cate_ratio, POI_feature_handed

    def process_zone(self, s):
        # 合并用地类型 在这里返回第几种类型id就行了，现在不是zone[i],而是zone[i]属于哪个大类用地类型
        for i in range(self.n_pluto_cate):
            if data_args.zones[i] in s:
                return i + 1
        return 0

    def load_pluto_data(self):
        pluto_data = pd.read_csv(data_args.pluto_path, usecols=[2, 3])
        pluto_data['ZONE_CMPLT'] = [self.process_zone(zone) for zone in pluto_data['ZONE_CMPLT']]
        na = [i for i in range(len(pluto_data['ZONE_CMPLT'])) if pluto_data['ZONE_CMPLT'][i] == 0]
        pluto_data = pluto_data.drop(na)
        pluto_data['the_geom'] = [process_polygon(geom) for geom in pluto_data['the_geom']]
        pluto_data['the_geom'] = [center_geolocation(geom) for geom in pluto_data['the_geom']]
        return pluto_data

        # pluto_data = pd.read_csv(data_args.pluto_path, usecols=[29, 87, 88])
        # pluto_data = pluto_data[pluto_data['landuse'] >= 0]
        # pluto_cate_set = set(pluto_data['landuse'].values.tolist())
        # n_pluto_cate = len(pluto_cate_set)
        # return pluto_data, n_pluto_cate

    def generate_pluto_data(self, pluto_data, n_pluto_cate_kind):
        pluto_feature_cate = np.zeros((self.n_grid, n_pluto_cate_kind))
        pluto_feature_cate_ratio = np.zeros((self.n_grid, n_pluto_cate_kind))
        pluto_feature_handed = np.zeros((self.n_grid, 3))

        for val in pluto_data.itertuples():
            lon, lat = val.the_geom
            s_lon_idx, s_lat_idx = self.get_grid_id(lon, lat)
            if not (s_lon_idx == -1 or s_lat_idx == -1):
                grid_id = s_lon_idx * self.n_lat_len + s_lat_idx
                if int(val.ZONE_CMPLT) == 0:
                    continue
                pluto_feature_cate[grid_id, int(val.ZONE_CMPLT) - 1] += 1

        pluto_feature_handed[:, 0] = np.sum(pluto_feature_cate, axis=1)
        pluto_feature_handed[:, 1] = pluto_feature_handed[:, 0] / (100 * 100)

        for idx in range(self.n_grid):
            if pluto_feature_handed[idx, 0] > 0:
                diversity = -1 * np.sum([(v / (1.0 * pluto_feature_handed[idx, 0])) *
                                         np.log(v / (1.0 * pluto_feature_handed[idx, 0]))
                                         if v != 0 else 0 for v in pluto_feature_cate[idx]])
                pluto_feature_handed[idx, 2] = diversity
                pluto_feature_cate_ratio[idx] = pluto_feature_cate[idx, :] / pluto_feature_handed[idx, 0]
            else:
                pluto_feature_handed[idx, 2] = 0

        return pluto_feature_cate, pluto_feature_cate_ratio, pluto_feature_handed

def process_polygon(s):
    s = s.replace('(((', ' ')
    s = s.replace(')))', ' ')
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace(',', ' ')
    s_list = s.split()
    n_s = len(s_list)
    geom_list = []
    for i in range(1, n_s, 2):
        lon = float(s_list[i])
        lat = float(s_list[i + 1])
        geom_list.append([lon, lat])
    return geom_list


def center_geolocation(geolocations):
    '''
    输入多个经纬度坐标(格式：[[lon1, lat1],[lon2, lat2],....[lonn, latn]])，找出中心点
    :param geolocations:
    :return:中心点坐标  [lon,lat]
    '''
    # 求平均数  同时角度弧度转化 得到中心点
    x = 0  # lon
    y = 0  # lat
    z = 0
    lenth = len(geolocations)
    for lon, lat in geolocations:
        lon = math.radians(float(lon))
        #  radians(float(lon))   Convert angle x from degrees to radians
        # 	                    把角度 x 从度数转化为 弧度
        lat = math.radians(float(lat))
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)
        x = float(x / lenth)
        y = float(y / lenth)
        z = float(z / lenth)
    return (math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.sqrt(x * x + y * y))))