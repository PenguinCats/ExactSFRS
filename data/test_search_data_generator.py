# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  test_search_data_generator.py
@Time    :  2021/2/25 0025 12:12
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

from args import args


class TestSearchDataGenerator(object):
    def __init__(self, city_data):
        self.city_data = city_data
        self.regions, self.coordinate_idx = self.generate_search_regions()

    def generate_search_regions(self):
        regions = []
        coordinate_idx = []
        for re in args.test_search_regions:
            lon_1, lon_2, lat_1, lat_2 = self.city_data.get_index_by_coordinate(re[0], re[1], re[2], re[3])
            region = self.city_data.get_region_feature_by_idx(lon_1, lon_2, lat_1, lat_2)
            regions.append(region)
            coordinate_idx.append([lon_1, lon_2, lat_1, lat_2])
        return regions, coordinate_idx
