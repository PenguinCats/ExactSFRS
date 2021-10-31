#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_args.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : 仅针对纽约单城市，不考虑 source target

import os


class DataArgs:
    def __init__(self):
        self.data_path = './data_file'
        self.POI_path = os.path.join(self.data_path, 'NYC_POI.csv')

        self.area_coordinate = [-74.006905, -73.959527, 40.707687, 40.751607]
        self.grid_size_longitude_degree = 0.001
        self.grid_size_latitude_degree = 0.001


data_args = DataArgs()
