#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_args.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : 仅针对洛杉矶单城市，不考虑 source target

import os


class DataArgs:
    def __init__(self):
        self.data_path = './data_file'
        self.POI_path = os.path.join(self.data_path, 'LA_POI.csv')
        self.pluto_path = os.path.join(self.data_path, 'LA_landuse.csv')

        self.area_coordinate = [-118.435276,  -118.240785, 33.929096, 34.117110]
        self.grid_size_longitude_degree = 0.001
        self.grid_size_latitude_degree = 0.001
        self.zones = ["OS", "A1", "A2", "RAS4", "RAS3", "RA", "RE", "RS", "R1", "RU", "RZ", "RW1", "R2", "RD",
                      "RMP", "RW2", "R3", "R4", "R5", "CR", "C1", "C1.5", "LAX",
                      "C4", "C2", "C5", "CM", "MR1", "M1", "MR2", "M2", "M3", "PF", 'NI', 'HJ', 'HR', "MU", "P", "CEC"]
        self.n_z = len(self.zones)

data_args = DataArgs()
