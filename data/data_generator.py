# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  data_generator.py
@Time    :  2021/2/6 0006 18:22
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import random
from data.args import args
from data.city_data import CityData
from data.train_data_generator import TrainDataGenerator
from data.test_effective_data_generator import TestEffectiveDataGenerator

if __name__ == '__main__':
    # init
    random.seed(args.seed)

    city_data = CityData()

    train_data = TrainDataGenerator(city_data)

    test_effective_data = TestEffectiveDataGenerator(city_data)

    print("data generate done.")
