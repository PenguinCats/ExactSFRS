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
from data.city_data_preprocessor import CityDataPreprocessor
from data.train_data_generator import TrainDataGenerator

if __name__ == '__main__':
    # init
    random.seed(args.seed)

    city_data = CityDataPreprocessor()

    train_data = TrainDataGenerator(city_data)

    print("data generate done.")
