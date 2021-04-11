#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : load_tool.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/4/10
# @Desc  : None

import os
import pandas as pd
from args import args


def load_category():
    big_category = pd.read_csv(os.path.join(args.raw_data_dir, 'big_category.csv'))
    big_category_dict = dict()
    for item in big_category.itertuples():
        big_category_dict[item.name] = item.ID
    n_category = len(big_category_dict)

    return big_category_dict, n_category
