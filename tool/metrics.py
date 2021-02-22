# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  metrics.py
@Time    :  2021/2/19 0019 11:33
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import math


def hit_ratio_at_K(v, predict_list, k=0):
    if k == 0:
        if v in predict_list:
            return 1
    else:
        if v in predict_list[:k]:
            return 1

    return 0


def MRR(v, predict_list, k=0):
    if k == 0:
        k = len(predict_list)

    for i in range(k):
        if v == predict_list[i]:
            return 1 / (i+1)

    return 0
