#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : metrics.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/4/8
# @Desc  : None


import math


def hit_ratio_at_K(predict_list, label_list, k):
    # 务必保证 label_list 长度 >= k

    r_l = predict_list[:k]

    hit_cnt = 0
    for v in r_l:
        if v in label_list:
            hit_cnt += 1

    return hit_cnt / min(k, len(predict_list))


def ndcg_at_K(predict_list, label_list, k):
    # 务必保证 label_list 长度 >= k

    r_l = predict_list[:k]

    IDCG = 0
    for i in range(len(r_l)):
        IDCG += 1 / math.log2(i+2)

    DCG = 0
    for k, v in enumerate(r_l):
        if v in label_list:
            DCG += 1 / math.log2(k+2)

    return DCG / IDCG
