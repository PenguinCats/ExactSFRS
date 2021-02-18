# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  triplet.py
@Time    :  2021/2/6 0006 11:19
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""
from typing import Any

import torch.nn as nn
from args import args


class Triplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Net = nn.Sequential()
        for i in range(len(args.filter_size)):
            self.Net.add_module("conv_{}".format(i),
                                nn.Conv2d(in_channels=args.feature_dim[i], out_channels=args.feature_dim[i+1],
                                          stride=args.stride[i], kernel_size=args.filter_size[i]))
            self.Net.add_module("relu_{}".format(i), nn.ReLU())

        self.Net.add_module("dropout", nn.Dropout(p=args.dropout_rate))

        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, data):
        data = self.Net(data.unsqueeze(0))
        data = self.global_max_pooling(data).squeeze()
        return data

