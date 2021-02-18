# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  main.py
@Time    :  2021/2/18 0018 16:42
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import random
import torch
import torch.nn as nn
from tqdm import tqdm
from args import args
from data.city_data import CityData
from data.train_data_generator import TrainDataGenerator
from Triplet.triplet import Triplet

if __name__ == '__main__':
    # init
    random.seed(args.seed)

    # load data
    city_data = CityData()

    # data generator
    train_data_generator = TrainDataGenerator(city_data)

    # model
    triplet = Triplet()
    optimizer = torch.optim.AdamW(triplet.parameters(), weight_decay=args.weight_decay)

    # train
    for epoch in tqdm(range(args.train_n_tuples//args.training_data_generation_batch)):
        optimizer.zero_grad()

        (rq, pos, neg), _ = train_data_generator.generate_tuples(batch_size=args.training_data_generation_batch)
        v_rq = torch.stack([triplet(single_tuple) for single_tuple in rq])
        v_pos = torch.stack([triplet(single_tuple) for single_tuple in pos])
        v_neg = torch.stack([triplet(single_tuple) for single_tuple in neg])

        dis_pos = torch.sum(nn.functional.mse_loss(v_rq, v_pos, reduction='none'), dim=1)
        dis_neg = torch.sum(nn.functional.mse_loss(v_rq, v_neg, reduction='none'), dim=1)

        loss = torch.sum(torch.relu(dis_pos/(dis_pos+dis_neg) - args.delta))
        loss.backward()
        optimizer.step()

    # test
    # test_effective_data = TestEffectiveDataGenerator(city_data)
