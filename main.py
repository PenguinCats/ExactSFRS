# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  main.py
@Time    :  2021/2/18 0018 16:42
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from metrics import hit_ratio_at_K, MRR
from args import args
from log_helper import log_tool_init, logging
from data.city_data import CityData
from data.train_data_generator import TrainDataGenerator
from data.test_effective_data_generator import TestEffectiveDataGenerator
from Triplet.triplet import Triplet, global_max_pooling

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # init
    random.seed(args.seed)
    log_tool_init(folder=args.train_log_dir, no_console=False)
    logging.info(' -- '.join(['%s:%s' % item for item in args.__dict__.items()]))

    # load data
    logging.info("loading city data...")
    city_data = CityData()

    # data generator
    logging.info("building data generator...")
    train_data_generator = TrainDataGenerator(city_data)
    evaluate_data_generator = TestEffectiveDataGenerator(city_data, args.evaluate_n_region)

    # model
    logging.info("building model...")
    triplet = Triplet()
    optimizer = torch.optim.AdamW(triplet.parameters(), weight_decay=args.weight_decay)

    # train
    logging.info("training...")
    loss_list = []
    hr_list = []
    mrr_list = []

    for epoch in range(args.train_n_tuples//args.training_data_generation_batch):
        # train step
        triplet.train()
        optimizer.zero_grad()

        (rq, pos, neg), _ = train_data_generator.generate_tuples(batch_size=args.training_data_generation_batch)
        v_rq = torch.stack([global_max_pooling(triplet(single_tuple)) for single_tuple in rq])
        v_pos = torch.stack([global_max_pooling(triplet(single_tuple)) for single_tuple in pos])
        v_neg = torch.stack([global_max_pooling(triplet(single_tuple)) for single_tuple in neg])

        dis_pos = torch.sum(nn.functional.mse_loss(v_rq, v_pos, reduction='none'), dim=1)
        dis_neg = torch.sum(nn.functional.mse_loss(v_rq, v_neg, reduction='none'), dim=1)

        loss = torch.sum(torch.relu(dis_pos/(dis_pos+dis_neg) - args.delta))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        logging.info('        Epoch {:03d} | Loss {:.4f}'.format(epoch, loss.item()))

        # evaluate
        if epoch % args.evaluate_gap == 0:

            triplet.eval()
            with torch.no_grad():
                hr_item = []
                mrr_item = []

                v_region = torch.stack([global_max_pooling(triplet(torch.Tensor(r)))
                                        for r in evaluate_data_generator.regions])
                v_pos = torch.stack([global_max_pooling(triplet(r)) for r in evaluate_data_generator.pos_set])

                dis_pos = torch.sum(nn.functional.mse_loss(v_region, v_pos, reduction='none'), dim=1)

                for idx, v_rq in enumerate(v_region):
                    dis_neg = torch.sum(nn.functional.mse_loss(v_rq, v_region[evaluate_data_generator.neg_set_idx[idx]],
                                                               reduction='none'), dim=1)
                    distances_total = torch.cat((dis_pos[idx].unsqueeze(0), dis_neg))
                    _, sorted_indices = torch.sort(distances_total, descending=False)
                    hr_item.append(hit_ratio_at_K(0, sorted_indices, k=args.K))
                    mrr_item.append(MRR(0, sorted_indices))

                hr_list.append(np.mean(hr_item))
                mrr_list.append(np.mean(mrr_item))

            logging.info('        Epoch {:03d} | HR {:.4f} | MRR {:.4f}'.format(epoch, np.mean(hr_item),
                                                                                np.mean(mrr_item)))

    local_time = time.localtime()

    # draw train result
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss epoch")
    plt.plot(range(len(loss_list)), loss_list)
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch")
    plt.ylabel("metrics")
    plt.plot(range(len(hr_list)), hr_list, color='red', label='HR@{}'.format(args.K), linestyle='-')
    plt.plot(range(len(mrr_list)), mrr_list, color='blue', label='MRR'.format(args.K), linestyle='-.')
    plt.savefig(os.path.join(args.trained_model_dir,
                             'train_result_{}.png'.format(time.strftime("%Y-%m-%d_%H-%M-%S", local_time))))
    plt.show()

    # save model
    torch.save(triplet,
               os.path.join(args.trained_model_dir, "model_{}.pkl".format(time.strftime("%Y-%m-%d_%H-%M-%S", local_time))))
