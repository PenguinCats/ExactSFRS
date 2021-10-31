# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  train.py
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tool.metrics import hit_ratio_at_K, MRR
from args import args
from compare_to_cats.exp_args import exp_args
from compare_to_cats.data_utils import TrainDataSet, TestDataSet
from compare_to_cats.metrics import hit_ratio, mrr
from tool.log_helper import log_tool_init, logging
from Triplet.triplet import Triplet, global_max_pooling


import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    start_local_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))

    # init
    random.seed(exp_args.seed)
    log_tool_init(folder=args.train_log_dir, no_console=False)
    logging.info(' -- '.join(['%s:%s' % item for item in args.__dict__.items()]))

    # device choose
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(exp_args.seed)
        N_GPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")

    # load data & data generator
    logging.info("loading city data...")
    city_base_info = np.load(os.path.join(exp_args.data_path, 'city_base_info_dict.npy'), allow_pickle=True).item()
    train_ds = TrainDataSet(city_base_info=city_base_info, size=exp_args.batch_per_epoch * exp_args.batch_size,
                            city_feature_path=os.path.join(exp_args.data_path, 'city_feature.npy'), exp_args=exp_args)
    POI_cater2ID_dict = np.load(os.path.join(exp_args.data_path, 'POI_cater2ID_dict.npy'), allow_pickle=True).item()
    train_dl = DataLoader(train_ds)
    # train_dl = DataLoader(train_ds, num_workers=8)

    test_ds = TestDataSet(city_base_info=city_base_info, size=exp_args.n_test_samples,
                          n_test_negative_size=exp_args.n_test_negative_size,
                          city_feature_path=os.path.join(exp_args.data_path, 'city_feature.npy'), exp_args=exp_args)
    test_dl = DataLoader(test_ds)

    # model
    logging.info("building model...")
    model = Triplet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=args.weight_decay)
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # train
    logging.info("training...")
    loss_list = []
    hr_list = []
    mrr_list = []

    for epoch in range(exp_args.train_epoch):
        # train
        total_loss = torch.tensor(0).float().to(DEVICE)
        iter_idx = 0
        iter_inner_cnt = 0

        model.train()
        for bd in train_dl:
            iter_inner_cnt += 1
            # print(bd[0][0].squeeze(0).size())
            # print(type(bd[0][0]))

            bd[0][0] = bd[0][0].squeeze(0)
            bd[0][1] = bd[0][1].squeeze(0)
            bd[0][2] = bd[0][2].squeeze(0)

            pos_lon_idx, pos_lat_idx = bd[2], bd[3]
            neg_lon_idx, neg_lat_idx = bd[4], bd[5]
            lon_len, lat_len = bd[6], bd[7]
            v_rq = bd[0][0][pos_lon_idx: pos_lon_idx + lon_len,  pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
            v_pos = bd[0][1][pos_lon_idx: pos_lon_idx + lon_len,  pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
            if neg_lon_idx < 0:
                v_neg = bd[0][2][pos_lon_idx: pos_lon_idx + lon_len,  pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
            else:
                v_neg = bd[0][2][neg_lon_idx: neg_lon_idx + lon_len, neg_lat_idx: neg_lat_idx + lat_len].to(DEVICE)

            v_rq = v_rq.permute(2, 0, 1).float()
            v_pos = v_pos.permute(2, 0, 1).float()
            v_neg = v_neg.permute(2, 0, 1).float()

            v_rq = global_max_pooling(model(v_rq))
            v_pos = global_max_pooling(model(v_pos))
            v_neg = global_max_pooling(model(v_neg))

            dis_pos = torch.sum(torch.pow(v_rq - v_pos, 2))
            dis_neg = torch.sum(torch.pow(v_rq - v_neg, 2))
            loss = torch.sum(torch.relu(dis_pos / (dis_pos + dis_neg) - args.delta))
            if loss.isnan():
                pass
            total_loss += loss
            if total_loss.isnan():
                pass

            if iter_inner_cnt == exp_args.batch_size:
                iter_idx += 1
                iter_inner_cnt = 0

                if not total_loss.isnan():
                    if iter_idx % exp_args.print_iter_frequency == 0:
                        logging.info('| Epoch {:02d} / {:04d} | Iter {:04d} / {:04d} | Iter Loss {:.4f}'.
                                     format(epoch + 1, exp_args.train_epoch, iter_idx, exp_args.batch_per_epoch, total_loss.cpu().item()))

                    total_loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                total_loss = torch.tensor(0).float().to(DEVICE)

        # evaluate
        with torch.no_grad():
            model.eval()

            hr_epoch_item = []
            mrr_epoch_item = []

            hr_epoch_item_20 = []
            mrr_epoch_item_20 = []

            for tbd in test_dl:
                # print(tbd)
                pos_lon_idx, pos_lat_idx = tbd[2], tbd[3]
                lon_len, lat_len = tbd[6], tbd[7]

                v_rq = tbd[0][0].squeeze(0)[pos_lon_idx: pos_lon_idx + lon_len, pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
                v_rq = v_rq.permute(2, 0, 1).float()
                v_rq = global_max_pooling(model(v_rq))

                v_pos = tbd[0][1].squeeze(0)[pos_lon_idx: pos_lon_idx + lon_len, pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
                v_pos = v_pos.permute(2, 0, 1).float()
                v_pos = global_max_pooling(model(v_pos))

                dis_pos = torch.sum(torch.pow(v_rq - v_pos, 2))

                dis_negative_list = []
                for nid, negative_f in enumerate(tbd[0][2]):
                    neg_lon_idx, neg_lat_idx = tbd[4][nid], tbd[5][nid]
                    if neg_lon_idx < 0:
                        neg_f = negative_f.squeeze(0)[pos_lon_idx: pos_lon_idx + lon_len,
                                                      pos_lat_idx: pos_lat_idx + lat_len].to(DEVICE)
                    else:
                        neg_f = negative_f.squeeze(0)[neg_lon_idx: neg_lon_idx + lon_len,
                                                      neg_lat_idx: neg_lat_idx + lat_len].to(DEVICE)
                    neg_f = neg_f.permute(2, 0, 1).float()
                    neg_f = global_max_pooling(model(neg_f))
                    dis_neg = torch.sum(torch.pow(v_rq - neg_f, 2))
                    dis_negative_list.append(dis_neg)

                distances_total = torch.cat((dis_pos.unsqueeze(0), torch.tensor(dis_negative_list, device=DEVICE)))
                _, sorted_indices = torch.sort(distances_total, descending=False)
                hr_epoch_item.append(hit_ratio(0, sorted_indices, exp_args.n_test_K))
                mrr_epoch_item.append(mrr(0, sorted_indices, exp_args.n_test_K))
                hr_epoch_item_20.append(hit_ratio(0, sorted_indices, 20))
                mrr_epoch_item_20.append(mrr(0, sorted_indices, 20))

            hr_epoch_item = np.mean(hr_epoch_item)
            mrr_epoch_item = np.mean(mrr_epoch_item)
            hr_epoch_item_20 = np.mean(hr_epoch_item_20)
            mrr_epoch_item_20 = np.mean(mrr_epoch_item_20)
            hr_list.append(hr_epoch_item)
            mrr_list.append(mrr_epoch_item)
            logging.info('| Epoch {:02d} / {:04d} | HR@{} {:.4f} | MRR@{} {:.4f} |'.
                         format(epoch + 1, exp_args.train_epoch,
                                exp_args.n_test_K, hr_epoch_item, exp_args.n_test_K, mrr_epoch_item))
            logging.info('| Epoch {:02d} / {:04d} | HR@20 {:.4f} | MRR@20 {:.4f} |'.
                         format(epoch + 1, exp_args.train_epoch,
                                hr_epoch_item_20, mrr_epoch_item_20))

    # save model
    torch.save(model, 'trained_model_{}.pth'.format(start_local_time))
    logging.info(start_local_time)

    #
    # # draw train result
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.xlabel("epoch")
    # plt.ylabel("loss epoch")
    # plt.plot(range(len(loss_list)), loss_list)
    # plt.subplot(1, 2, 2)
    # plt.xlabel("epoch")
    # plt.ylabel("metrics")
    # plt.plot(range(len(hr_list)), hr_list, color='red', label='HR@{}'.format(args.K), linestyle='-')
    # plt.plot(range(len(mrr_list)), mrr_list, color='blue', label='MRR'.format(args.K), linestyle='-.')
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.trained_model_dir,
    #                          'train_result_{}.png'.format(time.strftime("%Y-%m-%d_%H-%M", start_local_time))))
    # plt.show()