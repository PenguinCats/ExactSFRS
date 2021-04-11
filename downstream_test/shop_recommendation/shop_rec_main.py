#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : shop_rec_main.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/4/8
# @Desc  : None

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from downstream_test.shop_recommendation.tools.DataLoader import DataLoader
from downstream_test.shop_recommendation.tools.metrics import hit_ratio_at_K, ndcg_at_K
from args import args as main_args


class DNN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_dim, out_features=32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=1, bias=True)
        )
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=1, bias=True)

    def forward(self, f_in):
        return self.linear(f_in)


if __name__ == "__main__":

    # data & feature
    data = DataLoader()

    # spp
    pool_tool = torch.nn.AdaptiveMaxPool2d((1, 1))

    # model
    model = DNN(data.city_feature.shape[1])
    optimizer = torch.optim.AdamW(model.parameters())

    hr_list = []
    ndcg_list = []
    loss_list = []
    # train
    for batch in range(main_args.downstream_train_batch_n):
        model.train()
        train_features, train_labels = data.generate_train_feature(batch_size=main_args.downstream_train_batch_size)
        predict_label = torch.flatten(torch.stack([model(torch.flatten(pool_tool(feature))) for feature in train_features]))
        loss = torch.sum(torch.pow(predict_label - train_labels, 2))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.data)
        print("Batch {:03d} | Loss {:.4f}".format(batch, loss.data))

        # test
        if batch % main_args.downstream_evaluate_frequency == 0:
            model.eval()
            with torch.no_grad():
                features = data.get_test_features()
                predict_label = torch.flatten(torch.stack([model(torch.flatten(pool_tool(feature))) for feature in features]))
                _, sorted_indices = torch.sort(predict_label, descending=True)
                HR = hit_ratio_at_K(sorted_indices, data.test_pos_sorted_indices, main_args.downstream_evaluate_k)
                NDCG = ndcg_at_K(sorted_indices, data.test_pos_sorted_indices, main_args.downstream_evaluate_k)
                hr_list.append(HR)
                ndcg_list.append(NDCG)
                print('Batch {:03d} | HR {:.4f} | MRR {:.4f}'.
                      format(batch, HR, NDCG))

    # draw train result
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss epoch")
    plt.plot(range(len(loss_list)), loss_list)
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch")
    plt.ylabel("metrics")
    plt.plot(range(len(hr_list)), hr_list, color='red', label='HR@{}'.format(main_args.downstream_evaluate_k), linestyle='-')
    plt.plot(range(len(ndcg_list)), ndcg_list, color='blue', label='NDCG@{}'.format(main_args.downstream_evaluate_k), linestyle='-.')
    plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.savefig(os.path.join(main_args.downstream_task_test_result_path,
    #                          'downstream_result_{}.png'.format(main_args.downstream_test_model_name)))
    plt.savefig('downstream_result_{}.png'.format(main_args.downstream_test_model_name))
    plt.show()
