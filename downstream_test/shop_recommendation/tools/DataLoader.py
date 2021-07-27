#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : DataLoader.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/4/8
# @Desc  : None

import os
import torch
import random
import numpy as np
import pandas as pd
from args import args
from tool.load_tool import load_category


def get_train_pos_neg_position(train_label):
    pos = []
    neg = []
    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i][j] > 0:
                pos.append((i, j))
            else:
                neg.append((i, j))

    return pos, neg


def get_test_pos_indices(test_label):
    _, sorted_indices = torch.sort(test_label, descending=True)
    test_pos_cnt = int(torch.sum(test_label).item())

    return sorted_indices[:test_pos_cnt]


class DataLoader:
    def __init__(self):
        self.city_feature = torch.load(os.path.join(args.trained_model_dir,
                                                    "city_feature_{}.pt".format(
                                                        args.downstream_test_model_name))).to(torch.device('cpu'))

        self.before_cnn_city_size = (len(np.arange(args.city_range[0], args.city_range[2], args.grid_step[0])),
                                     len(np.arange(args.city_range[3], args.city_range[1], args.grid_step[1])))
        self.after_cnn_city_size = self.city_feature.shape[2:]
        self.scale = (self.before_cnn_city_size[0] / self.after_cnn_city_size[0],
                      self.before_cnn_city_size[1] / self.after_cnn_city_size[1])

        self.area_n_width, self.area_n_height, self.after_area_step_width, self.after_area_step_height = self.split_area()
        self.n_area = self.area_n_width * self.area_n_height

        self.train_label, self.test_label = self.load_train_and_test_label()
        self.test_pos_sorted_indices = get_test_pos_indices(self.test_label)
        self.pos_area, self.neg_area = get_train_pos_neg_position(self.train_label)
        # self.test_area_score, self.sorted_indices, self.test_pos_cnt = cal_area_score(self.test_label)

    def split_area(self):
        area_n_width = int(self.before_cnn_city_size[0] // args.downstream_area_step)
        area_n_height = int(self.before_cnn_city_size[1] // args.downstream_area_step)
        after_area_step_width = self.after_cnn_city_size[0] / area_n_width
        after_area_step_height = self.after_cnn_city_size[1] / area_n_height
        return area_n_width, area_n_height, after_area_step_width, after_area_step_height

    def load_train_and_test_label(self):
        data, n_category = self.load_dianping_data()
        train_label = np.zeros((self.area_n_width, self.area_n_height))
        test_label = np.zeros((self.area_n_width, self.area_n_height))

        cnt1, cnt2, cnt3 = 0, 0, 0
        for item in data.itertuples():
            lon, lat = item.longitude, item.latitude
            if item.name not in args.downstream_train_shops and \
                    item.name not in args.downstream_test_shops:
                continue

            lon_idx = int((lon - args.tencent_city_gps_range[0]) / args.grid_step[0])
            lat_idx = int((lat - args.tencent_city_gps_range[2]) / args.grid_step[1])

            area_width_idx = int(lon_idx / args.downstream_area_step)
            area_height_idx = int(lat_idx / args.downstream_area_step)

            if 0 <= area_width_idx < self.area_n_width and 0 <= area_height_idx < self.area_n_height:
                if item.name in args.downstream_train_shops:
                    train_label[area_width_idx, area_height_idx] += 1
                    cnt1 += 1
                else:
                    test_label[area_width_idx, area_height_idx] += 1
                    cnt2 += 1
            else:
                # print(item)
                cnt3 += 1

        # print(cnt1, cnt2, cnt3)

        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_label = torch.flatten(torch.tensor(test_label, dtype=torch.float32))

        return train_label, test_label

    def load_dianping_data(self):
        big_category_dict, n_category = load_category()
        data_path = os.path.join(args.raw_data_dir, args.downstream_test_city + '.csv')
        data = pd.read_csv(data_path, usecols=args.city_data_index_order)

        # format data
        data = data[data['status'] == 0].drop(columns='status')
        data['category'] = data['big_category'].map(lambda x: big_category_dict[x])
        data = data.drop(columns='big_category')

        return data, n_category

    def generate_train_feature(self, batch_size: int = 2):
        train_feature = []
        train_label = []

        pos_positions = random.sample(self.pos_area, int(batch_size / 2))
        neg_positions = random.sample(self.neg_area, int(batch_size / 2))

        for pos in pos_positions:
            train_feature.append(
                self.city_feature[:, :, int(pos[0] * self.after_area_step_width): int((pos[0] + 1) * self.after_area_step_width),
                int(pos[1] * self.after_area_step_height): int((pos[1] + 1) * self.after_area_step_height)])
            train_label.append(self.train_label[pos[0], pos[1]])

        for pos in neg_positions:
            train_feature.append(
                self.city_feature[:, :, int(pos[0] * self.after_area_step_width): int((pos[0] + 1) * self.after_area_step_width),
                int(pos[1] * self.after_area_step_height): int((pos[1] + 1) * self.after_area_step_height)])
            train_label.append(self.train_label[pos[0], pos[1]])

        cc = list(zip(train_feature, train_label))
        random.shuffle(cc)
        train_feature[:], train_label[:] = zip(*cc)

        return train_feature, torch.flatten(torch.stack(train_label))

    def get_test_feature(self, a_lon_idx, a_lat_idx):
        return self.city_feature[:, :,
               int(a_lon_idx * self.after_area_step_width): int((a_lon_idx + 1) * self.after_area_step_width),
               int(a_lat_idx * self.after_area_step_height): int((a_lat_idx + 1) * self.after_area_step_height)]

    def get_test_features(self):
        features = []
        for w in range(self.area_n_width):
            for h in range(self.area_n_height):
                features.append(self.get_test_feature(w, h))
        return features
