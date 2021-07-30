#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : recommendation.py
# @Author: Jun Tang (jtang@seu.edu.cn)
# @Date  : 2021/7/24
# @Desc  : None
import functools
from tqdm import tqdm
import os
import pandas as pd
import math
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
from compare_to_cats.data_utils import TrainDataSet, TestDataSet, RealWorldDataSet
from compare_to_cats.metrics import hit_ratio, mrr
from tool.log_helper import log_tool_init, logging
from Triplet.triplet import Triplet, global_max_pooling
from compare_to_cats.data.data_args import data_args


class RecommendationPreProcessor:
    def __init__(self, city_base_info, shops):
        self.n_grid, self.n_lon_len, self.n_lat_len = city_base_info['n_grid'], city_base_info['n_lon_len'], \
                                                      city_base_info['n_lat_len']
        self.shop_grid = self.load_and_process(shops)

    def get_grid_id(self, lon, lat):
        lon_idx = (lon - data_args.area_coordinate[0]) // data_args.grid_size_longitude_degree
        lat_idx = (lat - data_args.area_coordinate[2]) // data_args.grid_size_latitude_degree
        if not (0 <= lon_idx < self.n_lon_len and 0 <= lat_idx < self.n_lat_len):
            return -1, -1
        return int(lon_idx), int(lat_idx)

    def get_area_id_list(self, lon_idx, lat_idx, lon_len, lat_len):
        lon_start_idx = max(0, lon_idx - lon_len + 1)
        lat_start_idx = max(0, lat_idx - lat_len + 1)
        lon_end_idx = min(lon_idx, self.n_lon_len - exp_args.rwt_dg_length_and_width[0] - 1)
        lat_end_idx = min(lat_idx, self.n_lat_len - exp_args.rwt_dg_length_and_width[1] - 1)
        area_list = []
        for i in range(lon_start_idx, lon_end_idx + 1):
            for j in range(lat_start_idx, lat_end_idx + 1):
                area_list.append([i, j])
        return area_list

    def get_area_id(self, lon, lat):
        lon_idx, lat_idx = self.get_grid_id(lon, lat)
        if lon_idx == -1 or lat_idx == -1:
            return -1, -1
        area_lon_idx = (lon_idx // exp_args.rwt_dg_length_and_width[0]) * exp_args.rwt_dg_length_and_width[0]
        area_lat_idx = (lat_idx // exp_args.rwt_dg_length_and_width[1]) * exp_args.rwt_dg_length_and_width[1]
        if self.n_lon_len - area_lon_idx < exp_args.rwt_dg_length_and_width[0] or self.n_lat_len - area_lat_idx < \
                exp_args.rwt_dg_length_and_width[1]:
            return -1, -1
        return area_lon_idx, area_lat_idx

    def load_and_process(self, shops):
        shop_grid = {}
        for val in shops:
            shop_grid[val] = []
        POI_data = pd.read_csv(os.path.join(exp_args.data_path, 'NYC_POI.csv'), usecols=[1, 2, 3])

        for row in POI_data.itertuples():
            shop_name = getattr(row, '_1')

            for shop in shops:
                if shop.upper() in shop_name.upper():
                    lon = getattr(row, 'longitude')
                    lat = getattr(row, 'latitude')
                    lon_idx, lat_idx = self.get_area_id(lon, lat)
                    if lon_idx != -1 and lat_idx != -1 and ([lon_idx, lat_idx] not in shop_grid[shop]):
                        shop_grid[shop].append([lon_idx, lat_idx])
                    break
        for shop in shops:  # 弹出最后一个作为选择的区域
            print(shop, ' len: ', len(shop_grid[shop]), ' | ', shop_grid[shop])
        return shop_grid


def HR(ordered_dis_list, shop_grid):
    node_list = [5, 10, 15, 20]
    HR = []
    id = 0
    hits = 0
    shop_grid_num = len(shop_grid)
    for i in range(20):
        lon_idx = ordered_dis_list[i][1]
        lat_idx = ordered_dis_list[i][2]
        if [lon_idx, lat_idx] in shop_grid:
            hits += 1
        if (i + 1) == node_list[id]:
            HR.append(hits / shop_grid_num)
            id += 1
    return HR


def NDCG(ordered_dis_list, shop_grid):
    node_list = [5, 10, 15, 20]
    NDCG = []
    id = 0
    rel = []
    IDCG = []
    shop_num = len(shop_grid)
    idcg_sum = 0
    for i in range(20):
        lon_idx = ordered_dis_list[i][1]
        lat_idx = ordered_dis_list[i][2]
        if [lon_idx, lat_idx] in shop_grid:
            rel.append(1)
        else:
            rel.append(0)

        if (i + 1) < shop_num:
            idcg_sum += 1 / math.log(i + 2, 2)
        IDCG.append(idcg_sum)

    dcg_sum = 0
    for i in range(20):
        dcg_sum += (2 ** (rel[i]) - 1) / math.log(i + 2, 2)
        if (i + 1) == node_list[id]:
            NDCG.append(dcg_sum / IDCG[i])
            id += 1
    return NDCG


def cmp(x, y):
    sum_x_hr = sum(x[0])
    sum_x_ndcg = sum(x[1])
    sum_y_hr = sum(y[0])
    sum_y_ndcg = sum(y[1])
    if sum_x_hr == sum_y_hr:
        return sum_y_ndcg - sum_x_ndcg
    return sum_y_hr - sum_x_hr


if __name__ == '__main__':
    # shops = ['KFC', 'STARBUCKS', 'McDonald', 'Burger King', 'Subway', '7-Eleven', 'Nike']  # 'MARKET', 'PARK',
    shops = ['DUANE READE', 'DUNKIN', 'CHASE BANK', 'RITE AID', 'APARTMENT']

    # set random seed
    random.seed(exp_args.seed)
    np.random.seed(exp_args.seed)
    torch.manual_seed(exp_args.seed)

    # device choose
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(exp_args.seed)
        N_GPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda:{}".format(1) if CUDA_AVAILABLE else "cpu")

    logging.info("loading city data...")
    city_base_info = np.load(os.path.join(exp_args.data_path, 'city_base_info_dict.npy'), allow_pickle=True).item()
    POI_cater2ID_dict = np.load(os.path.join(exp_args.data_path, 'POI_cater2ID_dict.npy'), allow_pickle=True).item()
    test_ds = RealWorldDataSet(city_base_info=city_base_info, size=exp_args.batch_per_epoch * exp_args.batch_size,
                               city_feature_path=os.path.join(exp_args.data_path, 'city_feature.npy'),
                               exp_args=exp_args)
    # RecommendationPreProcessor
    RPP = RecommendationPreProcessor(city_base_info, shops)

    # model and optimizer
    model = torch.load("trained_model.pth")
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    model_val = {}
    now_city_feature = torch.tensor(test_ds.city_feature)
    candidate_lon_list, candidate_lat_list = test_ds.lon_start_list, test_ds.lat_start_list
    for j in range(test_ds.candidate_size):
        candi_lon_idx, candi_lat_idx = candidate_lon_list[j], candidate_lat_list[j]
        v_candi = now_city_feature[
                  candi_lon_idx: candi_lon_idx + exp_args.rwt_dg_length_and_width[0],
                  candi_lat_idx: candi_lat_idx + exp_args.rwt_dg_length_and_width[1]].to(DEVICE)
        v_candi = v_candi.permute(2, 0, 1).float()
        v_candi = global_max_pooling(model(v_candi))
        model_val[(candi_lon_idx, candi_lat_idx)] = v_candi

    outfile = open('out.txt', 'w')
    # test
    with torch.no_grad():
        model.eval()

        for i in range(len(shops)):  # 对每个shop进行测试
            now_shop = shops[i]
            now_hr_ndcg_list = []
            # 枚举当前店铺选中的gird，测试实验效果
            print(now_shop, file=outfile)
            print(now_shop, " start :")
            for k in tqdm(range(0, len(RPP.shop_grid[now_shop]))):
                now_shop_grid = RPP.shop_grid[now_shop].copy()
                lon_idx, lat_idx = now_shop_grid.pop(k)
                v_rq = model_val[(lon_idx, lat_idx)]

                dis_list = []
                # candidate_lon_list, candidate_lat_list = test_ds.lon_start_list, test_ds.lat_start_list
                for j in range(test_ds.candidate_size):
                    candi_lon_idx, candi_lat_idx = candidate_lon_list[j], candidate_lat_list[j]
                    if candi_lon_idx == lon_idx and candi_lat_idx == lat_idx:
                        continue
                    v_candi = model_val[(candi_lon_idx, candi_lat_idx)]
                    dis = torch.sum(torch.pow(v_rq - v_candi, 2))
                    dis_list.append((dis, candi_lon_idx, candi_lat_idx))

                ordered_dis_list = sorted(dis_list, key=functools.cmp_to_key(lambda x, y: x[0] - y[0]))
                now_hr = HR(ordered_dis_list, now_shop_grid)
                now_ndcg = NDCG(ordered_dis_list, now_shop_grid)
                now_hr_ndcg_list.append([now_hr, now_ndcg, lon_idx, lat_idx])

            # 对选中不同grid的结果进行从大到小排序
            now_hr_ndcg_list = sorted(now_hr_ndcg_list, key=functools.cmp_to_key(mycmp=cmp))
            print("========== {} ==========".format(i))
            print("shop: {}".format(now_shop))
            # 输出前五
            for k in range(min(5, len(now_hr_ndcg_list))):
                ori_lon_1, ori_lon_2, ori_lat_1, ori_lat_2 = test_ds.get_coordinate_by_idx(now_hr_ndcg_list[k][2],
                                                                                           now_hr_ndcg_list[k][3])
                print("origin: {},{}     {},{}".format(ori_lat_1, ori_lon_1, ori_lat_2, ori_lon_2))
                print("HR: ", now_hr_ndcg_list[k][0])
                print("NDCG: ", now_hr_ndcg_list[k][1])
                print("---------------------")

                print(k, " HR,", end='', file=outfile)
                for t in range(len(now_hr_ndcg_list[k][0])):
                    if t == len(now_hr_ndcg_list[k][0])-1:
                        print(now_hr_ndcg_list[k][0][t], file=outfile)
                    else:
                        print(now_hr_ndcg_list[k][0][t], ',', end='', file=outfile)
                print(k, " NDCG,", end='', file=outfile)
                for t in range(len(now_hr_ndcg_list[k][1])):
                    if t == len(now_hr_ndcg_list[k][1]) - 1:
                        print(now_hr_ndcg_list[k][1][t], file=outfile)
                    else:
                        print(now_hr_ndcg_list[k][1][t], ',', end='', file=outfile)
