#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_utils.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn) & Jun Tang (jtang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None

import torch
import torch.nn as nn
import numpy as np
import random


class BaseDataSet(torch.utils.data.Dataset):
    def __init__(self, city_base_info, size, city_feature_path, exp_args):
        super().__init__()

        self.area_coordinate = city_base_info['area_coordinate']
        self.grid_size_longitude_degree = city_base_info['grid_size_longitude_degree']
        self.grid_size_latitude_degree = city_base_info['grid_size_latitude_degree']
        self.n_grid = city_base_info['n_grid']
        self.n_lon_len = city_base_info['n_lon_len']
        self.n_lat_len = city_base_info['n_lat_len']
        self.choice_corresponding_features = city_base_info['choice_corresponding_features']
        self._choice_keys = list(self.choice_corresponding_features.keys())
        self.n_POI_cate = city_base_info['n_POI_cate']

        self.POI_index_range = [0, self.n_POI_cate - 1]
        self.POI_ratio_index_range = [self.n_POI_cate, self.n_POI_cate * 2 - 1]

        self._size = size

        city_feature = np.load(city_feature_path)
        self.city_feature = city_feature.reshape((self.n_lon_len, self.n_lat_len, -1))

        self.exp_args = exp_args

    def get_grid_id(self, lon, lat):
        lon_idx = (lon - self.area_coordinate[0]) // self.grid_size_longitude_degree
        lat_idx = (lat - self.area_coordinate[2]) // self.grid_size_latitude_degree
        if not(0 <= lon_idx < self.n_lon_len and 0 <= lat_idx < self.n_lat_len):
            return -1, -1
        return int(lon_idx), int(lat_idx)

    def get_area_grids_list(self, lon_idx, lat_idx, length, width):
        grids_list = []
        lower_left_grid_id = lon_idx * self.n_lat_len + lat_idx
        for i in range(length):
            start_id = i * self.n_lat_len + lower_left_grid_id
            grids_list.append([start_id + j for j in range(width)])
        grids_list = np.array(grids_list).flatten()
        return grids_list


class ForModifyDataSet(BaseDataSet):
    def __init__(self, city_base_info, size, city_feature_path, exp_args):
        """
        :param size: size = batch_size * number of batches per epoch
        """
        super().__init__(city_base_info=city_base_info, size=size,
                         city_feature_path=city_feature_path, exp_args=exp_args)

    def _change_features_total_ratio(self, sample, lon_idx, lat_idx, length, width,
                                     selected_feature_ratio_index_range,
                                     unselected_feature_ratio_index_range,
                                     selected_attributes_modify_radio):
        noise = 0.0000000001
        selected_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                 selected_feature_ratio_index_range], axis=2)
        unselected_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                   unselected_feature_ratio_index_range], axis=2)
        selected_attributes_total_ratio = np.reshape(selected_attributes_total_ratio, (length, width, 1))
        unselected_attributes_total_ratio = np.reshape(unselected_attributes_total_ratio, (length, width, 1))

        noise_matrix = np.ones((length, width, 1)) * noise  # avoid dividing by 0
        selected_attributes_total_ratio += noise_matrix
        unselected_attributes_total_ratio += noise_matrix

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] /= \
            selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_feature_ratio_index_range] /= \
            unselected_attributes_total_ratio

        selected_modify_matrix = np.random.choice([-selected_attributes_modify_radio, selected_attributes_modify_radio],
                                                  (length, width, 1))
        selected_attributes_total_ratio += selected_modify_matrix
        x1, y1, z1 = np.where(selected_attributes_total_ratio > 1)
        x0, y0, z0 = np.where(selected_attributes_total_ratio < 0)
        selected_attributes_total_ratio[x1, y1, z1] -= selected_attributes_modify_radio * 2
        selected_attributes_total_ratio[x0, y0, z0] += selected_attributes_modify_radio * 2

        unselected_attributes_total_ratio = 1 - selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] *= \
            selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_feature_ratio_index_range] *= \
            unselected_attributes_total_ratio

        return sample

    def _change_features_internal_ratio(self, sample, lon_idx, lat_idx, length, width,
                                        selected_feature_ratio_index_range,
                                        selected_attributes_modify_radio):
        noise = 0.0000000001
        selected_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                 selected_feature_ratio_index_range], axis=2)
        selected_attributes_total_ratio = np.reshape(selected_attributes_total_ratio, (length, width, 1))

        noise_matrix = np.ones((length, width, 1)) * noise  # avoid dividing by 0
        selected_attributes_total_ratio += noise_matrix
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] /= \
            selected_attributes_total_ratio

        selected_modify_matrix = np.random.choice([noise, selected_attributes_modify_radio],
                                                  (length, width, len(selected_feature_ratio_index_range)))
        for _ in range(self.exp_args.dg_modify_inner_circle_times - 1):  # a total of cycle dg_..._circle_times round
            selected_modify_matrix += np.random.choice([noise, selected_attributes_modify_radio],
                                                       (length, width, len(selected_feature_ratio_index_range)))

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
        selected_feature_ratio_index_range] += selected_modify_matrix
        now_selected_attributes_total_ratio = np.sum(
            sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range], axis=2)
        now_selected_attributes_total_ratio = np.reshape(now_selected_attributes_total_ratio, (length, width, 1))

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] /= \
            now_selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] *= \
            selected_attributes_total_ratio

        return sample

    def _change_features_total_and_internal_ratio(self, sample, lon_idx, lat_idx, length, width,
                                                  selected_feature_ratio_index_range,
                                                  unselected_feature_ratio_index_range,
                                                  modify_ratio):
        noise = 0.0000000001

        selected_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                 selected_feature_ratio_index_range], axis=2)
        unselected_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                   unselected_feature_ratio_index_range], axis=2)

        selected_attributes_total_ratio = np.reshape(selected_attributes_total_ratio, (length, width, 1))
        unselected_attributes_total_ratio = np.reshape(unselected_attributes_total_ratio, (length, width, 1))
        noise_matrix = np.ones((length, width, 1)) * noise  # avoid dividing by 0
        selected_attributes_total_ratio += noise_matrix
        unselected_attributes_total_ratio += noise_matrix

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] /= \
            selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_feature_ratio_index_range] /= \
            unselected_attributes_total_ratio

        # change inner ratio

        selected_modify_matrix = np.random.choice([noise, modify_ratio],
                                                  (length, width, len(selected_feature_ratio_index_range)))
        for _ in range(self.exp_args.dg_modify_inner_circle_times - 1):  # a total of cycle dg_..._circle_times round
            selected_modify_matrix += np.random.choice([noise, modify_ratio],
                                                       (length, width, len(selected_feature_ratio_index_range)))

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
        selected_feature_ratio_index_range] += selected_modify_matrix
        now_selected_attributes_total_ratio = np.sum(
            sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range], axis=2)
        now_selected_attributes_total_ratio = np.reshape(now_selected_attributes_total_ratio, (length, width, 1))
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] /= \
            now_selected_attributes_total_ratio

        # change total ratio
        selected_modify_matrix = np.random.choice([-modify_ratio, modify_ratio], (length, width, 1))
        selected_attributes_total_ratio += selected_modify_matrix
        x1, y1, z1 = np.where(selected_attributes_total_ratio > 1)
        x0, y0, z0 = np.where(selected_attributes_total_ratio < 0)
        selected_attributes_total_ratio[x1, y1, z1] -= modify_ratio * 2
        selected_attributes_total_ratio[x0, y0, z0] += modify_ratio * 2
        unselected_attributes_total_ratio = 1 - selected_attributes_total_ratio

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, selected_feature_ratio_index_range] *= \
            selected_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_feature_ratio_index_range] *= \
            unselected_attributes_total_ratio

        return sample

    def _change_unselected_features_internal_ratio(self, sample, lon_idx, lat_idx, length, width,
                                                   unselected_POI_feature_ratio_index_range,
                                                   unselected_modify_radio):
        noise = 0.0000000001
        unselected_POI_attributes_total_ratio = np.sum(sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                                       unselected_POI_feature_ratio_index_range], axis=2)
        unselected_POI_attributes_total_ratio = np.reshape(unselected_POI_attributes_total_ratio,
                                                           (length, width, 1))

        noise_matrix = np.ones((length, width, 1)) * noise  # avoid dividing by 0
        unselected_POI_attributes_total_ratio += noise_matrix
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_POI_feature_ratio_index_range] /= \
            unselected_POI_attributes_total_ratio

        unselected_POI_modify_matrix = np.random.uniform(noise, unselected_modify_radio,
                                                         (length, width,
                                                          len(unselected_POI_feature_ratio_index_range)))
        for _ in range(self.exp_args.dg_modify_inner_circle_times - 1):  # a total of cycle dg_..._circle_times round
            unselected_POI_modify_matrix += np.random.uniform(noise, unselected_modify_radio,
                                                              (length, width,
                                                               len(unselected_POI_feature_ratio_index_range)))

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
               unselected_POI_feature_ratio_index_range] += unselected_POI_modify_matrix
        now_unselected_POI_attributes_total_ratio = np.sum(
            sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_POI_feature_ratio_index_range],
            axis=2)
        now_unselected_POI_attributes_total_ratio = np.reshape(now_unselected_POI_attributes_total_ratio,
                                                               (length, width, 1))

        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_POI_feature_ratio_index_range] /= \
            now_unselected_POI_attributes_total_ratio
        sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, unselected_POI_feature_ratio_index_range] *= \
            unselected_POI_attributes_total_ratio

        return sample

    def _modify_sample(self, sample, lon_idx, lat_idx, length, width,
                       selected_POI_features, selected_attributes_modify_radio):
        selected_POI_feature_ratio_index_range = [self.POI_ratio_index_range[0] + idx
                                                  for idx in selected_POI_features]
        unselected_POI_features = list(set(range(self.n_POI_cate - 1)) - set(selected_POI_features))
        unselected_POI_feature_ratio_index_range = [self.POI_ratio_index_range[0] + idx
                                                    for idx in unselected_POI_features]

        # modify POI ratio
        if len(selected_POI_features) != 0:
            modify_type = random.randint(1, 3)
            if modify_type == 1:  # selected total ratio
                sample = self._change_features_total_ratio(sample, lon_idx, lat_idx, length, width,
                                                           selected_POI_feature_ratio_index_range,
                                                           unselected_POI_feature_ratio_index_range,
                                                           selected_attributes_modify_radio)
            elif modify_type == 2:  # selected inner ratio
                sample = self._change_features_internal_ratio(sample, lon_idx, lat_idx, length, width,
                                                              selected_POI_feature_ratio_index_range,
                                                              selected_attributes_modify_radio)
            elif modify_type == 3:  # both ratio
                sample = self._change_features_total_and_internal_ratio(sample, lon_idx, lat_idx, length, width,
                                                                        selected_POI_feature_ratio_index_range,
                                                                        unselected_POI_feature_ratio_index_range,
                                                                        selected_attributes_modify_radio)

        # -7: POI sum, -6: POI density, -5: POI diversity, -4: checkin sum, -3: pluto sum, -2: pluto density, -1: pluto diversity

        # change unselected feature
        if np.random.rand() < self.exp_args.dg_modify_unselected_feature_ratio_rate:
            sample = self._change_unselected_features_internal_ratio(
                sample, lon_idx, lat_idx, length, width, unselected_POI_feature_ratio_index_range,
                self.exp_args.dg_unselected_modify_ratio)

        # change the total number of POI and pluto, and then calculate the number of each category based on the proportion
        if np.random.randint(0, 1) == 0:
            POI_modify_matrix = np.random.choice(
                [1 - selected_attributes_modify_radio, 1 + selected_attributes_modify_radio], (length, width))
            sample[lon_idx:lon_idx + length, lat_idx:lat_idx + width, -7] *= POI_modify_matrix

        return sample

    def _check_cosine_similarity_is_below_threshold(self, selected_grids_vector, negative_grids_vector):
        a_norm = np.linalg.norm(selected_grids_vector)
        b_norm = np.linalg.norm(negative_grids_vector)
        cos = np.dot(selected_grids_vector, negative_grids_vector) / (a_norm * b_norm)
        return bool(cos < self.exp_args.dg_cos_sim_threshold)

    def _check_area_is_not_empty(self, lon_idx, lat_idx, length, width):
        area_feature_sum = np.sum(self.city_feature[lon_idx:lon_idx + length, lat_idx:lat_idx + width,
                                  :self.n_POI_cate])
        if area_feature_sum == 0:  # and (np.random.uniform() >= self.exp_args.dg_empty_area_reserved_probability):
            return False
        return True

    def _generate_samples(self, selected_POI_features, n_negative):
        """
        :param selected_POI_features: the POI features of regional similarity concerns (e.g. [0, 1,...,332])
        :param selected_pluto_features: the land use features of regional similarity concerns (e.g. [0, 1,...,10])
        :param change_checkin: whether to change checkin number
        :return: ternary group (selected sample, positive sample, negative sample) and the information of the selected area
        """

        # calculate mask
        mask_POI = np.zeros(self.n_POI_cate)
        mask_POI[selected_POI_features] = 1
        mask = np.concatenate((mask_POI, mask_POI))

        region_length = random.randint(self.exp_args.dg_length_and_width_range[0],
                                       self.exp_args.dg_length_and_width_range[1])
        region_width = random.randint(self.exp_args.dg_length_and_width_range[0],
                                      self.exp_args.dg_length_and_width_range[1])
        # TODO: the rationality of the initial selection area
        while True:
            start_lon_idx = random.randint(0, self.n_lon_len - (region_length + 1))
            start_lat_idx = random.randint(0, self.n_lat_len - (region_width + 1))
            if self._check_area_is_not_empty(start_lon_idx, start_lat_idx, region_length, region_width):
                break

        selected_sample = self.city_feature.copy()

        positive_sample = self._modify_sample(self.city_feature.copy(),
                                              start_lon_idx, start_lat_idx,
                                              region_length, region_width,
                                              selected_POI_features,
                                              self.exp_args.dg_positive_modify_ratio)

        # only one negative sample, for triplet train
        if n_negative == 1:
            # decide on the way to generate negative sample
            use_another_region_as_negative_flag = bool(
                np.random.uniform() < self.exp_args.dg_negative_choice_probability)
            negative_lon_idx = -1
            negative_lat_idx = -1

            if use_another_region_as_negative_flag:
                negative_sample = self.city_feature.copy()

                selected_grids_vector = np.array(self.get_area_grids_list(start_lon_idx, start_lat_idx,
                                                                          region_length, region_width))
                while True:
                    negative_lon_idx = random.randint(0, self.n_lon_len - (region_length + 1))
                    negative_lat_idx = random.randint(0, self.n_lat_len - (region_width + 1))
                    negative_grids_vector = np.array(self.get_area_grids_list(negative_lon_idx, negative_lat_idx,
                                                                              region_length, region_width))
                    if self._check_cosine_similarity_is_below_threshold(selected_grids_vector, negative_grids_vector):
                        break

            else:
                negative_sample = self._modify_sample(self.city_feature.copy(), start_lon_idx, start_lat_idx,
                                                      region_length, region_width,
                                                      selected_POI_features,
                                                      self.exp_args.dg_negative_modify_ratio)

            return [selected_sample, positive_sample, negative_sample], mask, \
                start_lon_idx, start_lat_idx, negative_lon_idx, negative_lat_idx, region_length, region_width

        # multiple negative samples, for test
        else:
            negative_sample_list = []
            negative_lon_idx_list = []
            negative_lat_idx_list = []
            for _ in range(n_negative):
                # decide on the way to generate negative sample
                use_another_region_as_negative_flag = bool(
                    np.random.uniform() < self.exp_args.dg_negative_choice_probability)
                negative_lon_idx = -1
                negative_lat_idx = -1

                if use_another_region_as_negative_flag:
                    negative_sample = self.city_feature.copy()

                    selected_grids_vector = np.array(self.get_area_grids_list(start_lon_idx, start_lat_idx,
                                                                              region_length, region_width))
                    while True:
                        negative_lon_idx = random.randint(0, self.n_lon_len - (region_length + 1))
                        negative_lat_idx = random.randint(0, self.n_lat_len - (region_width + 1))
                        negative_grids_vector = np.array(self.get_area_grids_list(negative_lon_idx, negative_lat_idx,
                                                                                  region_length, region_width))
                        if self._check_cosine_similarity_is_below_threshold(selected_grids_vector,
                                                                            negative_grids_vector):
                            break

                else:
                    negative_sample = self._modify_sample(self.city_feature.copy(), start_lon_idx, start_lat_idx,
                                                          region_length, region_width,
                                                          selected_POI_features,
                                                          self.exp_args.dg_negative_modify_ratio)
                negative_sample_list.append(negative_sample)
                negative_lon_idx_list.append(negative_lon_idx)
                negative_lat_idx_list.append(negative_lat_idx)

            return [selected_sample, positive_sample, negative_sample_list], mask, start_lon_idx, start_lat_idx, \
                negative_lon_idx_list, negative_lat_idx_list, region_length, region_width


class TrainDataSet(ForModifyDataSet):
    # https://pytorch.org/docs/1.9.0/data.html?highlight=dataset#torch.utils.data.Dataset
    # refer to the above link to quick load
    def __init__(self, city_base_info, size, city_feature_path, exp_args):
        """
        :param size: size = batch_size * number of batches per epoch
        """
        super().__init__(city_base_info=city_base_info, size=size,
                         city_feature_path=city_feature_path, exp_args=exp_args)

    def __getitem__(self, item):
        choice_key = random.choice(self._choice_keys)
        choice = self.choice_corresponding_features[choice_key]
        POI_features = choice['POI']
        return self._generate_samples(POI_features, n_negative=1)

    def __len__(self):
        return self._size


class TestDataSet(ForModifyDataSet):
    def __init__(self, city_base_info, size, n_test_negative_size, city_feature_path, exp_args):
        """
        :param size: number of test samples
        :param n_test_negative_size: how many negative samples are used for a test case
        """

        super().__init__(city_base_info=city_base_info, size=size,
                         city_feature_path=city_feature_path, exp_args=exp_args)
        self.n_test_negative_size = n_test_negative_size

    def __getitem__(self, item):
        choice_key = random.choice(self._choice_keys)
        choice = self.choice_corresponding_features[choice_key]
        POI_features, land_use_features = choice['POI'], choice['land_use']
        return self._generate_samples(POI_features, n_negative=self.n_test_negative_size)

    def __len__(self):
        return self._size
