# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  exact_sfrs.py
@Time    :  2021/2/20 0020 12:24
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

import torch
import copy
from queue import PriorityQueue
from args import args
from Triplet.triplet import global_max_pooling


def _cal_distance(v1, v2):
    return torch.nn.functional.mse_loss(v1, v2, reduction='sum')


def _cal_bound(vq, v_union, v_intersection):
    # Tip: 个人认为文中的 k1, k2 定义中少了等号,即,这里应当使用 torch.le 和 torch.ge

    # k2 = torch.lt(v_union, vq)
    k2 = torch.le(v_union, vq)
    bound = torch.sum(torch.pow(vq[k2] - v_union[k2], 2))

    if v_intersection is not None:
        # k1 = torch.gt(v_intersection, vq)
        k1 = torch.ge(v_intersection, vq)
        bound += torch.sum(torch.pow(vq[k1] - v_intersection[k1], 2))

    return bound


class _SpaceSet(object):
    def __init__(self, l_min, r_min, b_min, t_min, l_max, r_max, b_max, t_max):
        self.l_min, self.r_min, self.b_min, self.t_min = l_min, r_min, b_min, t_min
        self.l_max, self.r_max, self.b_max, self.t_max = l_max, r_max, b_max, t_max

    def __lt__(self, other):
        # 随便定义一个比较函数，这样就可以放进优先队列里了
        return self.l_max < other.l_max

    def has_intersection(self):
        # return (self.r_min - self.l_max) >= args.filter_size[0] and (self.t_min - self.b_max) > args.filter_size[0]
        # return self.r_min >= self.l_max and self.t_min >= self.b_max
        return self.r_min - self.l_max >= 2 and self.t_min - self.b_max >= 2

    def has_unique_region(self):
        return self.l_min == self.l_max and self.r_min == self.r_max and \
               self.t_min == self.t_max and self.b_min == self.b_max

    def split(self):
        m_dis = max(self.l_max - self.l_min, self.r_max - self.r_min, self.b_max - self.b_min, self.t_max - self.t_min)
        a = copy.copy(self)
        b = copy.copy(self)
        if self.l_max - self.l_min == m_dis:
            a.l_max = a.l_min + int(m_dis//2)
            b.l_min = a.l_max + 1
            b.r_min = max(b.r_min, b.l_min)
        elif self.r_max - self.r_min == m_dis:
            a.r_max = a.r_min + int(m_dis//2)
            a.l_max = min(a.l_max, a.r_max)
            b.r_min = a.r_max + 1
        elif self.b_max - self.b_min == m_dis:
            a.b_max = a.b_min + int(m_dis // 2)
            b.b_min = a.b_max + 1
            b.t_min = max(b.t_min, b.b_min)
        else:
            # self.t_max - self.t_min == m_dis:
            a.t_max = a.t_min + int(m_dis // 2)
            a.b_max = min(a.b_max, a.t_max)
            b.t_min = a.t_max + 1

        return a, b

    def too_small(self, width, height):
        return self.r_max - self.l_min < width * 0.5 or self.t_max - self.b_min < height * 0.5

    def too_big(self, width, height):
        return self.r_min - self.l_max > width * 2 or self.t_min - self.b_max > height * 2

    def has_proper_size(self, width, height):
        return not self.too_small(width, height) and not self.too_big(width, height)


def cal_origin_index(lef, rig, bot, top):
    # 计算特征图上的坐标对应原图坐标.显而易见会有数个格子的误差,不过考虑到格子较小,可以容忍
    i = len(args.filter_size) - 1
    while i >= 0:
        lef = lef * args.stride[i]
        rig = rig * args.stride[i]
        bot = bot * args.stride[i]
        top = top * args.stride[i]
        i -= 1
    return [lef, rig, bot, top]


class ExactSFRS(object):
    def __init__(self, search_space_feature):
        self.feature = search_space_feature
        self.height = self.feature.shape[2]
        self.width = self.feature.shape[3]

        self.space_set_p = _SpaceSet(0, 0, 0, 0, self.width - 1, self.width - 1, self.height - 1, self.height - 1)

    def search(self, target_feature):
        with torch.no_grad():
            target_feature_pooling = global_max_pooling(target_feature)
            ans = []

            q = PriorityQueue()
            q.put((_cal_bound(target_feature_pooling, global_max_pooling(self.feature), None), self.space_set_p))

            while q.not_empty:
                val, space_set = q.get()

                # TODO 预测的区域不重叠
                if space_set.has_unique_region():
                    if space_set.has_proper_size(target_feature.shape[3], target_feature.shape[2]):
                        ans.append(cal_origin_index(space_set.l_min, space_set.r_min, space_set.b_min, space_set.t_min))
                    if len(ans) == args.N:
                        return ans

                else:
                    sub_region_a, sub_region_b = space_set.split()

                    if sub_region_a.has_proper_size(target_feature.shape[3], target_feature.shape[2]):
                        v_a_union = global_max_pooling(self.feature[:, :,
                                                       sub_region_a.b_min:sub_region_a.t_max+1,
                                                       sub_region_a.l_min:sub_region_a.r_max+1])
                        if sub_region_a.has_intersection():
                            v_a_intersection = global_max_pooling(self.feature[:, :,
                                                                               sub_region_a.b_max+1:sub_region_a.t_min,
                                                                               sub_region_a.l_max+1:sub_region_a.r_min])
                            bound_a = _cal_bound(target_feature_pooling, v_a_union, v_a_intersection)
                        else:
                            bound_a = _cal_bound(target_feature_pooling, v_a_union, None)
                        q.put((bound_a, sub_region_a))

                    if sub_region_b.has_proper_size(target_feature.shape[3], target_feature.shape[2]):
                        v_b_union = global_max_pooling(self.feature[:, :,
                                                                    sub_region_b.b_min:sub_region_b.t_max+1,
                                                                    sub_region_b.l_min:sub_region_b.r_max+1])
                        if sub_region_b.has_intersection():
                            v_b_intersection = global_max_pooling(self.feature[:, :,
                                                                               sub_region_b.b_max+1:sub_region_b.t_min,
                                                                               sub_region_b.l_max+1:sub_region_b.r_min])
                            bound_b = _cal_bound(target_feature_pooling, v_b_union, v_b_intersection)
                        else:
                            bound_b = _cal_bound(target_feature_pooling, v_b_union, None)
                        q.put((bound_b, sub_region_b))

        return ans
