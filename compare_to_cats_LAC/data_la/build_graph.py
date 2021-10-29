#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_graph.py
# @Author: Binjie Zhang (bj_zhang@seu.edu.cn)
# @Date  : 2021/7/7
# @Desc  : None

import dgl
import torch


def _get_grid_adjacent_grid(n_lon_len, n_lat_len):
    src_ids = []
    dst_ids = []
    for i in range(n_lon_len):
        for j in range(n_lat_len):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= i + dx < n_lon_len and 0 <= j + dy < n_lat_len and not (dx == 0 and dy == 0):
                        src_ids.append(i * n_lat_len + j)
                        dst_ids.append((i + dx) * n_lat_len + j + dy)
    return src_ids, dst_ids


def build_graph(n_lon_len, n_lat_len):
    src_ids, dst_ids = _get_grid_adjacent_grid(n_lon_len=n_lon_len, n_lat_len=n_lat_len)

    g = dgl.graph((src_ids, dst_ids), num_nodes=n_lon_len*n_lat_len)

    num_nodes = len(g.nodes())

    num_graph_relation = 1

    graph_edge_types = torch.zeros(len(g.edges()[0])).long()

    graph_info = {
        'num_nodes': num_nodes,
        'num_graph_relation': num_graph_relation,
        'graph_edge_types': graph_edge_types
    }

    return g, graph_info
