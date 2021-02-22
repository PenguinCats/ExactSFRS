# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  draw_map.py
@Time    :  2021/2/22 0022 19:19
@Author  :  Binjie Zhang (bj_zhang@seu.edu.cn)
@Desc    :  None
"""

from bottle import template


# var target = [[40.042054, 116.273303], [40.040419, 116.273721], [40.040764, 116.275824], [40.042374, 116.275491]];
# var search_result = [[[40.041054, 116.272303], [40.039419, 116.272721], [40.039764, 116.274824], [40.041374, 116.274491]],
#                      [[40.043054, 116.275303], [40.042419, 116.275721], [40.042764, 116.277824], [40.044374, 116.277491]]];


def draw_search_result_by_search_result(target_region, search_result):
    target_region = str(target_region)
    search_result = str(search_result)

    html = template('./tool/draw_map.html', template_settings={'syntax': '<{% %}> %% {%{ }%}'},
                    target_var=target_region, search_result_var=search_result)

    return html
