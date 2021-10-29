import math


def hit_ratio(v, predict_list, k=0):
    if k == 0:
        if v in predict_list:
            return 1
    else:
        if v in predict_list[:k]:
            return 1

    return 0


def mrr(v, predict_list, k=0):
    if k == 0:
        k = len(predict_list)

    for i in range(k):
        if v == predict_list[i]:
            return 1 / (i+1)

    return 0
