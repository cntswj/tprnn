'''
Evaluation metrics functions.
'''
import numpy as np
import collections


def top_k_accuracy(y_prob, y, k=10):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return acc
