'''
Evaluation metrics functions.
'''
import math
import numpy as np
import collections

# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def _retype(y_prob, y):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)

    return y_prob, y


def top_k_accuracy(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)


def roc_auc(y_prob, y):
    n_classes = y_prob.shape[1]
    y = label_binarize(y, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y.ravel(), y_prob.ravel())
    return auc(fpr, tpr)


def log_prob(y_prob, y):
    scores = []
    for p_, y_ in zip(y_prob, y):
        assert abs(np.sum(p_) - 1) < 1e-8
        scores += [-math.log(p_[y_]) + 1e-8]
        print p_, y_

    return sum(scores) / len(scores)


def portfolio(y_prob, y, k_list=None):
    y_prob, y = _retype(y_prob, y)
    scores = {'auc': roc_auc(y_prob, y),
              # 'log-prob': log_prob(y_prob, y)
              }
    for k in k_list:
        scores['accuracy@' + str(k)] = top_k_accuracy(y_prob, y, k=k)

    return scores
