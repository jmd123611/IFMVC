import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    dimension = int(max(y_pred.max(), y_true.max()) + 1)
    weights = np.zeros((dimension, dimension), dtype=np.int64)
    for index in range(y_pred.size):
        weights[y_pred[index], y_true[index]] += 1
    rows, cols = linear_sum_assignment(weights.max() - weights)
    return float(weights[rows, cols].sum() / y_pred.size)


def purity(y_true, y_pred):
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.max(contingency, axis=0).sum() / len(y_true))


def f_measure(y_true, y_pred):
    (_, false_positive), (false_negative, true_positive) = metrics.cluster.pair_confusion_matrix(
        y_true, y_pred
    )
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return float(2.0 * precision * recall / (precision + recall))


def evaluate(y_true, y_pred):
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "PUR": purity(y_true, y_pred),
        "F1": f_measure(y_true, y_pred),
    }
