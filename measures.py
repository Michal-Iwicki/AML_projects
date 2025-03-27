import numpy as np

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f_measure(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return (1 + beta**2) * (p * r) / (beta**2 * p + r)

def balanced_accuracy(y_true, y_pred):
    recall_pos = recall(y_true, y_pred)
    recall_neg = recall(1 - y_true, 1 - y_pred)
    return (recall_pos + recall_neg) / 2

def auc_roc(y_true, y_scores):
    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    tpr = np.cumsum(y_true) / np.sum(y_true)
    fpr = np.cumsum(1 - y_true) / np.sum(1 - y_true)
    return np.trapz(tpr, fpr)

def auc_pr(y_true, y_scores):
    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    precision_vals = np.cumsum(y_true) / np.arange(1, len(y_true) + 1)
    recall_vals = np.cumsum(y_true) / np.sum(y_true)
    return np.trapz(precision_vals, recall_vals)

# To be verified