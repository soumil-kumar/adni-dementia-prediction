import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss, precision_recall_curve)


def best_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    # Prefer recall for tied F1 values; then the lowest threshold.
    winner = sorted(best, key=lambda i: (-recall[i], thresholds[i]))[0]
    return float(thresholds[winner])


def scores(y, probabilities, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0,1]).ravel()
    result = dict(accuracy=float(accuracy_score(y,predicted)),
        precision=float(precision_score(y,predicted,zero_division=0)),
        recall=float(recall_score(y,predicted,zero_division=0)),
        f1=float(f1_score(y,predicted,zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        balanced_accuracy=float(balanced_accuracy_score(y,predicted)),
        confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],
        brier=float(brier_score_loss(y,p)), n=len(y), positives=int(y.sum()))
    result['roc_auc'] = float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    result['average_precision'] = float(average_precision_score(y,p)) if y.sum() else None
    return result
