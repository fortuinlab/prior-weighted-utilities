import numpy as np
from sklearn.metrics import roc_auc_score


def error_detection_binary_classification(y_true, y_pred_labels, uncertainty):
    """
    Error detection AUROC per Bouvier et al.: use uncertainty as score to detect errors.
    Args:
        y_true:        (n,) true labels {0,1}
        y_pred_labels: (n,) predicted labels {0,1}
        uncertainty:   (n,) uncertainty scores (higher = more uncertain)
    Returns:
        auroc
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_labels = np.asarray(y_pred_labels).astype(int)
    u = np.asarray(uncertainty).astype(float)

    # Error indicator: 1 if model is wrong, else 0
    y_error = (y_pred_labels != y_true).astype(int)

    return roc_auc_score(y_error, u)


def error_detection_regression(
    y_true, y_pred, uncertainty, threshold, *, error_kind="abs", scale=None
):
    """
    AUROC for regression error detection:
      positive class = 'unacceptable error' (error > tau)
      score          = uncertainty (higher = more likely error)

    Returns: auroc (float)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    errors = y_pred - y_true
    rel_errors = np.abs(errors / (y_true + 1e-10))
    y_err = (rel_errors > float(threshold)).astype(int)
    # handle degenerate case: all 0s or all 1s
    if y_err.min() == y_err.max():
        return 0.0
    return roc_auc_score(y_err, u)
