import numpy as np
from sklearn.metrics import accuracy_score


def rauc_binary_classification(y_true, y_pred, uncertainty, num_points=101):
    """
    Compute the retention curve (1-Acc vs r) and its AUC.
    Args:
        y_true:               (n,) true labels {0,1}
        y_pred:               (n,) model hard predictions {0,1}
        uncertainty:          (n,) uncertainty scores (higher = more uncertain)
        num_points:           number of r-points in [0,1]
    Returns:
        r_grid, f1_grid, auc
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    uncertainty = np.asarray(uncertainty).astype(float)
    n = len(y_true)
    order = np.argsort(-uncertainty)  # most uncertain first

    corrected = y_pred.copy()
    r_grid = np.linspace(0.0, 1.0, num_points)
    err_grid = np.zeros_like(r_grid)

    err_grid[0] = 1 - accuracy_score(y_true, y_pred)
    prev_k = 0
    for i in range(1, num_points):
        k = int(round(r_grid[i] * n))
        if k > prev_k:
            idx = order[prev_k:k]
            corrected[idx] = y_true[idx]  # oracle fixes top-k
            prev_k = k
        err_grid[i] = 1 - accuracy_score(y_true, corrected)

    auc = np.trapezoid(err_grid, r_grid)
    return r_grid, err_grid, float(auc)


def rauc_regression(y_true, y_pred, uncertainty, num_points=101):
    """
    Compute the retention curve (MSE vs r) and its AUC.
    Args:
        y_true:               (n,) true labels
        y_pred:               (n,) model hard predictions
        uncertainty:          (n,) uncertainty scores (higher = more uncertain)
        num_points:           number of r-points in [0,1]
    Returns:
        r_grid, f1_grid, auc
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    uncertainty = np.asarray(uncertainty).astype(float)
    n = len(y_true)
    order = np.argsort(-uncertainty)  # most uncertain first

    corrected = y_pred.copy()
    r_grid = np.linspace(0.0, 1.0, num_points)
    err_grid = np.zeros_like(r_grid)

    err_grid[0] = ((y_true - y_pred) ** 2).mean()
    prev_k = 0
    for i in range(1, num_points):
        k = int(round(r_grid[i] * n))
        if k > prev_k:
            idx = order[prev_k:k]
            corrected[idx] = y_true[idx]  # oracle fixes top-k
            prev_k = k
        err_grid[i] = ((y_true - corrected) ** 2).mean()

    auc = np.trapezoid(err_grid, r_grid)
    return r_grid, err_grid, float(auc)
