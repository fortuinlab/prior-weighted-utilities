import numpy as np


def ece_binary_classification(y_true, y_prob, n_bins=10):
    """
    Compute ECE with equal-width bins.
    y_true: array of shape (n,) with labels {0,1}
    y_prob: array of shape (n,) with predicted probabilities for class 1
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for m in range(n_bins):
        mask = binids == m
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return ece


def ace_from_intervals(y_true, intervals, coverages=None, weights=None):
    """
    Compute the ECE regression analog, the average calibration error, from provided prediction intervals.

    Parameters
    ----------
    y_true : (n,) array
        True targets.
    intervals : dict[float -> (lower, upper)]
        Keys are nominal coverages in (0,1) (e.g., 0.5, 0.8, 0.9, 0.95).
        Values are tuples of arrays (lower, upper), each shape (n,) - the prediction intervals corresponding to the coverage.
    coverages : iterable or None
        If given, restrict/order the coverages used; otherwise use sorted keys(intervals).
    weights : None or array-like of same length as coverages
        Optional nonnegative weights; normalized internally.

    Returns
    -------
    ace : float
    levels : (M,) array of coverages used
    empirical : (M,) array of empirical coverages
    """
    y = np.asarray(y_true)
    levels = np.array(
        sorted(intervals.keys()) if coverages is None else list(coverages), dtype=float
    )

    empirical = np.empty_like(levels)
    for k, c in enumerate(levels):
        lower, upper = intervals[c]
        lower, upper = np.asarray(lower), np.asarray(upper)
        empirical[k] = ((y >= lower) & (y <= upper)).mean()

    if weights is None:
        w = np.ones_like(levels)
    else:
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.shape != levels.shape:
            raise ValueError("weights must be nonnegative and same length as coverages")
    w = w / w.sum()

    ace = float(np.sum(w * np.abs(empirical - levels)))
    return ace, levels, empirical


# z-values for central two-sided Normal intervals
_Z_FROM_COVERAGE = {
    0.50: 0.67448975,
    0.68: 0.99445788,  # ~1σ
    0.80: 1.28155157,
    0.90: 1.64485363,
    0.95: 1.95996398,
    0.98: 2.32634787,
    0.99: 2.57582930,
}


def ece_regression(
    y_true, mu, sigma, coverages=(0.5, 0.8, 0.9, 0.95), weights=None, eps=1e-12
):
    """
    Compute the ECE regression analog, the average calibration error, from provided prediction intervals,
    for the Gaussian case.

    Returns
    -------
    ace : float
    levels : (M,) array
    empirical : (M,) array
    """
    y = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.maximum(np.asarray(sigma), eps)

    intervals = {}
    for c in coverages:
        c = float(c)
        if c not in _Z_FROM_COVERAGE:
            raise ValueError(f"Coverage {c} not in lookup; add its z-value.")
        z = _Z_FROM_COVERAGE[c]
        lower = mu - z * sigma
        upper = mu + z * sigma
        intervals[c] = (lower, upper)

    return ace_from_intervals(y, intervals, coverages=coverages, weights=weights)
