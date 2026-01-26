import numpy as np


def binary_decision_utility(p, y, c: float = 0.1, threshold: float = None):

    if threshold is None:
        # pick Bayes-optimal threshold
        threshold = c

    cost = (1 - c) * ((y == 1) & (p < threshold)).sum() + c * (
        (y == 0) & (p >= threshold)
    ).sum()
    return -cost


def binary_decision_pwu(p, y, alpha_param: float, beta_param: float):
    """
    Monte Carlo estimate of l(p,y) with c ~ Beta(a,b).
    Reuses binary_decision_utility.
    """
    p = np.asarray(p)
    y = np.asarray(y)
    rng = np.random.default_rng(42)

    c = rng.beta(alpha_param, beta_param, size=10000)
    costs = np.array(
        [-binary_decision_utility(p, y, c=float(ci), threshold=float(ci)) for ci in c]
    )
    return float(costs.mean())
