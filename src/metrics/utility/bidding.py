import numpy as np


def bidding_utility(y_true, y_pred, y_std, prices_da, prices_balancing, alpha, beta):

    # go from relative to absolute alpha
    alphas_min = y_std**2
    alphas_max = y_std**2 + (beta - y_pred) ** 2
    alphas = alphas_min + (alphas_max - alphas_min) * alpha

    case1 = prices_da > prices_balancing
    case2 = ~case1

    bids1 = np.minimum(y_pred + np.sqrt(alphas - y_std**2), beta)
    bids2 = np.maximum(y_pred - np.sqrt(alphas - y_std**2), 0)

    bids = np.zeros_like(y_true)
    bids[case1] = bids1[case1]
    bids[case2] = bids2[case2]

    # count how many times zero bids are made
    num_zeros = np.sum(bids == 0)

    profit = prices_da * bids + prices_balancing * (y_true - bids)

    return np.sum(profit), np.cumsum(profit), num_zeros
