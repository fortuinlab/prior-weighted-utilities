import numpy as np


def topk_utility_binary_classification(p, y, k: int = 3):
    p = np.asarray(p)
    y = np.asarray(y)
    idx = np.argsort(-p)[:k]  # indices of top-k probabilities
    picked_y = y[idx]
    utility = (1 / k) * np.sum(picked_y == 1)
    return utility


def topk_utility_binary_classification_pwu(
    p, y, alpha_param: float, beta_param: float
):
    """
    Monte Carlo estimate of l(p,y) with k_frac ~ Beta(alpha_param,beta_param).
    """
    p = np.asarray(p)
    y = np.asarray(y)
    rng = np.random.default_rng(42)

    k_frac = rng.beta(alpha_param, beta_param, size=10000)
    costs = np.array(
        [
            -topk_utility_binary_classification(p, y, k=int(np.ceil(len(y) * ki)))
            for ki in k_frac
        ]
    )
    return float(costs.mean())


def topk_utility_regression(mu, sd, y, k: int = 3, gamma: float = 0.0):
    mu = np.asarray(mu)
    sd = np.asarray(sd)
    y = np.asarray(y)

    ranking = mu - gamma * sd**2

    idx = np.argsort(-ranking)[:k]  # indices of predicted top-k values
    picked_y = y[idx]
    picked_sd = sd[idx]
    utility = (1 / k) * np.sum((picked_y - gamma * picked_sd**2))
    return utility


def topk_utility_regression_pwu(
    mu,
    sd,
    y,
    alpha_param_k: float,
    beta_param_k: float,
    alpha_param_gamma: float,
    beta_param_gamma: float,
):
    """
    Monte Carlo estimate of l(p,y) with k_frac ~ Beta(a,b), gamma ~ Beta(a,b).
    """
    mu = np.asarray(mu)
    sd = np.asarray(sd)
    y = np.asarray(y)
    rng = np.random.default_rng(42)

    # data variance
    y_var = ((y - np.mean(y)) ** 2).mean()

    k_frac = rng.beta(alpha_param_k, beta_param_k, size=10000)
    gamma_factor_vec = rng.beta(alpha_param_gamma, beta_param_gamma, size=10000)
    gamma_vec = gamma_factor_vec * y_var
    costs = np.array(
        [
            -topk_utility_regression(
                mu, sd, y, k=int(np.ceil(len(y) * k_frac[i])), gamma=gamma_vec[i]
            )
            for i in range(len(k_frac))
        ]
    )
    return float(costs.mean())
