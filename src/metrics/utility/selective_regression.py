import numpy as np


def selective_prediction_utility(mu, sd, y, abst_cost):
    trade = sd**2 <= abst_cost
    abst = ~trade
    n_abst = np.sum(abst)
    cost_abstain = abst_cost * n_abst
    cost_mse = (y - mu) ** 2

    cost = np.zeros_like(y, dtype=float)
    cost[trade] = cost_mse[trade]
    cost[abst] = cost_abstain
    return -np.mean(cost), n_abst


def selective_prediction_pwu(
    mu, sd, y, alpha_param: float, beta_param: float
):
    """
    Monte Carlo estimate of l(p,y) with lambda ~ Beta(a,b).
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    sd = np.asarray(sd)
    rng = np.random.default_rng(42)

    # data variance
    y_var = ((y - np.mean(y)) ** 2).mean()

    lambda_factor_vec = rng.beta(alpha_param, beta_param, size=10000)
    lambda_vec = lambda_factor_vec * y_var
    costs = np.array(
        [
            -selective_prediction_utility(mu, sd, y, abst_cost=float(li))[0]
            for li in lambda_vec
        ]
    )
    return float(costs.mean())
