import numpy as np
import pandas as pd

L = 24  # term [months]
K = 3  # times income
CL_MAX = 25000  # maximum credit line


# Helpers
def _costumer_montly_payment(credit_line, rate, term):
    return credit_line * ((rate * (1 + rate) ** term) / ((1 + rate) ** term - 1))


def _present_value(x, rate, term):
    return (x / rate) * (1 - (1 / (1 + rate) ** term))


def compute_average_credit_quantities(df: pd.DataFrame, dataset: str):

    if dataset == "kaggle":
        INT_R = 0.0479  # financial institution lending rate
        INT_CF = 0.0294  # financial institution cost of funds
    if dataset == "pakdd":
        INT_R = 0.63
        INT_CF = 0.165

    # CL_AVG
    inc_vec = df["MonthlyIncome"]  # income [months] (available)
    debt_vec = df["DebtRatio"]  # debt ratio (available)
    inc_vec_safe = np.maximum(inc_vec, 1e-12)
    payment = _costumer_montly_payment(K * inc_vec_safe, INT_R, L)  # vector
    pm_debt = np.minimum(payment / inc_vec_safe, 1.0 - debt_vec)  # elementwise min
    cl_max_debt = _present_value(inc_vec_safe * pm_debt, INT_R, L)
    cl = np.minimum(
        np.minimum(K * inc_vec_safe, CL_MAX),
        cl_max_debt,
    )

    # PI_1, PI_0
    if dataset == "kaggle":
        y_vec = df["SeriousDlqin2yrs"].to_numpy()
    if dataset == "pakdd":
        y_vec = df["y"].to_numpy()
    pi_0 = np.mean(y_vec == 0)
    pi_1 = np.mean(y_vec == 1)

    # R_AVG
    a = _costumer_montly_payment(cl, INT_R, L)  # customer monthly payment
    r = _present_value(a, INT_CF, L) - cl  # profit through the customer

    return cl.mean(), pi_0, pi_1, r.mean()
