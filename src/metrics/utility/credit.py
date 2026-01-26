import numpy as np
import pandas as pd

L = 24  # term [months]
L_GD = 0.75  # loss given default
K = 3  # times income
CL_MAX = 25000  # maximum credit line


# Helpers
def _costumer_montly_payment(credit_line, rate, term):
    return credit_line * ((rate * (1 + rate) ** term) / ((1 + rate) ** term - 1))


def _present_value(x, rate, term):
    return (x / rate) * (1 - (1 / (1 + rate) ** term))


def credit_utility(
    p, y, credit_features: pd.DataFrame, average_credit_quantities: list, dataset: str
):
    if dataset == "kaggle":
        # Table 3 values (Kaggle dataset)
        INT_R = 0.0479  # financial institution lending rate
        INT_CF = 0.0294  # financial institution cost of funds
    if dataset == "pakdd":
        INT_R = 0.63
        INT_CF = 0.165

    # Per-customer quantities
    y_vec = y
    p_vec = p
    inc_vec = credit_features["MonthlyIncome"]  # income [months] (available)
    inc_vec_safe = np.maximum(inc_vec, 1e-12)
    debt_vec = credit_features["DebtRatio"]  # debt ratio (available)

    # training set averages/quantities
    CL_AVG = average_credit_quantities[0]  # average credit line
    PI_0 = average_credit_quantities[1]  # prior negative (no default) rate
    PI_1 = average_credit_quantities[2]  # prior positive (default) rate
    R_AVG = average_credit_quantities[3]  # average profit

    payment = _costumer_montly_payment(K * inc_vec_safe, INT_R, L)  # vector
    pm_debt = np.minimum(payment / inc_vec_safe, 1.0 - debt_vec)  # elementwise min
    cl_max_debt = _present_value(inc_vec_safe * pm_debt, INT_R, L)
    cl = np.minimum(
        np.minimum(K * inc_vec_safe, CL_MAX),
        cl_max_debt,
    )
    ca_fp = (
        -R_AVG * PI_0 + CL_AVG * L_GD * PI_1
    )  # - gain through alternative other customer

    a = _costumer_montly_payment(cl, INT_R, L)  # customer monthly payment
    r = _present_value(a, INT_CF, L) - cl  # profit through the customer
    c_FP = (
        r + ca_fp
    )  # missed profit: missed customer profit - gain through alternative customer
    c_FN = cl * L_GD  # default loss: credit line * loss given default

    threshold = c_FP / (c_FP + c_FN)

    mask_fn = (y_vec == 1) & (p_vec < threshold)
    mask_fp = (y_vec == 0) & (p_vec >= threshold)

    cost = c_FN[mask_fn].sum() + c_FP[mask_fp].sum()
    util = -cost

    return util, threshold
