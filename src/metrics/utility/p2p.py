import numpy as np

initial_budget = 15000
INT_CF = 0.0294  # financial institution cost of funds


# Helpers
def _costumer_montly_payment(credit_line, rate, term):
    return credit_line * ((rate * (1 + rate) ** term) / ((1 + rate) ** term - 1))


def _present_value(x, rate, term):
    return (x / rate) * (1 - (1 / (1 + rate) ** term))


def p2p_utility(p, y, credit_lines, terms, lending_rates):

    p = np.asarray(p)
    y = np.asarray(y)
    credit_lines = np.asarray(credit_lines)
    terms = np.asarray(terms)
    lending_rates = np.asarray(lending_rates) / 100.0

    a = _costumer_montly_payment(
        credit_lines, lending_rates, terms
    )  # customer monthly payment
    r = _present_value(a, INT_CF, terms) - credit_lines  # profit through the customer

    budget = initial_budget
    okay = True
    idx = []
    idx_count = 0
    while okay is True:
        pick = np.argsort(-p)[idx_count]
        if budget - credit_lines[pick] >= 0:
            budget -= credit_lines[pick]
            idx.append(pick)
            idx_count += 1
        else:
            okay = False

    utility = 0
    k = len(idx)
    k_frac = k / len(y)
    if k > 0:
        idx = np.array(idx)
        for i in idx:
            if y[i] == 1:
                utility += r[i]
            else:
                utility -= credit_lines[i]
    hit_rate = np.sum(y[idx]) / k if k > 0 else 0
    return utility, k_frac, hit_rate
