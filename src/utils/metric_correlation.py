from typing import Sequence
from typing import List

import numpy as np


def compute_metric_correlation(
    ranking_m: Sequence[str],
    ranking_u: Sequence[str],
) -> float:
    """
    Compute Kendall's tau between two strict rankings.

    Parameters
    ----------
    ranking_m : sequence of str
        Model names ordered from best -> worst under metric M.
    ranking_u : sequence of str
        Model names ordered from best -> worst under utility U.

    Returns
    -------
    tau : float
        Kendall's tau (no ties assumed).

    Notes
    -----
    Assumes both rankings are permutations of the same set of items.
    """

    ranking_m = list(ranking_m)
    ranking_u = list(ranking_u)

    n = len(ranking_m)
    if n < 2:
        return 0.0, 0.0

    pos_u = {item: i for i, item in enumerate(ranking_u)}
    r_u = np.array([pos_u[item] for item in ranking_m], dtype=float)

    # Kendall: count inversions in r_u (since r_m is sorted by construction)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if r_u[i] > r_u[j]:
                inv += 1

    total_pairs = n * (n - 1) / 2
    concordant = total_pairs - inv
    discordant = inv
    tau = (concordant - discordant) / total_pairs  # = 1 - 2*inv/total_pairs

    return float(tau)


def top1_agreement(ranking_m: List[str], ranking_u: List[str]) -> int:
    """1 if both rankings agree on the best model, 0 otherwise."""
    return int(ranking_m[0] == ranking_u[0])


def topk_agreement(ranking_m: List[str], ranking_u: List[str], k: int) -> float:
    """Jaccard similarity of the top-k sets from two rankings."""
    set_m = set(ranking_m[:k])
    set_u = set(ranking_u[:k])
    return len(set_m & set_u) / len(set_m | set_u)
