import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta, pearsonr
from sklearn.metrics import (
    accuracy_score,  # high is good
    log_loss,        # low is good
)

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.utils import (  # noqa: E402
    compute_metric_correlation,
    top1_agreement,
    topk_agreement,
)


# ---------------------------------------------------------------------------
# Multiclass metrics
# ---------------------------------------------------------------------------
def brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Brier score, summed over classes:
        BS = (1/n) Σ_i Σ_j (p_ij - 1{y_i = j})^2.
    """
    n, K = probs.shape
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y_true] = 1.0
    return float(((probs - y_onehot) ** 2).sum(axis=1).mean())


def ece_multiclass(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Class-wise (one-vs-all) ECE, averaged over classes.

    For each class j, treat the problem as binary (class j vs rest) and
    compute the standard equal-width-bin ECE. Return the mean over classes.
    """
    n, K = probs.shape
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    per_class_ece = np.zeros(K)
    for j in range(K):
        y_bin = (y_true == j).astype(int)
        p_j = probs[:, j]
        binids = np.digitize(p_j, bins) - 1
        binids = np.clip(binids, 0, n_bins - 1)

        ece_j = 0.0
        for m in range(n_bins):
            mask = binids == m
            if not np.any(mask):
                continue
            acc = y_bin[mask].mean()
            conf = p_j[mask].mean()
            ece_j += np.abs(acc - conf) * mask.mean()
        per_class_ece[j] = ece_j

    return float(per_class_ece.mean())


# ---------------------------------------------------------------------------
# Multiclass utility (one-vs-rest binary decision) + PWU
# ---------------------------------------------------------------------------
def multiclass_decision_utility(
    probs: np.ndarray, y_true: np.ndarray, j: int, c: float, threshold: Optional[float] = None
) -> float:
    """One-vs-rest binary decision utility for class j at threshold c.

    Cost of false negatives (y=j, p_j < threshold): (1-c)
    Cost of false positives (y≠j, p_j ≥ threshold): c
    Bayes-optimal threshold is c itself.
    """
    if threshold is None:
        threshold = c

    p_j = probs[:, j]
    y_bin = (y_true == j).astype(int)

    cost = (1 - c) * ((y_bin == 1) & (p_j < threshold)).sum() \
        + c * ((y_bin == 0) & (p_j >= threshold)).sum()
    return -float(cost)


def multiclass_decision_pwu(
    probs: np.ndarray,
    y_true: np.ndarray,
    alpha_param: float,
    beta_param: float,
    n_mc: int = 10000,
) -> float:
    """Monte Carlo estimate of the multiclass decision PWU.

    Samples (j, c) with j uniform over K classes and c ~ Beta(a, b),
    then averages -U_{j,c}(f, y).
    """
    n, K = probs.shape
    rng = np.random.default_rng(42)
    js = rng.integers(low=0, high=K, size=n_mc)
    cs = rng.beta(alpha_param, beta_param, size=n_mc)

    costs = np.empty(n_mc)
    for i in range(n_mc):
        costs[i] = -multiclass_decision_utility(
            probs, y_true, j=int(js[i]), c=float(cs[i])
        )
    return float(costs.mean())


# ---------------------------------------------------------------------------
# Per-model metric computation
# ---------------------------------------------------------------------------
def _compute_metrics_for_one_model(
    dataset: str,
    model: str,
    repeat: int,
    fold: int,
    y_true: np.ndarray,
    probs: np.ndarray,
    alpha_param_c: float,
    beta_param_c: float,
    j_vec: np.ndarray,
    c_vec: np.ndarray,
    c_vec_name: np.ndarray,
) -> Dict[str, float]:
    PWU_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multiclass_classification/results/{dataset}/pwus/{model}"
    )
    PWU_PATH.mkdir(parents=True, exist_ok=True)

    n, K = probs.shape
    y_pred = probs.argmax(axis=1)

    # Clip and renormalize for log_loss stability
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1.0)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

    nll = log_loss(y_true, probs_clipped, labels=list(range(K)))
    brier = brier_multiclass(y_true, probs)
    acc = accuracy_score(y_true, y_pred)
    ece = ece_multiclass(y_true, probs, n_bins=10)

    pwu_fp = PWU_PATH / f"repeat_{repeat}_fold_{fold}_mcd_pwu.npy"
    if pwu_fp.exists():
        mcd_p = float(np.load(pwu_fp))
    else:
        mcd_p = multiclass_decision_pwu(probs, y_true, alpha_param_c, beta_param_c)
        np.save(pwu_fp, mcd_p)

    out = {
        "NLL ↓": float(nll),
        "Brier ↓": float(brier),
        "Acc ↑": float(acc),
        "ECE ↓": float(ece),
        "MCD-PWU ↓": float(mcd_p),
    }

    # Sampled utilities: one per (j, c) pair
    for i, (j, c) in enumerate(zip(j_vec, c_vec)):
        key = f"u_jc, j={int(j)}, c={c_vec_name[i]:.2f} ↑"
        out[key] = float(multiclass_decision_utility(probs, y_true, int(j), float(c)))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(dataset: Optional[str] = "iris"):
    PRED_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multiclass_classification/predictions/{dataset}"
    )

    # configure utilities — same prior Beta(2, 10) on thresholds c as binary
    alpha_param_c = 2.0
    beta_param_c = 10.0
    n_samples = 5

    # Infer K from the first available prediction file so we can sample
    # class indices over the correct range.
    example_files = sorted(PRED_PATH.glob("repeat_*/fold_*/predictions.npz"))
    if len(example_files) == 0:
        raise FileNotFoundError(f"No predictions.npz files found under {PRED_PATH}")
    with np.load(example_files[0]) as npz:
        any_key = list(npz.keys())[0]
        K = int(npz[any_key].shape[1])

    # Fix the (j, c) pairs once (global across repeats) for comparability
    rng = np.random.default_rng(0)
    j_vec = rng.integers(low=0, high=K, size=n_samples)
    c_vec = beta.rvs(alpha_param_c, beta_param_c, size=n_samples, random_state=rng)
    c_vec_name = np.round(c_vec, 2)

    RESULT_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multiclass_classification/results/{dataset}"
    )
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in example_files:
        repeat_str = fp.parents[1].name.replace("repeat_", "")
        fold_str = fp.parents[0].name.replace("fold_", "")
        repeat, fold = int(repeat_str), int(fold_str)

        meta_fp = fp.with_name("meta.csv")
        if not meta_fp.exists():
            raise FileNotFoundError(f"Missing meta.csv next to {fp}")
        meta = pd.read_csv(meta_fp)
        y_true = meta["y_true"].to_numpy().astype(int)

        with np.load(fp) as npz:
            model_names = list(npz.keys())
            for model in model_names:
                probs = npz[model].astype(float)
                if probs.shape[0] != y_true.shape[0]:
                    raise ValueError(
                        f"Row mismatch in {fp}:{model}: probs {probs.shape} vs "
                        f"y_true {y_true.shape}"
                    )
                metrics = _compute_metrics_for_one_model(
                    dataset=dataset,
                    model=model,
                    repeat=repeat,
                    fold=fold,
                    y_true=y_true,
                    probs=probs,
                    alpha_param_c=alpha_param_c,
                    beta_param_c=beta_param_c,
                    j_vec=j_vec,
                    c_vec=c_vec,
                    c_vec_name=c_vec_name,
                )
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model,
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(rows)

    # ---- Aggregate over folds to get per-repeat, per-model metrics ----
    metric_cols = [
        c for c in metrics_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_values_by_repeat = metrics_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols
    ].mean()
    metric_values_by_repeat.to_csv(
        RESULT_PATH / "metric_values_by_repeat.csv", index=False
    )

    # ---- Ranking + alignment per repeat ----
    smaller_is_better = {
        "NLL ↓": True,
        "Brier ↓": True,
        "Acc ↑": False,
        "ECE ↓": True,
        "MCD-PWU ↓": True,
    }
    for i in range(len(c_vec)):
        smaller_is_better[f"u_jc, j={int(j_vec[i])}, c={c_vec_name[i]:.2f} ↑"] = False

    def clean_metric_name(name: str) -> str:
        return name.replace("↑", "").replace("↓", "").strip()

    base_metrics = [
        clean_metric_name(m)
        for m in ["NLL ↓", "Brier ↓", "Acc ↑", "ECE ↓", "MCD-PWU ↓"]
    ]
    utilities = [
        clean_metric_name(f"u_jc, j={int(j_vec[i])}, c={c_vec_name[i]:.2f} ↑")
        for i in range(len(c_vec))
    ]

    ranking_rows = []
    tau_rows = []

    repeats = sorted(metric_values_by_repeat["repeat"].unique().tolist())

    for r in repeats:
        sub = metric_values_by_repeat[metric_values_by_repeat["repeat"] == r].set_index(
            "model"
        )

        rank_dict = {}
        for metric, asc in smaller_is_better.items():
            indices_ordered_models = (
                sub[metric].sort_values(ascending=asc).index.tolist()
            )
            rank_dict[clean_metric_name(metric)] = indices_ordered_models

        row = {"repeat": r}
        for metric, order in rank_dict.items():
            row[metric] = ">".join(order)
        ranking_rows.append(row)

        for m in base_metrics:
            for u in utilities:
                tau = compute_metric_correlation(
                    ranking_m=rank_dict[m], ranking_u=rank_dict[u]
                )
                t1 = top1_agreement(ranking_m=rank_dict[m], ranking_u=rank_dict[u])
                t3 = topk_agreement(
                    ranking_m=rank_dict[m], ranking_u=rank_dict[u], k=3
                )
                tau_rows.append(
                    {
                        "repeat": r,
                        "utility": u,
                        "metric": m,
                        "kendall_tau": tau,
                        "top1": t1,
                        "top3": t3,
                    }
                )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(RESULT_PATH / "model_rankings_by_repeat.csv", index=False)
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(RESULT_PATH / "kendall_by_repeat.csv", index=False)

    # ---- Summary over repeats (mean + quantiles) ----
    def summarize_alignment(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        g = df.groupby(["utility", "metric"])[value_col]
        return g.agg(
            mean="mean",
            median="median",
            q05=lambda x: np.quantile(x, 0.05),
            q95=lambda x: np.quantile(x, 0.95),
        ).reset_index()

    for value_col in ["kendall_tau", "top1", "top3"]:
        summary = summarize_alignment(tau_df, value_col)
        summary.to_csv(
            RESULT_PATH / f"{value_col}_summary_over_repeats.csv", index=False
        )

    # ---- Pearson on raw metric values (sign-flipped for "smaller is better") ----
    pearson_rows = []
    clean_to_orig = {clean_metric_name(m): m for m in smaller_is_better.keys()}

    for r in repeats:
        sub = metric_values_by_repeat[metric_values_by_repeat["repeat"] == r]
        for m in base_metrics:
            m_values = sub[clean_to_orig[m]].to_numpy()
            for u in utilities:
                u_values = sub[clean_to_orig[u]].to_numpy()
                sign = -1.0 if smaller_is_better[clean_to_orig[m]] else 1.0
                corr, pval = pearsonr(sign * m_values, u_values)
                pearson_rows.append(
                    {
                        "repeat": r,
                        "utility": u,
                        "metric": m,
                        "pearson_r": corr,
                        "pearson_pval": pval,
                    }
                )

    pearson_df = pd.DataFrame(pearson_rows)
    pearson_df.to_csv(RESULT_PATH / "pearson_by_repeat.csv", index=False)
    pearson_summary = summarize_alignment(pearson_df, "pearson_r")
    pearson_summary.to_csv(
        RESULT_PATH / "pearson_summary_over_repeats.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="iris", help="Which dataset to use."
    )
    args = parser.parse_args()
    main(**vars(args))
