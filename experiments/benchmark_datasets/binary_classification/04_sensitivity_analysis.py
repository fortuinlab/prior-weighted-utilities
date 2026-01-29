import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,             # high is good
    brier_score_loss,           # low is good
    log_loss,                   # low is good
)

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.metrics import (  # noqa: E402
    ece_binary_classification,
    mce_binary_classification,
    rauc_binary_classification,
    error_detection_binary_classification,
)
from src.metrics.utility import (  # noqa: E402
    binary_decision_pwu,
    binary_decision_utility,
    topk_utility_binary_classification,
    topk_utility_binary_classification_pwu,
)
from src.utils import compute_metric_correlation  # noqa: E402


def _compute_metrics_for_one_model(
    y_true: np.ndarray,
    probs: np.ndarray,
    c_vec_slight: np.ndarray,
    c_vec_slight_name: np.ndarray,
    c_vec_strong: np.ndarray,
    c_vec_strong_name: np.ndarray,
    c_vec_extreme: np.ndarray,
    c_vec_extreme_name: np.ndarray,
    k_frac_vec_slight: np.ndarray,
    k_frac_vec_slight_name: np.ndarray,
    k_frac_vec_strong: np.ndarray,
    k_frac_vec_strong_name: np.ndarray,
    k_frac_vec_extreme: np.ndarray,
    k_frac_vec_extreme_name: np.ndarray,
) -> Dict[str, float]:
    y_pred = (probs > 0.5).astype(int)

    nll = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    acc = accuracy_score(y_true, y_pred)
    ece = ece_binary_classification(y_true, probs, n_bins=10)
    mce = mce_binary_classification(y_true, probs, n_bins=10)

    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1 - eps)
    entropy = -probs_clipped * np.log(probs_clipped) - (1 - probs_clipped) * np.log(
        1 - probs_clipped
    )
    _, _, rauc = rauc_binary_classification(
        y_true, y_pred, uncertainty=entropy, num_points=101
    )
    e_det = error_detection_binary_classification(y_true, y_pred, uncertainty=entropy)

    bd_p = binary_decision_pwu(probs, y_true, 2, 10)
    topk_p = topk_utility_binary_classification_pwu(
        probs, y_true, 1.2, 20.8
    )

    out_slight = {
        "NLL ↓": float(nll),
        "Brier ↓": float(brier),
        "Acc ↑": float(acc),
        "ECE ↓": float(ece),
        "MCE ↓": float(mce),
        "R-AUC ↓": float(rauc),
        "E-Det ↑": float(e_det),
        "BD-PWU ↓": float(bd_p),
        "TopK-PWU ↓": float(topk_p),
    }
    out_strong = out_slight.copy()
    out_extreme = out_slight.copy()

    for i, c in enumerate(c_vec_slight):
        out_slight[f"u_c, c={c_vec_slight_name[i]:.2f} ↑"] = float(
            binary_decision_utility(probs, y_true, c)
        )
    for i, k_frac in enumerate(k_frac_vec_slight):
        k = int(np.ceil(len(y_true) * k_frac))
        out_slight[f"u_k, k/n={k_frac_vec_slight_name[i]:.2f} ↑"] = float(
            topk_utility_binary_classification(probs, y_true, k)
        )

    for i, c in enumerate(c_vec_strong):
        out_strong[f"u_c, c={c_vec_strong_name[i]:.2f} ↑"] = float(
            binary_decision_utility(probs, y_true, c)
        )
    for i, k_frac in enumerate(k_frac_vec_strong):
        k = int(np.ceil(len(y_true) * k_frac))
        out_strong[f"u_k, k/n={k_frac_vec_strong_name[i]:.2f} ↑"] = float(
            topk_utility_binary_classification(probs, y_true, k)
        )

    for i, c in enumerate(c_vec_extreme):
        out_extreme[f"u_c, c={c_vec_extreme_name[i]:.2f} ↑"] = float(
            binary_decision_utility(probs, y_true, c)
        )
    for i, k_frac in enumerate(k_frac_vec_extreme):
        k = int(np.ceil(len(y_true) * k_frac))
        out_extreme[f"u_k, k/n={k_frac_vec_extreme_name[i]:.2f} ↑"] = float(
            topk_utility_binary_classification(probs, y_true, k)
        )

    return out_slight, out_strong, out_extreme


def main(dataset: Optional[str] = "bank"):
    PRED_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/binary_classification/predictions/{dataset}"
    )

    n_c_samples = 5
    n_k_samples = 5
    rng = np.random.default_rng(
        0
    )  # Fix the c_vec once (global across repeats) for comparability

    # utilities for sensitivity analysis
    alpha_param_c_slight = 3.0
    beta_param_c_slight = 9.0
    alpha_param_c_strong = 10.0
    beta_param_c_strong = 10.0
    alpha_param_c_extreme = 10.0
    beta_param_c_extreme = 2.0
    alpha_param_k_slight = 1.5
    beta_param_k_slight = 25.5
    alpha_param_k_strong = 2.0
    beta_param_k_strong = 10.0
    alpha_param_k_extreme = 10.0
    beta_param_k_extreme = 10.0

    # vecs for sensitivity analysis
    c_vec_slight = beta.rvs(
        alpha_param_c_slight, beta_param_c_slight, size=n_c_samples, random_state=rng
    )
    c_vec_slight_name = np.round(c_vec_slight, 2)
    c_vec_strong = beta.rvs(
        alpha_param_c_strong, beta_param_c_strong, size=n_c_samples, random_state=rng
    )
    c_vec_strong_name = np.round(c_vec_strong, 2)
    c_vec_extreme = beta.rvs(
        alpha_param_c_extreme, beta_param_c_extreme, size=n_c_samples, random_state=rng
    )
    c_vec_extreme_name = np.round(c_vec_extreme, 2)
    k_frac_vec_slight = beta.rvs(
        alpha_param_k_slight, beta_param_k_slight, size=n_k_samples, random_state=rng
    )
    k_frac_vec_slight_name = np.round(k_frac_vec_slight, 2)
    k_frac_vec_strong = beta.rvs(
        alpha_param_k_strong, beta_param_k_strong, size=n_k_samples, random_state=rng
    )
    k_frac_vec_strong_name = np.round(k_frac_vec_strong, 2)
    k_frac_vec_extreme = beta.rvs(
        alpha_param_k_extreme, beta_param_k_extreme, size=n_k_samples, random_state=rng
    )
    k_frac_vec_extreme_name = np.round(k_frac_vec_extreme, 2)

    RESULT_PATH_SLIGHT = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/binary_classification/results_slight/{dataset}"
    )
    RESULT_PATH_SLIGHT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_STRONG = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/binary_classification/results_strong/{dataset}"
    )
    RESULT_PATH_STRONG.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_EXTREME = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/binary_classification/results_extreme/{dataset}"
    )
    RESULT_PATH_EXTREME.mkdir(parents=True, exist_ok=True)

    files = sorted(PRED_PATH.glob("repeat_*/fold_*/predictions.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No predictions.csv files found under {PRED_PATH}")

    rows_slight = []
    rows_strong = []
    rows_extreme = []
    for fp in files:
        repeat_str = fp.parents[1].name.replace("repeat_", "")
        fold_str = fp.parents[0].name.replace("fold_", "")
        repeat, fold = int(repeat_str), int(fold_str)

        df = pd.read_csv(fp)
        if "y_true" not in df.columns:
            raise ValueError(f"{fp} is missing y_true column.")

        y_true = df["y_true"].to_numpy().astype(int)
        model_cols = [
            c
            for c in df.columns
            if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
        ]

        for model in model_cols:
            probs = df[model].to_numpy().astype(float)
            metrics_slight, metrics_strong, metrics_extreme = _compute_metrics_for_one_model(
                y_true=y_true,
                probs=probs,
                c_vec_slight=c_vec_slight,
                c_vec_slight_name=c_vec_slight_name,
                c_vec_strong=c_vec_strong,
                c_vec_strong_name=c_vec_strong_name,
                c_vec_extreme=c_vec_extreme,
                c_vec_extreme_name=c_vec_extreme_name,
                k_frac_vec_slight=k_frac_vec_slight,
                k_frac_vec_slight_name=k_frac_vec_slight_name,
                k_frac_vec_strong=k_frac_vec_strong,
                k_frac_vec_strong_name=k_frac_vec_strong_name,
                k_frac_vec_extreme=k_frac_vec_extreme,
                k_frac_vec_extreme_name=k_frac_vec_extreme_name,
            )
            rows_slight.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics_slight,
                }
            )
            rows_strong.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics_strong,
                }
            )
            rows_extreme.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics_extreme,
                }
            )

    metrics_slight_df = pd.DataFrame(rows_slight)
    metrics_strong_df = pd.DataFrame(rows_strong)
    metrics_extreme_df = pd.DataFrame(rows_extreme)

    # ---- Aggregate over folds to get per-repeat, per-model metrics ----
    # Mean over folds within each repeat and model
    metric_cols_slight = [
        c for c in metrics_slight_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_cols_strong = [
        c for c in metrics_strong_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_cols_extreme = [
        c for c in metrics_extreme_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_values_by_repeat_slight = metrics_slight_df.groupby(
        ["repeat", "model"], as_index=False
    )[metric_cols_slight].mean()
    metric_values_by_repeat_strong = metrics_strong_df.groupby(
        ["repeat", "model"], as_index=False
    )[metric_cols_strong].mean()
    metric_values_by_repeat_extreme = metrics_extreme_df.groupby(
        ["repeat", "model"], as_index=False
    )[metric_cols_extreme].mean()

    # Contains per-repeat, per-model metric values; averaged over folds
    metric_values_by_repeat_slight.to_csv(
        RESULT_PATH_SLIGHT / "metric_values_by_repeat.csv", index=False
    )
    metric_values_by_repeat_strong.to_csv(
        RESULT_PATH_STRONG / "metric_values_by_repeat.csv", index=False
    )
    metric_values_by_repeat_extreme.to_csv(
        RESULT_PATH_EXTREME / "metric_values_by_repeat.csv", index=False
    )

    # ---- Ranking + alignment per repeat ----
    # Identify which metrics should be ascending or descending
    smaller_is_better_slight = {
        "NLL ↓": True,
        "Brier ↓": True,
        "Acc ↑": False,
        "ECE ↓": True,
        "MCE ↓": True,
        "R-AUC ↓": True,
        "E-Det ↑": False,
        "BD-PWU ↓": True,
        "TopK-PWU ↓": True,
    }
    smaller_is_better_strong = smaller_is_better_slight.copy()
    smaller_is_better_extreme = smaller_is_better_slight.copy()
    for i in range(len(c_vec_slight)):
        smaller_is_better_slight[f"u_c, c={c_vec_slight_name[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_slight)):
        smaller_is_better_slight[f"u_k, k/n={k_frac_vec_slight_name[i]:.2f} ↑"] = False
    for i in range(len(c_vec_strong)):
        smaller_is_better_strong[f"u_c, c={c_vec_strong_name[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_strong)):
        smaller_is_better_strong[f"u_k, k/n={k_frac_vec_strong_name[i]:.2f} ↑"] = False
    for i in range(len(c_vec_extreme)):
        smaller_is_better_extreme[f"u_c, c={c_vec_extreme_name[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_extreme)):
        smaller_is_better_extreme[f"u_k, k/n={k_frac_vec_extreme_name[i]:.2f} ↑"] = False

    # Clean names (remove arrows) for output tables
    def clean_metric_name(name: str) -> str:
        return name.replace("↑", "").replace("↓", "").strip()

    base_metrics = [
        clean_metric_name(m)
        for m in [
            "NLL ↓",
            "Brier ↓",
            "Acc ↑",
            "ECE ↓",
            "MCE ↓",
            "R-AUC ↓",
            "E-Det ↑",
            "BD-PWU ↓",
            "TopK-PWU ↓",
        ]
    ]
    utilities_slight = [clean_metric_name(f"u_c, c={c:.2f} ↑") for c in c_vec_slight_name] + [
        clean_metric_name(f"u_k, k/n={k_frac:.2f} ↑") for k_frac in k_frac_vec_slight_name
    ]
    utilities_strong = [clean_metric_name(f"u_c, c={c:.2f} ↑") for c in c_vec_strong_name] + [
        clean_metric_name(f"u_k, k/n={k_frac:.2f} ↑") for k_frac in k_frac_vec_strong_name
    ]
    utilities_extreme = [clean_metric_name(f"u_c, c={c:.2f} ↑") for c in c_vec_extreme_name] + [
        clean_metric_name(f"u_k, k/n={k_frac:.2f} ↑") for k_frac in k_frac_vec_extreme_name
    ]

    # Store rankings per repeat (as strings for easy inspection)
    ranking_rows_slight = []
    ranking_rows_strong = []
    ranking_rows_extreme = []

    # Store kendall per repeat (utility x metric)
    tau_rows_slight = []
    tau_rows_strong = []
    tau_rows_extreme = []

    repeats = sorted(metric_values_by_repeat_slight["repeat"].unique().tolist())

    for r in repeats:
        sub_slight = metric_values_by_repeat_slight[metric_values_by_repeat_slight["repeat"] == r].set_index(
            "model"
        )
        sub_strong = metric_values_by_repeat_strong[metric_values_by_repeat_strong["repeat"] == r].set_index(
            "model"
        )
        sub_extreme = metric_values_by_repeat_extreme[metric_values_by_repeat_extreme["repeat"] == r].set_index(
            "model"
        )

        # Build rank dict
        rank_dict_slight = {}
        rank_dict_strong = {}
        rank_dict_extreme = {}

        for metric, asc in smaller_is_better_slight.items():
            indices_ordered_models = (
                sub_slight[metric].sort_values(ascending=asc).index.tolist()
            )  # lowest index is best model
            rank_dict_slight[clean_metric_name(metric)] = indices_ordered_models
        for metric, asc in smaller_is_better_strong.items():
            indices_ordered_models = (
                sub_strong[metric].sort_values(ascending=asc).index.tolist()
            )  # lowest index is best model
            rank_dict_strong[clean_metric_name(metric)] = indices_ordered_models
        for metric, asc in smaller_is_better_extreme.items():
            indices_ordered_models = (
                sub_extreme[metric].sort_values(ascending=asc).index.tolist()
            )  # lowest index is best model
            rank_dict_extreme[clean_metric_name(metric)] = indices_ordered_models

        # Save ranking list as a single string "A>B>C>D>E" for CSV friendliness
        row_slight = {"repeat": r}
        row_strong = {"repeat": r}
        row_extreme = {"repeat": r}
        for metric, order in rank_dict_slight.items():
            row_slight[metric] = ">".join(order)
        ranking_rows_slight.append(row_slight)
        for metric, order in rank_dict_strong.items():
            row_strong[metric] = ">".join(order)
        ranking_rows_strong.append(row_strong)
        for metric, order in rank_dict_extreme.items():
            row_extreme[metric] = ">".join(order)
        ranking_rows_extreme.append(row_extreme)

        for m in base_metrics:
            for u in utilities_slight:
                tau = compute_metric_correlation(
                    ranking_m=rank_dict_slight[m], ranking_u=rank_dict_slight[u]
                )
                tau_rows_slight.append(
                    {"repeat": r, "utility": u, "metric": m, "kendall_tau": tau}
                )
            for u in utilities_strong:
                tau = compute_metric_correlation(
                    ranking_m=rank_dict_strong[m], ranking_u=rank_dict_strong[u]
                )
                tau_rows_strong.append(
                    {"repeat": r, "utility": u, "metric": m, "kendall_tau": tau}
                )
            for u in utilities_extreme:
                tau = compute_metric_correlation(
                    ranking_m=rank_dict_extreme[m], ranking_u=rank_dict_extreme[u]
                )
                tau_rows_extreme.append(
                    {"repeat": r, "utility": u, "metric": m, "kendall_tau": tau}
                )

    tau_df_slight = pd.DataFrame(tau_rows_slight)
    tau_df_slight.to_csv(RESULT_PATH_SLIGHT / "kendall_by_repeat.csv", index=False)
    tau_df_strong = pd.DataFrame(tau_rows_strong)
    tau_df_strong.to_csv(RESULT_PATH_STRONG / "kendall_by_repeat.csv", index=False)
    tau_df_extreme = pd.DataFrame(tau_rows_extreme)
    tau_df_extreme.to_csv(RESULT_PATH_EXTREME / "kendall_by_repeat.csv", index=False)

    # ---- Summary over repeats (mean + quantiles) ----
    def summarize_alignment(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        g = df.groupby(["utility", "metric"])[value_col]
        out = g.agg(
            mean="mean",
            median="median",
            q05=lambda x: np.quantile(x, 0.05),
            q95=lambda x: np.quantile(x, 0.95),
        ).reset_index()
        return out

    tau_summary_slight = summarize_alignment(tau_df_slight, "kendall_tau")
    tau_summary_slight.to_csv(RESULT_PATH_SLIGHT / "kendall_summary_over_repeats.csv", index=False)
    tau_summary_strong = summarize_alignment(tau_df_strong, "kendall_tau")
    tau_summary_strong.to_csv(RESULT_PATH_STRONG / "kendall_summary_over_repeats.csv", index=False)
    tau_summary_extreme = summarize_alignment(tau_df_extreme, "kendall_tau")
    tau_summary_extreme.to_csv(RESULT_PATH_EXTREME / "kendall_summary_over_repeats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="bank", help="Which dataset to use."
    )

    args = parser.parse_args()
    main(**vars(args))
