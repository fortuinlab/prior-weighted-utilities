import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta, pearsonr

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.metrics import (   # noqa: E402
    ece_regression,
    gaussian_nll,
    mce_regression,
    rauc_regression,
    error_detection_regression
)

from src.metrics.utility import (   # noqa: E402
    selective_prediction_utility,
    selective_prediction_pwu,
    topk_utility_regression,
    topk_utility_regression_pwu
)
from src.utils import compute_metric_correlation, top1_agreement, topk_agreement  # noqa: E402


def _compute_metrics_for_one_model(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    alpha_param_lambda: float,
    beta_param_lambda: float,
    alpha_param_k: float,
    beta_param_k: float,
    alpha_param_gamma: float,
    beta_param_gamma: float,
    lambda_factor_vec: np.ndarray,
    lambda_factor_vec_name: np.ndarray,
    k_frac_vec: np.ndarray,
    k_frac_name: np.ndarray,
    gamma_factor_vec: np.ndarray,
    gamma_factor_vec_name: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-12
    std_safe = np.clip(std, eps, None)

    mse = ((y_true - mu) ** 2).mean()
    nll = gaussian_nll(y_true, mu, std_safe)
    ece, _, _ = ece_regression(y_true, mu, std_safe)
    mce, _, _ = mce_regression(y_true, mu, std_safe)

    entropy = 0.5 * np.log(2 * np.pi * np.e * std_safe**2)
    _, _, rauc = rauc_regression(y_true, mu, uncertainty=entropy, num_points=101)
    e_det = error_detection_regression(y_true, mu, uncertainty=entropy, threshold=0.1)

    sp_p = selective_prediction_pwu(
        mu,
        std_safe,
        y_true,
        alpha_param_lambda,
        beta_param_lambda,
    )
    topk_p = topk_utility_regression_pwu(
        mu,
        std_safe,
        y_true,
        alpha_param_k,
        beta_param_k,
        alpha_param_gamma,
        beta_param_gamma,
    )

    out = {
        "NLL ↓": float(nll),
        "MSE ↓": float(mse),
        "ECE ↓": float(ece),
        "MCE ↓": float(mce),
        "R-AUC ↓": float(rauc),
        "E-Det ↑": float(e_det),
        "SP-PWU ↓": float(sp_p),
        "TopK-PWU ↓": float(topk_p),
    }

    # data variance
    y_var = ((y_true - np.mean(y_true)) ** 2).mean()
    lambda_vec = lambda_factor_vec * y_var
    gamma_vec = gamma_factor_vec * y_var

    for i in range(len(lambda_vec)):
        out[f"u_lambda, lambda={lambda_factor_vec_name[i]:.2f} ↑"] = float(
            selective_prediction_utility(mu, std_safe, y_true, lambda_vec[i])[0]
        )

    for i in range(len(k_frac_vec)):
        k = int(np.ceil(len(y_true) * k_frac_vec[i]))
        out[
            f"u_k_gamma, k/n={k_frac_name[i]:.2f}, gamma={gamma_factor_vec_name[i]:.2f} ↑"
        ] = float(topk_utility_regression(mu, std_safe, y_true, k, gamma_vec[i]))

    return out


def main(dataset: Optional[str] = "air"):
    PRED_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/predictions/{dataset}"
    )

    # configure utilities
    alpha_param_lambda = 2
    beta_param_lambda = 10
    n_lambda_samples = 5
    alpha_param_k = 1.2
    beta_param_k = 20.8
    alpha_param_gamma = 2
    beta_param_gamma = 6
    n_k_gamma_samples = 5
    rng = np.random.default_rng(
        0
    )  # Fix the c_vec once (global across repeats) for comparability
    lambda_factor_vec = beta.rvs(
        alpha_param_lambda, beta_param_lambda, size=n_lambda_samples, random_state=rng
    )
    lambda_factor_vec_name = np.round(lambda_factor_vec, 2)
    k_frac_vec = beta.rvs(
        alpha_param_k, beta_param_k, size=n_k_gamma_samples, random_state=rng
    )
    k_frac_name = np.round(k_frac_vec, 2)
    gamma_factor_vec = beta.rvs(
        alpha_param_gamma, beta_param_gamma, size=n_k_gamma_samples, random_state=rng
    )
    gamma_factor_vec_name = np.round(gamma_factor_vec, 2)

    RESULT_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/results/{dataset}"
    )
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    files_mu = list(PRED_PATH.glob("repeat_*/fold_*/predictions_mu.csv"))
    files_std = list(PRED_PATH.glob("repeat_*/fold_*/predictions_std.csv"))

    # Key by (repeat, fold) extracted from parent directory names
    def _key_from_path(p: Path):
        repeat = int(p.parents[1].name.replace("repeat_", ""))
        fold = int(p.parents[0].name.replace("fold_", ""))
        return repeat, fold

    mu_map = {_key_from_path(p): p for p in files_mu}
    std_map = {_key_from_path(p): p for p in files_std}

    common_keys = sorted(set(mu_map.keys()) & set(std_map.keys()))
    missing_mu = sorted(set(std_map.keys()) - set(mu_map.keys()))
    missing_std = sorted(set(mu_map.keys()) - set(std_map.keys()))

    if missing_mu:
        print(
            f"WARNING: missing predictions_mu.csv for {len(missing_mu)} folds. Examples: {missing_mu[:5]}"
        )
    if missing_std:
        print(
            f"WARNING: missing predictions_std.csv for {len(missing_std)} folds. Examples: {missing_std[:5]}"
        )

    rows = []
    for repeat, fold in common_keys:
        fp_mu = mu_map[(repeat, fold)]
        fp_std = std_map[(repeat, fold)]

        df_mu = pd.read_csv(fp_mu)
        df_std = pd.read_csv(fp_std)

        y_true = df_mu["y_true"].to_numpy().astype(float)
        model_cols = [
            c
            for c in df_mu.columns
            if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
        ]

        for model in model_cols:
            mu = df_mu[model].to_numpy().astype(float)
            std = df_std[model].to_numpy().astype(float)
            metrics = _compute_metrics_for_one_model(
                y_true=y_true,
                mu=mu,
                std=std,
                alpha_param_lambda=alpha_param_lambda,
                beta_param_lambda=beta_param_lambda,
                alpha_param_k=alpha_param_k,
                beta_param_k=beta_param_k,
                alpha_param_gamma=alpha_param_gamma,
                beta_param_gamma=beta_param_gamma,
                lambda_factor_vec=lambda_factor_vec,
                lambda_factor_vec_name=lambda_factor_vec_name,
                k_frac_vec=k_frac_vec,
                k_frac_name=k_frac_name,
                gamma_factor_vec=gamma_factor_vec,
                gamma_factor_vec_name=gamma_factor_vec_name,
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

    if metrics_df.empty:
        print(f"ERROR: No prediction files found in {PRED_PATH}")
        print(
            "Expected structure: repeat_*/fold_*/predictions_mu.csv and predictions_std.csv"
        )
        sys.exit(1)

    # ---- Aggregate over folds to get per-repeat, per-model metrics ----
    # Mean over folds within each repeat and model
    metric_cols = [
        c for c in metrics_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_values_by_repeat = metrics_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols
    ].mean()

    # Contains per-repeat, per-model metric values; averaged over folds
    metric_values_by_repeat.to_csv(
        RESULT_PATH / "metric_values_by_repeat.csv", index=False
    )

    # ---- Ranking + alignment per repeat ----
    # Identify which metrics should be ascending or descending
    smaller_is_better = {
        "NLL ↓": True,
        "MSE ↓": True,
        "ECE ↓": True,
        "MCE ↓": True,
        "R-AUC ↓": True,
        "E-Det ↑": False,
        "SP-PWU ↓": True,
        "TopK-PWU ↓": True,
    }
    for i in range(len(lambda_factor_vec)):
        smaller_is_better[f"u_lambda, lambda={lambda_factor_vec_name[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec)):
        smaller_is_better[
            f"u_k_gamma, k/n={k_frac_name[i]:.2f}, gamma={gamma_factor_vec_name[i]:.2f} ↑"
        ] = False

    # Clean names (remove arrows) for output tables
    def clean_metric_name(name: str) -> str:
        return name.replace("↑", "").replace("↓", "").strip()

    base_metrics = [
        clean_metric_name(m)
        for m in [
            "NLL ↓",
            "MSE ↓",
            "ECE ↓",
            "MCE ↓",
            "R-AUC ↓",
            "E-Det ↑",
            "SP-PWU ↓",
            "TopK-PWU ↓"]
    ]
    utilities = [
        clean_metric_name(f"u_lambda, lambda={c:.2f} ↑") for c in lambda_factor_vec_name
    ] + [
        clean_metric_name(
            f"u_k_gamma, k/n={k_frac:.2f}, gamma={gamma_factor_vec_name[i]:.2f} ↑"
        )
        for i, k_frac in enumerate(k_frac_name)
    ]

    # Store rankings per repeat (as strings for easy inspection)
    ranking_rows = []

    # Store kendall per repeat (utility x metric)
    tau_rows = []

    repeats = sorted(metric_values_by_repeat["repeat"].unique().tolist())

    for r in repeats:
        sub = metric_values_by_repeat[metric_values_by_repeat["repeat"] == r].set_index(
            "model"
        )

        # Build rank dict
        rank_dict = {}

        for metric, asc in smaller_is_better.items():
            indices_ordered_models = (
                sub[metric].sort_values(ascending=asc).index.tolist()
            )  # lowest index is best model
            rank_dict[clean_metric_name(metric)] = indices_ordered_models

        # Save ranking list as a single string "A>B>C>D>E" for CSV friendliness
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
                t3 = topk_agreement(ranking_m=rank_dict[m], ranking_u=rank_dict[u], k=3)
                tau_rows.append(
                    {"repeat": r, "utility": u, "metric": m, "kendall_tau": tau, "top1": t1, "top3": t3}
                )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(RESULT_PATH / "model_rankings_by_repeat.csv", index=False)
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(RESULT_PATH / "kendall_by_repeat.csv", index=False)

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

    for value_col in ["kendall_tau", "top1", "top3"]:
        summary = summarize_alignment(tau_df, value_col)
        summary.to_csv(RESULT_PATH / f"{value_col}_summary_over_repeats.csv", index=False)

    # ---- Pearson on raw metric/utility values ----
    pearson_rows = []

    clean_to_orig = {clean_metric_name(m): m for m in smaller_is_better.keys()}

    for r in repeats:
        sub = metric_values_by_repeat[metric_values_by_repeat["repeat"] == r]

        for m in base_metrics:
            sign = -1.0 if smaller_is_better[clean_to_orig[m]] else 1.0
            m_values = sign * sub[clean_to_orig[m]].to_numpy()
            for u in utilities:
                u_values = sub[clean_to_orig[u]].to_numpy()
                corr, pval = pearsonr(m_values, u_values)
                pearson_rows.append({
                    "repeat": r,
                    "utility": u,
                    "metric": m,
                    "pearson_r": corr,
                    "pearson_pval": pval,
                })

    pearson_df = pd.DataFrame(pearson_rows)
    pearson_df.to_csv(RESULT_PATH / "pearson_by_repeat.csv", index=False)

    pearson_summary = summarize_alignment(pearson_df, "pearson_r")
    pearson_summary.to_csv(RESULT_PATH / "pearson_summary_over_repeats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression evaluation.")
    parser.add_argument(
        "--dataset", type=str, default="air", help="Which dataset to use."
    )

    args = parser.parse_args()
    main(**vars(args))
