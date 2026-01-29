import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta

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
from src.utils import compute_metric_correlation  # noqa: E402


def _compute_metrics_for_one_model(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    lambda_factor_vec_slight: np.ndarray,
    lambda_factor_vec_name_slight: np.ndarray,
    k_frac_vec_slight: np.ndarray,
    k_frac_name_slight: np.ndarray,
    gamma_factor_vec_slight: np.ndarray,
    gamma_factor_vec_name_slight: np.ndarray,
    lambda_factor_vec_strong: np.ndarray,
    lambda_factor_vec_name_strong: np.ndarray,
    k_frac_vec_strong: np.ndarray,
    k_frac_name_strong: np.ndarray,
    gamma_factor_vec_strong: np.ndarray,
    gamma_factor_vec_name_strong: np.ndarray,
    lambda_factor_vec_extreme: np.ndarray,
    lambda_factor_vec_name_extreme: np.ndarray,
    k_frac_vec_extreme: np.ndarray,
    k_frac_name_extreme: np.ndarray,
    gamma_factor_vec_extreme: np.ndarray,
    gamma_factor_vec_name_extreme: np.ndarray,
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
        2.0,
        10.0,
    )
    topk_p = topk_utility_regression_pwu(
        mu,
        std_safe,
        y_true,
        1.2,
        20.8,
        2.0,
        6.0,
    )

    out_slight = {
        "NLL ↓": float(nll),
        "MSE ↓": float(mse),
        "ECE ↓": float(ece),
        "MCE ↓": float(mce),
        "R-AUC ↓": float(rauc),
        "E-Det ↑": float(e_det),
        "SP-PWU ↓": float(sp_p),
        "TopK-PWU ↓": float(topk_p),
    }
    out_strong = out_slight.copy()
    out_extreme = out_slight.copy()

    # data variance
    y_var = ((y_true - np.mean(y_true)) ** 2).mean()
    lambda_vec_slight = lambda_factor_vec_slight * y_var
    gamma_vec_slight = gamma_factor_vec_slight * y_var
    lambda_vec_strong = lambda_factor_vec_strong * y_var
    gamma_vec_strong = gamma_factor_vec_strong * y_var
    lambda_vec_extreme = lambda_factor_vec_extreme * y_var
    gamma_vec_extreme = gamma_factor_vec_extreme * y_var

    for i in range(len(lambda_vec_slight)):
        out_slight[f"u_lambda, lambda={lambda_factor_vec_name_slight[i]:.2f} ↑"] = float(
            selective_prediction_utility(mu, std_safe, y_true, lambda_vec_slight[i])[0]
        )
    for i in range(len(k_frac_vec_slight)):
        k = int(np.ceil(len(y_true) * k_frac_vec_slight[i]))
        out_slight[
            f"u_k_gamma, k/n={k_frac_name_slight[i]:.2f}, gamma={gamma_factor_vec_name_slight[i]:.2f} ↑"
        ] = float(topk_utility_regression(mu, std_safe, y_true, k, gamma_vec_slight[i]))
    for i in range(len(lambda_vec_strong)):
        out_strong[f"u_lambda, lambda={lambda_factor_vec_name_strong[i]:.2f} ↑"] = float(
            selective_prediction_utility(mu, std_safe, y_true, lambda_vec_strong[i])[0]
        )
    for i in range(len(k_frac_vec_strong)):
        k = int(np.ceil(len(y_true) * k_frac_vec_strong[i]))
        out_strong[
            f"u_k_gamma, k/n={k_frac_name_strong[i]:.2f}, gamma={gamma_factor_vec_name_strong[i]:.2f} ↑"
        ] = float(topk_utility_regression(mu, std_safe, y_true, k, gamma_vec_strong[i]))
    for i in range(len(lambda_vec_extreme)):
        out_extreme[f"u_lambda, lambda={lambda_factor_vec_name_extreme[i]:.2f} ↑"] = float(
            selective_prediction_utility(mu, std_safe, y_true, lambda_vec_extreme[i])[0]
        )
    for i in range(len(k_frac_vec_extreme)):
        k = int(np.ceil(len(y_true) * k_frac_vec_extreme[i]))
        out_extreme[
            f"u_k_gamma, k/n={k_frac_name_extreme[i]:.2f}, gamma={gamma_factor_vec_name_extreme[i]:.2f} ↑"
        ] = float(topk_utility_regression(mu, std_safe, y_true, k, gamma_vec_extreme[i]))

    return out_slight, out_strong, out_extreme


def main(dataset: Optional[str] = "air"):
    PRED_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/predictions/{dataset}"
    )

    n_lambda_samples = 5
    n_k_gamma_samples = 5
    rng = np.random.default_rng(
        0
    )  # Fix the c_vec once (global across repeats) for comparability

    alpha_param_lambda_slight = 3
    beta_param_lambda_slight = 9
    alpha_param_k_slight = 1.5
    beta_param_k_slight = 25.5
    alpha_param_gamma_slight = 2
    beta_param_gamma_slight = 10
    alpha_param_lambda_strong = 10
    beta_param_lambda_strong = 10
    alpha_param_k_strong = 2
    beta_param_k_strong = 10
    alpha_param_gamma_strong = 5
    beta_param_gamma_strong = 7
    alpha_param_lambda_extreme = 10
    beta_param_lambda_extreme = 2
    alpha_param_k_extreme = 10
    beta_param_k_extreme = 10
    alpha_param_gamma_extreme = 9
    beta_param_gamma_extreme = 3

    lambda_factor_vec_slight = beta.rvs(
        alpha_param_lambda_slight, beta_param_lambda_slight, size=n_lambda_samples, random_state=rng
    )
    lambda_factor_vec_name_slight = np.round(lambda_factor_vec_slight, 2)
    k_frac_vec_slight = beta.rvs(
        alpha_param_k_slight, beta_param_k_slight, size=n_k_gamma_samples, random_state=rng
    )
    k_frac_name_slight = np.round(k_frac_vec_slight, 2)
    gamma_factor_vec_slight = beta.rvs(
        alpha_param_gamma_slight, beta_param_gamma_slight, size=n_k_gamma_samples, random_state=rng
    )
    gamma_factor_vec_name_slight = np.round(gamma_factor_vec_slight, 2)
    lambda_factor_vec_strong = beta.rvs(
        alpha_param_lambda_strong, beta_param_lambda_strong, size=n_lambda_samples, random_state=rng
    )
    lambda_factor_vec_name_strong = np.round(lambda_factor_vec_strong, 2)
    k_frac_vec_strong = beta.rvs(
        alpha_param_k_strong, beta_param_k_strong, size=n_k_gamma_samples, random_state=rng
    )
    k_frac_name_strong = np.round(k_frac_vec_strong, 2)
    gamma_factor_vec_strong = beta.rvs(
        alpha_param_gamma_strong, beta_param_gamma_strong, size=n_k_gamma_samples, random_state=rng
    )
    gamma_factor_vec_name_strong = np.round(gamma_factor_vec_strong, 2)
    lambda_factor_vec_extreme = beta.rvs(
        alpha_param_lambda_extreme, beta_param_lambda_extreme, size=n_lambda_samples, random_state=rng
    )
    lambda_factor_vec_name_extreme = np.round(lambda_factor_vec_extreme, 2)
    k_frac_vec_extreme = beta.rvs(
        alpha_param_k_extreme, beta_param_k_extreme, size=n_k_gamma_samples, random_state=rng
    )
    k_frac_name_extreme = np.round(k_frac_vec_extreme, 2)
    gamma_factor_vec_extreme = beta.rvs(
        alpha_param_gamma_extreme, beta_param_gamma_extreme, size=n_k_gamma_samples, random_state=rng
    )
    gamma_factor_vec_name_extreme = np.round(gamma_factor_vec_extreme, 2)

    RESULT_PATH_SLIGHT = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/results_slight/{dataset}"
    )
    RESULT_PATH_SLIGHT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_STRONG = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/results_strong/{dataset}"
    )
    RESULT_PATH_STRONG.mkdir(parents=True, exist_ok=True)
    RESULT_PATH_EXTREME = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/results_extreme/{dataset}"
    )
    RESULT_PATH_EXTREME.mkdir(parents=True, exist_ok=True)

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

    rows_slight = []
    rows_strong = []
    rows_extreme = []
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
            metrics_slight, metrics_strong, metrics_extreme = _compute_metrics_for_one_model(
                y_true=y_true,
                mu=mu,
                std=std,
                lambda_factor_vec_slight=lambda_factor_vec_slight,
                lambda_factor_vec_name_slight=lambda_factor_vec_name_slight,
                k_frac_vec_slight=k_frac_vec_slight,
                k_frac_name_slight=k_frac_name_slight,
                gamma_factor_vec_slight=gamma_factor_vec_slight,
                gamma_factor_vec_name_slight=gamma_factor_vec_name_slight,
                lambda_factor_vec_strong=lambda_factor_vec_strong,
                lambda_factor_vec_name_strong=lambda_factor_vec_name_strong,
                k_frac_vec_strong=k_frac_vec_strong,
                k_frac_name_strong=k_frac_name_strong,
                gamma_factor_vec_strong=gamma_factor_vec_strong,
                gamma_factor_vec_name_strong=gamma_factor_vec_name_strong,
                lambda_factor_vec_extreme=lambda_factor_vec_extreme,
                lambda_factor_vec_name_extreme=lambda_factor_vec_name_extreme,
                k_frac_vec_extreme=k_frac_vec_extreme,
                k_frac_name_extreme=k_frac_name_extreme,
                gamma_factor_vec_extreme=gamma_factor_vec_extreme,
                gamma_factor_vec_name_extreme=gamma_factor_vec_name_extreme,
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

    metric_cols_slight = [
        c for c in metrics_slight_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_cols_strong = [
        c for c in metrics_strong_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_cols_extreme = [
        c for c in metrics_extreme_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_values_by_repeat_slight = metrics_slight_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols_slight
    ].mean()
    metric_values_by_repeat_strong = metrics_strong_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols_strong
    ].mean()
    metric_values_by_repeat_extreme = metrics_extreme_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols_extreme
    ].mean()

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
        "MSE ↓": True,
        "ECE ↓": True,
        "MCE ↓": True,
        "R-AUC ↓": True,
        "E-Det ↑": False,
        "SP-PWU ↓": True,
        "TopK-PWU ↓": True,
    }
    smaller_is_better_strong = smaller_is_better_slight.copy()
    smaller_is_better_extreme = smaller_is_better_slight.copy()
    for i in range(len(lambda_factor_vec_slight)):
        smaller_is_better_slight[f"u_lambda, lambda={lambda_factor_vec_name_slight[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_slight)):
        smaller_is_better_slight[
            f"u_k_gamma, k/n={k_frac_name_slight[i]:.2f}, gamma={gamma_factor_vec_name_slight[i]:.2f} ↑"
        ] = False
    for i in range(len(lambda_factor_vec_strong)):
        smaller_is_better_strong[f"u_lambda, lambda={lambda_factor_vec_name_strong[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_strong)):
        smaller_is_better_strong[
            f"u_k_gamma, k/n={k_frac_name_strong[i]:.2f}, gamma={gamma_factor_vec_name_strong[i]:.2f} ↑"
        ] = False
    for i in range(len(lambda_factor_vec_extreme)):
        smaller_is_better_extreme[f"u_lambda, lambda={lambda_factor_vec_name_extreme[i]:.2f} ↑"] = False
    for i in range(len(k_frac_vec_extreme)):
        smaller_is_better_extreme[
            f"u_k_gamma, k/n={k_frac_name_extreme[i]:.2f}, gamma={gamma_factor_vec_name_extreme[i]:.2f} ↑"
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
    utilities_slight = [
        clean_metric_name(f"u_lambda, lambda={c:.2f} ↑") for c in lambda_factor_vec_name_slight
    ] + [
        clean_metric_name(
            f"u_k_gamma, k/n={k_frac:.2f}, gamma={gamma_factor_vec_name_slight[i]:.2f} ↑"
        )
        for i, k_frac in enumerate(k_frac_name_slight)
    ]
    utilities_strong = [
        clean_metric_name(f"u_lambda, lambda={c:.2f} ↑") for c in lambda_factor_vec_name_strong
    ] + [
        clean_metric_name(
            f"u_k_gamma, k/n={k_frac:.2f}, gamma={gamma_factor_vec_name_strong[i]:.2f} ↑"
        )
        for i, k_frac in enumerate(k_frac_name_strong)
    ]
    utilities_extreme = [
        clean_metric_name(f"u_lambda, lambda={c:.2f} ↑") for c in lambda_factor_vec_name_extreme
    ] + [
        clean_metric_name(
            f"u_k_gamma, k/n={k_frac:.2f}, gamma={gamma_factor_vec_name_extreme[i]:.2f} ↑"
        )
        for i, k_frac in enumerate(k_frac_name_extreme)
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
    parser = argparse.ArgumentParser(description="Regression evaluation.")
    parser.add_argument(
        "--dataset", type=str, default="air", help="Which dataset to use."
    )

    args = parser.parse_args()
    main(**vars(args))
