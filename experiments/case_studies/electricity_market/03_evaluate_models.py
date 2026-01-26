import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.metrics import (   # noqa: E402
    gaussian_nll,
    ece_regression,
    mce_regression,
    rauc_regression,
    error_detection_regression,
)
from src.metrics.utility import (   # noqa: E402
    bidding_utility,
    selective_prediction_pwu,
    topk_utility_regression_pwu,
)
from src.utils import compute_metric_correlation  # noqa: E402

alpha = 0.1
beta = 165  # see https://en.wikipedia.org/wiki/Belwind_Offshore_Wind_Farm


def _compute_metrics_for_one_model(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    prices_da: np.ndarray,
    prices_balancing: np.ndarray,
    alpha_param_lambda: float,
    beta_param_lambda: float,
    alpha_param_k: float,
    beta_param_k: float,
    alpha_param_gamma: float,
    beta_param_gamma: float,
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

    bid_util, profit_cumsum, _ = bidding_utility(
        y_true, mu, std_safe, prices_da, prices_balancing, alpha, beta
    )
    out["Bid-Util ↑"] = float(bid_util)

    return out, profit_cumsum


def main():

    PRED_PATH = Path(str(root_dir) + "/experiments/case_studies/electricity_market/predictions")

    # configure utilities
    alpha_param_lambda = 2
    beta_param_lambda = 10
    alpha_param_k = 1.2
    beta_param_k = 20.8
    alpha_param_gamma = 2
    beta_param_gamma = 6

    RESULT_PATH = Path(str(root_dir) + "/experiments/case_studies/electricity_market/results")
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    files_mu = list(PRED_PATH.glob("repeat_*/fold_*/predictions_mu.csv"))
    files_std = list(PRED_PATH.glob("repeat_*/fold_*/predictions_std.csv"))
    files_prices = list(PRED_PATH.glob("repeat_*/fold_*/prices.csv"))

    # Key by (repeat, fold) extracted from parent directory names
    def _key_from_path(p: Path):
        repeat = int(p.parents[1].name.replace("repeat_", ""))
        fold = int(p.parents[0].name.replace("fold_", ""))
        return repeat, fold

    mu_map = {_key_from_path(p): p for p in files_mu}
    std_map = {_key_from_path(p): p for p in files_std}
    prices_map = {_key_from_path(p): p for p in files_prices}

    common_keys = sorted(
        set(mu_map.keys()) & set(std_map.keys()) & set(prices_map.keys())
    )

    rows = []
    profit_cumsum_dict = {}
    dec_hours = pd.date_range("2024-12-01", "2025-01-01", freq="h", tz="UTC")[:-1]
    for repeat, fold in common_keys:
        fp_mu = mu_map[(repeat, fold)]
        fp_std = std_map[(repeat, fold)]
        fp_prices = prices_map[(repeat, fold)]

        df_mu = pd.read_csv(fp_mu)
        df_std = pd.read_csv(fp_std)
        df_prices = pd.read_csv(fp_prices)

        y_true = df_mu["y_true"].to_numpy().astype(float)
        prices_da = df_prices["Day-Ahead Price"].to_numpy().astype(float)
        prices_balancing = df_prices["Imbalance Price"].to_numpy().astype(float)
        model_cols = [
            c
            for c in df_mu.columns
            if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
        ]

        hours = dec_hours[repeat * 24 : (repeat + 1) * 24]
        for model in model_cols:
            mu = df_mu[model].to_numpy().astype(float)
            std = df_std[model].to_numpy().astype(float)
            metrics, profit_cumsum = _compute_metrics_for_one_model(
                y_true=y_true,
                mu=mu,
                std=std,
                prices_da=prices_da,
                prices_balancing=prices_balancing,
                alpha_param_lambda=alpha_param_lambda,
                beta_param_lambda=beta_param_lambda,
                alpha_param_k=alpha_param_k,
                beta_param_k=beta_param_k,
                alpha_param_gamma=alpha_param_gamma,
                beta_param_gamma=beta_param_gamma,
            )
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics,
                }
            )
            if model not in profit_cumsum_dict:
                profit_cumsum_dict[model] = {}
            base_profit = profit_cumsum_dict[model].get(str(hours[0] - pd.Timedelta(hours=1)), 0.0)
            profit_cumsum_dict[model].update({
                f"{timestamp}": (base_profit + profit_cumsum[i]) for i, timestamp in enumerate(hours)
            })

    metrics_df = pd.DataFrame(rows)
    df_profit_cumsum = pd.DataFrame(profit_cumsum_dict)
    df_profit_cumsum.to_csv(RESULT_PATH / 'profit_cumsum.csv')

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
        "Bid-Util ↑": False,
    }

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
            "TopK-PWU ↓",
        ]
    ]
    utilities = [
        clean_metric_name("Bid-Util ↑"),
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
                tau_rows.append(
                    {"repeat": r, "utility": u, "metric": m, "kendall_tau": tau}
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

    tau_summary = summarize_alignment(tau_df, "kendall_tau")
    tau_summary.to_csv(RESULT_PATH / "kendall_summary_over_repeats.csv", index=False)


if __name__ == "__main__":
    main()
