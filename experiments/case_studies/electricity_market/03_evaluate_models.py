import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

# Block bootstrap configuration
BOOTSTRAP_SEED = 0
N_BOOTSTRAP = 100  # matches the 100-repeat setup in the benchmark experiments


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
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
) -> Tuple[Dict[str, float], np.ndarray]:
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


# ---------------------------------------------------------------------------
# Ranking / alignment utilities
# ---------------------------------------------------------------------------
SMALLER_IS_BETTER = {
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

BASE_METRICS_RAW = [
    "NLL ↓", "MSE ↓", "ECE ↓", "MCE ↓",
    "R-AUC ↓", "E-Det ↑", "SP-PWU ↓", "TopK-PWU ↓",
]
UTILITIES_RAW = ["Bid-Util ↑"]


def clean_metric_name(name: str) -> str:
    return name.replace("↑", "").replace("↓", "").strip()


def _rank_and_tau(
    per_model_metrics: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """Given {model: {metric: value}}, return rankings and Kendall τ rows.

    Rankings are lists of models from best to worst (lowest index = best).
    """
    models = sorted(per_model_metrics.keys())
    # Build a model x metric table
    table = {m: {mt: per_model_metrics[m][mt] for mt in SMALLER_IS_BETTER}
             for m in models}

    rank_dict: Dict[str, List[str]] = {}
    for metric, asc in SMALLER_IS_BETTER.items():
        ordered = sorted(models, key=lambda mdl: table[mdl][metric],
                         reverse=not asc)
        rank_dict[clean_metric_name(metric)] = ordered

    tau_rows = []
    for m_raw in BASE_METRICS_RAW:
        m = clean_metric_name(m_raw)
        for u_raw in UTILITIES_RAW:
            u = clean_metric_name(u_raw)
            tau = compute_metric_correlation(
                ranking_m=rank_dict[m], ranking_u=rank_dict[u],
            )
            tau_rows.append({"utility": u, "metric": m, "kendall_tau": tau})
    return rank_dict, tau_rows


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def _sample_day_indices(
    n_days: int, n_sample_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample day indices via moving-block bootstrap.

    We sample ceil(n_sample_days / block_size_days) block starting positions
    (with replacement) from {0, ..., n_days - block_size_days}, each
    contributing block_size_days consecutive day indices.

    Block starts wrap around when a block would run past n_days, using
    modular indexing so every day has equal probability of being sampled.
    """
    n_blocks = n_sample_days
    starts = rng.integers(low=0, high=n_days, size=n_blocks)
    idx = np.concatenate([
        (np.arange(s, s + 1) % n_days) for s in starts
    ])
    return idx[:n_sample_days]


def _pool_predictions(
    common_keys: List[Tuple[int, int]],
    mu_map, std_map, prices_map,
    hours_per_day: int,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Concatenate all predictions across days, preserving chronological order.

    Returns dicts keyed by model for mu/std (shape: n_days x 24), plus
    y_true / prices arrays (same shape), and the list of model column names.
    """
    sorted_keys = sorted(common_keys)  # chronological by repeat
    n_days = len(sorted_keys)

    # Peek at first file to get model columns
    first_mu = pd.read_csv(mu_map[sorted_keys[0]])
    model_cols = [
        c for c in first_mu.columns
        if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
    ]

    mu_arr = {m: np.empty((n_days, hours_per_day)) for m in model_cols}
    std_arr = {m: np.empty((n_days, hours_per_day)) for m in model_cols}
    y_true_arr = np.empty((n_days, hours_per_day))
    prices_da_arr = np.empty((n_days, hours_per_day))
    prices_bal_arr = np.empty((n_days, hours_per_day))

    for di, key in enumerate(sorted_keys):
        df_mu = pd.read_csv(mu_map[key])
        df_std = pd.read_csv(std_map[key])
        df_prices = pd.read_csv(prices_map[key])

        y_true_arr[di] = df_mu["y_true"].to_numpy().astype(float)
        prices_da_arr[di] = df_prices["Day-Ahead Price"].to_numpy().astype(float)
        prices_bal_arr[di] = df_prices["Imbalance Price"].to_numpy().astype(float)
        for m in model_cols:
            mu_arr[m][di] = df_mu[m].to_numpy().astype(float)
            std_arr[m][di] = df_std[m].to_numpy().astype(float)

    return (mu_arr, std_arr,
            y_true_arr, prices_da_arr, prices_bal_arr,
            model_cols)


def _compute_metrics_on_day_subset(
    day_idx: np.ndarray,
    mu_arr: Dict[str, np.ndarray],
    std_arr: Dict[str, np.ndarray],
    y_true_arr: np.ndarray,
    prices_da_arr: np.ndarray,
    prices_bal_arr: np.ndarray,
    model_cols: List[str],
    util_params: Dict,
) -> Dict[str, Dict[str, float]]:
    """Compute per-model metrics on a flattened subset of days."""
    # Flatten selected days into hour-level arrays
    y_true = y_true_arr[day_idx].ravel()
    prices_da = prices_da_arr[day_idx].ravel()
    prices_bal = prices_bal_arr[day_idx].ravel()

    per_model = {}
    for m in model_cols:
        mu = mu_arr[m][day_idx].ravel()
        std = std_arr[m][day_idx].ravel()
        metrics, _ = _compute_metrics_for_one_model(
            y_true=y_true, mu=mu, std=std,
            prices_da=prices_da, prices_balancing=prices_bal,
            **util_params,
        )
        per_model[m] = metrics
    return per_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PRED_PATH = Path(
        str(root_dir) + "/experiments/case_studies/electricity_market/predictions"
    )
    RESULT_PATH = Path(
        str(root_dir) + "/experiments/case_studies/electricity_market/results"
    )
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    util_params = dict(
        alpha_param_lambda=2, beta_param_lambda=10,
        alpha_param_k=1.2, beta_param_k=20.8,
        alpha_param_gamma=2, beta_param_gamma=6,
    )

    files_mu = list(PRED_PATH.glob("repeat_*/fold_*/predictions_mu.csv"))
    files_std = list(PRED_PATH.glob("repeat_*/fold_*/predictions_std.csv"))
    files_prices = list(PRED_PATH.glob("repeat_*/fold_*/prices.csv"))

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

    # ================================================================
    # Part 1: Per-day evaluation (original behaviour, for profit curve plotting)
    # ================================================================
    rows = []
    profit_cumsum_dict: Dict[str, Dict[str, float]] = {}
    dec_hours = pd.date_range(
        "2024-12-01", "2025-01-01", freq="h", tz="UTC",
    )[:-1]

    for repeat, fold in common_keys:
        df_mu = pd.read_csv(mu_map[(repeat, fold)])
        df_std = pd.read_csv(std_map[(repeat, fold)])
        df_prices = pd.read_csv(prices_map[(repeat, fold)])

        y_true = df_mu["y_true"].to_numpy().astype(float)
        prices_da = df_prices["Day-Ahead Price"].to_numpy().astype(float)
        prices_balancing = df_prices["Imbalance Price"].to_numpy().astype(float)
        model_cols = [
            c for c in df_mu.columns
            if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
        ]

        hours = dec_hours[repeat * 24 : (repeat + 1) * 24]
        for model in model_cols:
            mu = df_mu[model].to_numpy().astype(float)
            std = df_std[model].to_numpy().astype(float)
            metrics, profit_cumsum = _compute_metrics_for_one_model(
                y_true=y_true, mu=mu, std=std,
                prices_da=prices_da, prices_balancing=prices_balancing,
                **util_params,
            )
            rows.append({"repeat": repeat, "fold": fold, "model": model, **metrics})

            if model not in profit_cumsum_dict:
                profit_cumsum_dict[model] = {}
            base_profit = profit_cumsum_dict[model].get(
                str(hours[0] - pd.Timedelta(hours=1)), 0.0,
            )
            profit_cumsum_dict[model].update({
                f"{t}": (base_profit + profit_cumsum[i])
                for i, t in enumerate(hours)
            })

    pd.DataFrame(profit_cumsum_dict).to_csv(
        RESULT_PATH / "profit_cumsum.csv"
    )

    # ================================================================
    # Part 2: Block bootstrap evaluation
    # ================================================================
    print("Running block bootstrap...")
    (mu_arr, std_arr,
     y_true_arr, prices_da_arr, prices_bal_arr,
     model_cols) = _pool_predictions(
        common_keys, mu_map, std_map, prices_map, hours_per_day=24,
    )
    n_days = y_true_arr.shape[0]
    print(f"Pooled {n_days} days × 24 hours = {n_days * 24} test hours.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    tau_rows_boot = []
    metric_rows_boot = []

    for b in range(N_BOOTSTRAP):
        # Sample block-bootstrap day indices (same total length = n_days)
        day_idx = _sample_day_indices(
            n_days=n_days, n_sample_days=n_days, rng=rng,
        )
        per_model = _compute_metrics_on_day_subset(
            day_idx=day_idx,
            mu_arr=mu_arr, std_arr=std_arr,
            y_true_arr=y_true_arr,
            prices_da_arr=prices_da_arr, prices_bal_arr=prices_bal_arr,
            model_cols=model_cols, util_params=util_params,
        )

        # Store raw per-model metrics for this bootstrap draw
        for m, metrics in per_model.items():
            metric_rows_boot.append({
                "repeat": b,
                "model": m, **metrics,
            })

        # Compute rankings and τ for this draw
        _, tau_rows = _rank_and_tau(per_model)
        for tr in tau_rows:
            tau_rows_boot.append({
                "repeat": b, **tr,
            })

    tau_df_boot = pd.DataFrame(tau_rows_boot)
    tau_df_boot.to_csv(
        RESULT_PATH / "kendall_by_repeat.csv", index=False,
    )
    pd.DataFrame(metric_rows_boot).to_csv(
        RESULT_PATH / "metric_values_by_repeat.csv", index=False,
    )

    # ================================================================
    # Part 3: Summaries
    # ================================================================
    def summarize_alignment(
        df: pd.DataFrame, value_col: str, group_cols: List[str],
    ) -> pd.DataFrame:
        g = df.groupby(group_cols)[value_col]
        return g.agg(
            mean="mean",
            median="median",
            q05=lambda x: np.quantile(x, 0.05),
            q95=lambda x: np.quantile(x, 0.95),
        ).reset_index()

    # Bootstrap summary — broken out by block size
    summarize_alignment(
        tau_df_boot, "kendall_tau", ["utility", "metric"],
    ).to_csv(RESULT_PATH / "kendall_summary_over_repeats.csv", index=False)

    print("Done. Results written to:", RESULT_PATH)


if __name__ == "__main__":
    main()
