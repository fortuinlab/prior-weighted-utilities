import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,             # high is good
    brier_score_loss,           # low is good
    log_loss,                   # low is good
)

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.metrics import (   # noqa: E402
    ece_binary_classification,
    mce_binary_classification,
    rauc_binary_classification,
    error_detection_binary_classification,
)
from src.metrics.utility import (  # noqa: E402
    credit_utility,
    binary_decision_pwu,
    topk_utility_binary_classification_pwu,
)
from src.utils import (  # noqa: E402
    compute_average_credit_quantities,
    compute_metric_correlation,
)


def _compute_metrics_for_one_model(
    dataset: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    alpha_param_c: float,
    beta_param_c: float,
    alpha_param_k: float,
    beta_param_k: float,
    credit_features: pd.DataFrame,
    average_credit_quantities: list,
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

    bd_p = binary_decision_pwu(probs, y_true, alpha_param_c, beta_param_c)
    topk_p = topk_utility_binary_classification_pwu(
        probs, y_true, alpha_param_k, beta_param_k
    )

    out = {
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

    util, threshold = credit_utility(
        probs, y_true, credit_features, average_credit_quantities, dataset=dataset
    )

    out["Credit-Util ↑"] = float(util)
    return out, threshold


def main(dataset: str = "kaggle"):

    PRED_PATH = Path(str(root_dir) + f"/experiments/case_studies/credit_approval/predictions/{dataset}")
    DATA_PATH = Path(str(root_dir) + f"/data/case_studies/credit_approval/preprocessed/{dataset}")

    # Load dataset once
    X = pd.read_parquet(DATA_PATH / "X.parquet")

    df_features = pd.read_csv(
        str(root_dir) + f"/data/case_studies/credit_approval/preprocessed/{dataset}/features.csv"
    )
    cl_avg, pi_0, pi_1, r_avg = compute_average_credit_quantities(df_features, dataset=dataset)

    # Load split indices
    with open(DATA_PATH / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    # configure utilities
    alpha_param_c = 2.0
    beta_param_c = 10.0
    alpha_param_k = 1.2
    beta_param_k = 20.8

    RESULT_PATH = Path(str(root_dir) + f"/experiments/case_studies/credit_approval/results/{dataset}")
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    files = sorted(PRED_PATH.glob("repeat_*/fold_*/predictions.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No predictions.csv files found under {PRED_PATH}")

    rows = []
    thresholds = np.array([])
    for fp in files:
        repeat_str = fp.parents[1].name.replace("repeat_", "")
        fold_str = fp.parents[0].name.replace("fold_", "")
        repeat, fold = int(repeat_str), int(fold_str)

        # load features (required for credit utility)
        split = splits[repeat][fold - 1]
        test_idx = split["test_idx"]
        X_te_df = X.iloc[test_idx].copy()

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
            metrics, threshold = _compute_metrics_for_one_model(
                dataset=dataset,
                y_true=y_true,
                probs=probs,
                alpha_param_c=alpha_param_c,
                beta_param_c=beta_param_c,
                alpha_param_k=alpha_param_k,
                beta_param_k=beta_param_k,
                credit_features=X_te_df,
                average_credit_quantities=[cl_avg, pi_0, pi_1, r_avg],
            )
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics,
                }
            )

        thresholds = np.append(thresholds, threshold)

    metrics_df = pd.DataFrame(rows)
    np.save(RESULT_PATH / "thresholds.npy", thresholds)

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
        "Brier ↓": True,
        "Acc ↑": False,
        "ECE ↓": True,
        "MCE ↓": True,
        "R-AUC ↓": True,
        "E-Det ↑": False,
        "BD-PWU ↓": True,
        "TopK-PWU ↓": True,
        "Credit-Util ↑": False,
    }

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
    utilities = [clean_metric_name("Credit-Util ↑")]

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="kaggle", help="Which dataset to use, either 'kaggle' or 'pakdd'."
    )

    args = parser.parse_args()
    main(**vars(args))
