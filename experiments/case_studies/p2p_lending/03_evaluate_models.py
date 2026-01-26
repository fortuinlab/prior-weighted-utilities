import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score  # high is good
from sklearn.metrics import brier_score_loss  # low is good
from sklearn.metrics import log_loss  # low is good

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
    p2p_utility,
    topk_utility_binary_classification_pwu,
)
from src.utils import compute_metric_correlation  # noqa: E402


def _compute_metrics_for_one_model(
    y_true: np.ndarray,
    probs: np.ndarray,
    alpha_param_c: float,
    beta_param_c: float,
    alpha_param_k: float,
    beta_param_k: float,
    credit_lines: np.ndarray,
    terms: np.ndarray,
    lending_rates: np.ndarray,
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

    bd_p = binary_decision_pwu(
        probs, y_true, alpha_param_c, beta_param_c
    )
    topk_p = topk_utility_binary_classification_pwu(
        probs, y_true, alpha_param_k, beta_param_k
    )

    p2p_util, p2p_k_frac, _ = p2p_utility(
        probs, y_true, credit_lines, terms, lending_rates
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
        "P2P-Util ↑": float(p2p_util),
    }

    return out, p2p_k_frac


def main():

    PRED_PATH = Path(
        str(root_dir)
        + "/experiments/case_studies/p2p_lending/predictions"
    )

    # configure utilities
    alpha_param_c = 2.0
    beta_param_c = 10.0
    alpha_param_k = 1.2
    beta_param_k = 20.8

    RESULT_PATH = Path(
        str(root_dir)
        + "/experiments/case_studies/p2p_lending/results"
    )
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    files = sorted(PRED_PATH.glob("repeat_*/fold_*/predictions.csv"))
    files_features = sorted(PRED_PATH.glob("repeat_*/fold_*/features.csv"))

    def _key_from_path(p: Path):
        repeat = int(p.parents[1].name.replace("repeat_", ""))
        fold = int(p.parents[0].name.replace("fold_", ""))
        return repeat, fold

    pred_map = {_key_from_path(p): p for p in files}
    features_map = {_key_from_path(p): p for p in files_features}

    common_keys = sorted(set(pred_map.keys()) & set(features_map.keys()))

    rows = []
    p2p_k_fracs = np.array([])
    for repeat, fold in common_keys:
        fp_pred = pred_map[(repeat, fold)]
        fp_features = features_map[(repeat, fold)]

        df = pd.read_csv(fp_pred)
        df_features = pd.read_csv(fp_features)

        y_true = df["y_true"].to_numpy().astype(int)
        model_cols = [
            c
            for c in df.columns
            if c not in {"repeat", "fold", "seed", "test_idx", "y_true"}
        ]

        credit_lines = df_features["loan_amnt"].to_numpy().astype(float)
        terms = df_features["term"].to_numpy().astype(float)
        lending_rates = df_features["int_rate"].to_numpy().astype(float)

        for model in model_cols:
            probs = df[model].to_numpy().astype(float)
            metrics, p2p_k_frac = _compute_metrics_for_one_model(
                y_true=y_true,
                probs=probs,
                alpha_param_c=alpha_param_c,
                beta_param_c=beta_param_c,
                alpha_param_k=alpha_param_k,
                beta_param_k=beta_param_k,
                credit_lines=credit_lines,
                terms=terms,
                lending_rates=lending_rates,
            )
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model,
                    **metrics,
                }
            )

        p2p_k_fracs = np.append(p2p_k_fracs, p2p_k_frac)

    metrics_df = pd.DataFrame(rows)
    np.save(RESULT_PATH / "p2p_k_fracs.npy", p2p_k_fracs)

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
        "P2P-Util ↑": False,
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
    utilities = [clean_metric_name("P2P-Util ↑")]

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
