from __future__ import annotations
import argparse

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style_NeurIPSish,
    make_boxplot_on_ax,
    load_corr_by_repeat,
    prep_panel_df,
)

MEASURE_CONFIG = {
    "kendall": {
        "file": "kendall_by_repeat.csv",
        "col": "kendall_tau",
        "ylabel": r"alignment (Kendall's $\tau$)",
        "ylim_default": [-1.05, 1.05],
    },
    "top1": {
        "file": "kendall_by_repeat.csv",   # top1 lives in the same file
        "col": "top1",
        "ylabel": "top-1 agreement",
        "ylim_default": [-0.05, 1.05],
    },
    "top3": {
        "file": "kendall_by_repeat.csv",   # same file
        "col": "top3",
        "ylabel": "top-3 agreement (Jaccard)",
        "ylim_default": [-0.05, 1.05],
    },
    "pearson": {
        "file": "pearson_by_repeat.csv",
        "col": "pearson_r",
        "ylabel": r"alignment (Pearson's $r$)",
        "ylim_default": [-1.05, 1.05],
    },
}

results_root_bc = root_dir / "experiments/benchmark_datasets/binary_classification/results"
results_root_reg = root_dir / "experiments/benchmark_datasets/univariate_regression/results"
results_root_electricity = root_dir / "experiments/case_studies/electricity_market/results"
results_root_credit = root_dir / "experiments/case_studies/credit_approval/results"
results_root_p2p = root_dir / "experiments/case_studies/p2p_lending/results"
results_root_mc = root_dir / "experiments/benchmark_datasets/multiclass_classification/results"
results_root_mr = root_dir / "experiments/benchmark_datasets/multivariate_regression/results"

# ---------- colors and metrics ----------
cols = sns.color_palette()
PLAUSIBLE = cols[2]
PATHOLOGICAL = cols[1]
NOTALIGNED = cols[0]
METRICS_BC = [
    "NLL", "Brier", "Acc",
    "ECE", "MCE",
    "R-AUC", "E-Det",
    "BD-PWU", "TopK-PWU"
]
METRICS_BC_LABELS = [
    "NLL", "BS", "Acc",
    "ECE", "MCE",
    "R-AUC", "E-Det",
    r"$M_{\pi_c}$", r"$M_{\pi_k}$"]
METRICS_REG = [
    "NLL", "MSE",
    "ECE", "MCE",
    "R-AUC", "E-Det",
    "SP-PWU", "TopK-PWU"
]
METRICS_REG_LABELS = [
    "NLL", "MSE",
    "ECE", "MCE",
    "R-AUC", "E-Det",
    r"$M_{\pi_\lambda}$",
    r"$M_{\pi_\phi}$"
]
METRICS_MC = [
    "NLL", "Brier", "ECE",
    "MCD-PWU"
]
METRICS_MC_LABELS = [
    "NLL", "BS", "ECE",
    r"$M_{\pi_{j,c}}$"]
METRICS_MR = [
    "NLL", "MSE", "ES",
    "SP-PWU"
]
METRICS_MR_LABELS = [
    "NLL", "MSE", "ES",
    r"$M_{\pi_\lambda}$",
]


def plot_alignment(experiment: str, measure: str = "kendall"):
    cfg = MEASURE_CONFIG[measure]
    val_col = cfg["col"]
    ylabel = cfg["ylabel"]
    ylim = cfg["ylim_default"]

    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style_NeurIPSish()
    if experiment == "benchmark":
        figsize = (5.5, 3.1)
        COLOR_MAP_BC = {
            "u_c": [
                PATHOLOGICAL, PATHOLOGICAL, PATHOLOGICAL,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                PLAUSIBLE, NOTALIGNED
            ],
            "u_k": [
                NOTALIGNED, NOTALIGNED, PATHOLOGICAL,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, PLAUSIBLE
            ],
        }
        COLOR_MAP_REG = {
            "u_lambda": [
                PATHOLOGICAL, PATHOLOGICAL,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                PLAUSIBLE, NOTALIGNED,
            ],
            "u_k_gamma": [
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, PLAUSIBLE,
            ],
        }
        legend_handles = [
            Patch(facecolor=NOTALIGNED, label="not decision-aligned"),
            Patch(facecolor=PATHOLOGICAL, label="pathological prior"),
            Patch(facecolor=PLAUSIBLE, label="plausible prior (ours)"),
        ]
        # ---------- load data ----------
        df_bc = load_corr_by_repeat(results_root_bc, cfg["file"])
        df_reg = load_corr_by_repeat(results_root_reg, cfg["file"])

        # ---------- figure layout ----------
        # 2 rows x 2 cols = 4 plots
        # share x-axis per column
        fig, axes = plt.subplots(
            2, 2, figsize=figsize,
            sharex="col",
            sharey=False,
            gridspec_kw={'width_ratios': [1, 0.9]},
        )
        bc_titles = {
            "u_c": "Binary Decision",
            "u_k": r"Top-$k$ Selection",
        }
        reg_titles = {
            "u_lambda": "Selective Prediction",
            "u_k_gamma": r"Top-$k$ Selection",
        }
        # ---------- BC panels (left column) ----------
        # Top: u_c avg over c; Bottom: u_k avg over k
        for row, fam in enumerate(("u_c", "u_k")):
            ax = axes[row, 0]
            df_plot = prep_panel_df(df_bc, fam, val_col)

            make_boxplot_on_ax(
                ax=ax,
                df=df_plot,
                metrics=METRICS_BC,
                metric_labels=METRICS_BC_LABELS,
                colors=COLOR_MAP_BC[fam],
                title=bc_titles.get(fam, fam),
                ylim=ylim,
                value_col=val_col,
            )
            if row == 0:
                ax.tick_params(labelbottom=False)
        # ---------- REG panels (right column) ----------
        for row, fam in enumerate(("u_lambda", "u_k_gamma")):
            ax = axes[row, 1]
            df_plot = prep_panel_df(df_reg, fam, val_col)

            make_boxplot_on_ax(
                ax=ax,
                df=df_plot,
                metrics=METRICS_REG,
                metric_labels=METRICS_REG_LABELS,
                colors=COLOR_MAP_REG[fam],
                title=reg_titles.get(fam, fam),
                ylim=ylim,
                value_col=val_col,
            )
            if row == 0:
                ax.tick_params(labelbottom=False)
        # ---------- legend (shared) ----------
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )
        # Shared y-axis label for left column (BC)
        fig.text(
            -0.01,            # x-position (figure coordinates)
            0.5,             # centered vertically
            ylabel,
            va="center",
            rotation="vertical",
        )

    if experiment == "electricity":
        figsize = (3.25, 1.9)
        COLOR_MAP_REG = {
            "Bid-Util": [
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                PLAUSIBLE, PLAUSIBLE,
            ],
        }
        legend_handles = [
            Patch(facecolor=NOTALIGNED, label="conventional metrics"),
            Patch(facecolor=PLAUSIBLE, label="PWUs (ours)"),
        ]
        df = pd.read_csv(results_root_electricity / cfg["file"])
        df = df.groupby(["repeat", "metric"], as_index=False)[val_col].mean()
        df = df.dropna(subset=[val_col])
        fig, ax = plt.subplots(
            1, 1, figsize=figsize,
        )
        make_boxplot_on_ax(
            ax=ax,
            df=df,
            metrics=METRICS_REG,
            metric_labels=METRICS_REG_LABELS,
            colors=COLOR_MAP_REG["Bid-Util"],
            title="Electricity Market Bidding",
            ylim=ylim,
            value_col=val_col,
        )
        ax.set_ylabel(ylabel)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.56, -0.05),
        )

    if experiment == "credit_and_p2p":
        figsize = (5.5, 1.9)
        COLOR_MAP_BC = {
            "Credit-Util": [
                NOTALIGNED, NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                PLAUSIBLE, PLAUSIBLE
            ],
            "P2P-Util": [
                NOTALIGNED, NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                NOTALIGNED, NOTALIGNED,
                PLAUSIBLE, PLAUSIBLE
            ],
        }
        legend_handles = [
            Patch(facecolor=NOTALIGNED, label="conventional metrics"),
            Patch(facecolor=PLAUSIBLE, label="PWUs (ours)"),
        ]
        df_credit = load_corr_by_repeat(results_root_credit, cfg["file"])
        df_credit = df_credit.groupby(["dataset", "repeat", "metric"], as_index=False)[val_col].mean()
        df_credit = df_credit.dropna(subset=[val_col])

        df_p2p = pd.read_csv(results_root_p2p / cfg["file"])
        df_p2p_plot = df_p2p[df_p2p["utility"] == "P2P-Util"].copy()
        df_p2p_plot = df_p2p_plot.groupby(["repeat", "metric"], as_index=False)[val_col].mean()
        df_p2p_plot = df_p2p_plot.dropna(subset=[val_col])
        fig, ax = plt.subplots(
            1, 2, figsize=figsize,
        )
        make_boxplot_on_ax(
            ax=ax[0],
            df=df_credit,
            metrics=METRICS_BC,
            metric_labels=METRICS_BC_LABELS,
            colors=COLOR_MAP_BC["Credit-Util"],
            title="Credit Approval",
            ylim=ylim,
            value_col=val_col,
        )
        ax[0].set_ylabel(ylabel)
        make_boxplot_on_ax(
            ax=ax[1],
            df=df_p2p_plot,
            metrics=METRICS_BC,
            metric_labels=METRICS_BC_LABELS,
            colors=COLOR_MAP_BC["P2P-Util"],
            title="P2P Lending",
            ylim=ylim,
            value_col=val_col,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )

    if experiment == "benchmark_multi":
        figsize = (5.5, 1.9)
        COLOR_MAP_MC = {
            "u_jc": [
                PATHOLOGICAL, PATHOLOGICAL, NOTALIGNED,
                PLAUSIBLE,
            ],
        }
        COLOR_MAP_MR = {
            "u_lambda": [
                NOTALIGNED, NOTALIGNED, NOTALIGNED,
                PLAUSIBLE,
            ],
        }
        legend_handles = [
            Patch(facecolor=NOTALIGNED, label="not decision-aligned"),
            Patch(facecolor=PATHOLOGICAL, label="pathological prior"),
            Patch(facecolor=PLAUSIBLE, label="plausible prior (ours)"),
        ]
        # ---------- load data ----------
        df_mc = load_corr_by_repeat(results_root_mc, cfg["file"])
        df_mr = load_corr_by_repeat(results_root_mr, cfg["file"])

        # ---------- figure layout ----------
        # 1 row x 2 cols = 2 plots
        # share x-axis per column
        fig, axes = plt.subplots(
            1, 2, figsize=figsize,
            sharex="col",
            sharey=False,
            gridspec_kw={'width_ratios': [1, 0.9]},
        )
        mc_titles = {
            "u_jc": "Binary Decision",
        }
        mr_titles = {
            "u_lambda": "Selective Prediction",
        }
        # ---------- BC panel (left column) ----------
        # u_jc avg over c
        ax = axes[0]
        df_plot = prep_panel_df(df_mc, "u_jc", val_col)

        make_boxplot_on_ax(
            ax=ax,
            df=df_plot,
            metrics=METRICS_MC,
            metric_labels=METRICS_MC_LABELS,
            colors=COLOR_MAP_MC["u_jc"],
            title=mc_titles.get("u_jc", "u_jc"),
            ylim=ylim,
            value_col=val_col,
        )
        # ---------- REG panel (right column) ----------
        ax = axes[1]
        df_plot = prep_panel_df(df_mr, "u_lambda", val_col)

        make_boxplot_on_ax(
            ax=ax,
            df=df_plot,
            metrics=METRICS_MR,
            metric_labels=METRICS_MR_LABELS,
            colors=COLOR_MAP_MR["u_lambda"],
            title=mr_titles.get("u_lambda", "u_lambda"),
            ylim=ylim,
            value_col=val_col,
        )
        # ---------- legend (shared) ----------
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )
        # Shared y-axis label for left column (BC)
        fig.text(
            -0.01,            # x-position (figure coordinates)
            0.5,             # centered vertically
            ylabel,
            va="center",
            rotation="vertical",
        )

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    fig.savefig(fig_dir / f"{experiment}_{measure}.pdf", bbox_inches="tight")
    plt.close(fig)

    return fig


def main(experiment: str, measure: str = "kendall"):
    plot_alignment(experiment, measure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, default="benchmark",
        help="Which experiment to plot. Can be 'benchmark', 'electricity', 'credit_and_p2p', and 'benchmark_multi'."
    )
    parser.add_argument(
        "--measure", type=str, default="kendall",
        choices=list(MEASURE_CONFIG.keys()),
        help="Which alignment measure to plot. Can be 'kendall', 'top1', 'top3', and 'pearson'.",
    )

    args = parser.parse_args()
    main(**vars(args))
