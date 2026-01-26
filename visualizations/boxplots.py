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
    set_paper_style_icmlish,
    make_boxplot_on_ax,
    load_corr_by_repeat,
    dataset_filename,
    prep_panel_df,
)

results_root_bc = root_dir / "experiments/benchmark_datasets/binary_classification/results"
results_root_reg = root_dir / "experiments/benchmark_datasets/regression/results"
results_root_electricity = root_dir / "experiments/case_studies/electricity_market/results"
results_root_credit = root_dir / "experiments/case_studies/credit_approval/results"
results_root_p2p = root_dir / "experiments/case_studies/p2p_lending/results"

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


def plot_alignment(
    experiment: str,
):
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style_icmlish()
    if experiment == "benchmark":
        figsize = (6.8, 3.8)
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
        df_bc, datasets_bc = load_corr_by_repeat(results_root_bc)
        df_reg, datasets_reg = load_corr_by_repeat(results_root_reg)

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
            df_plot = prep_panel_df(df_bc, fam)

            make_boxplot_on_ax(
                ax=ax,
                df=df_plot,
                metrics=METRICS_BC,
                metric_labels=METRICS_BC_LABELS,
                colors=COLOR_MAP_BC[fam],
                title=bc_titles.get(fam, fam),
                ylim=[-1.05, 1.05],
            )
            if row == 0:
                ax.tick_params(labelbottom=False)
        # ---------- REG panels (right column) ----------
        for row, fam in enumerate(("u_lambda", "u_k_gamma")):
            ax = axes[row, 1]
            df_plot = prep_panel_df(df_reg, fam)

            make_boxplot_on_ax(
                ax=ax,
                df=df_plot,
                metrics=METRICS_REG,
                metric_labels=METRICS_REG_LABELS,
                colors=COLOR_MAP_REG[fam],
                title=reg_titles.get(fam, fam),
                ylim=[-1.05, 1.05],
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
            "alignment (Kendall's τ)",
            va="center",
            rotation="vertical",
        )
        bc_name = dataset_filename(datasets_bc)
        reg_name = dataset_filename(datasets_reg)
        fname = f"{bc_name}_{reg_name}.pdf"

    if experiment == "electricity":
        figsize = (4.0, 2.3)
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
        df = pd.read_csv(results_root_electricity / "kendall_by_repeat.csv")
        df = df.groupby(["repeat", "metric"], as_index=False)["kendall_tau"].mean()
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
            ylim=[-1.05, 1.05],
        )
        ax.set_ylabel("alignment (Kendall's τ)")
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.56, -0.05),
        )
        fname = "electricity.pdf"

    if experiment == "credit_and_p2p":
        figsize = (6.8, 2.3)
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
        df_credit, _ = load_corr_by_repeat(results_root_credit)
        df_credit = df_credit.groupby(["dataset", "repeat", "metric"], as_index=False)["kendall_tau"].mean()
        file_p2p = results_root_p2p / "kendall_by_repeat.csv"
        df_p2p = pd.read_csv(file_p2p)
        df_p2p_plot = df_p2p[df_p2p["utility"] == "P2P-Util"].copy()
        df_p2p_plot = df_p2p_plot.groupby(["repeat", "metric"], as_index=False)["kendall_tau"].mean()
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
            ylim=[-0.3, 1.05],
        )
        ax[0].set_ylabel("alignment (Kendall's τ)")
        make_boxplot_on_ax(
            ax=ax[1],
            df=df_p2p_plot,
            metrics=METRICS_BC,
            metric_labels=METRICS_BC_LABELS,
            colors=COLOR_MAP_BC["P2P-Util"],
            title="P2P Lending",
            ylim=[-0.3, 1.05],
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )
        fname = "credit_and_p2p.pdf"

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)

    return fig


def main(experiment: str):
    plot_alignment(experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, default="benchmark",
        help="Which experiment to plot. Can be 'benchmark', 'electricity', and 'credit_and_p2p'."
    )

    args = parser.parse_args()
    main(**vars(args))
