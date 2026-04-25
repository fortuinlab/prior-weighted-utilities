from __future__ import annotations
import argparse

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style_NeurIPSish,
    make_grouped_boxplot_on_ax,
    load_corr_by_repeat,
    prep_panel_df,
)

# ---------- colors and metrics ----------
cols = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
labels = ["base", "slight", "strong", "extreme"]

legend_handles = [
    plt.Line2D(
        [0], [0],
        color=cols(0.2 + 0.2 * i),
        lw=8,
        solid_capstyle="butt",
    )
    for i in range(len(labels))
]

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
bc_titles = {
    "u_c": "Binary Decision",
    "u_k": r"Top-$k$ Selection",
}
reg_titles = {
    "u_lambda": "Selective Prediction",
    "u_k_gamma": r"Top-$k$ Selection",
}


def plot_alignment(task: str):
    if task == "binary_classification":
        METRICS = METRICS_BC
        METRICS_LABELS = METRICS_BC_LABELS
        titles = bc_titles
        utilities = ("u_c", "u_k")
    elif task == "regression":
        METRICS = METRICS_REG
        METRICS_LABELS = METRICS_REG_LABELS
        titles = reg_titles
        utilities = ("u_lambda", "u_k_gamma")
    result_root_base = root_dir / f"experiments/benchmark_datasets/{task}/results"
    results_root_slight = root_dir / f"experiments/benchmark_datasets/{task}/results_slight"
    results_root_strong = root_dir / f"experiments/benchmark_datasets/{task}/results_strong"
    results_root_extreme = root_dir / f"experiments/benchmark_datasets/{task}/results_extreme"
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style_NeurIPSish()
    figsize = (5.5, 3.1)

    # ---------- load data ----------
    df_base = load_corr_by_repeat(result_root_base)
    df_slight = load_corr_by_repeat(results_root_slight)
    df_strong = load_corr_by_repeat(results_root_strong)
    df_extreme = load_corr_by_repeat(results_root_extreme)

    # ---------- figure layout ----------
    # 2 rows x 2 cols = 4 plots
    # share x-axis per column
    fig, axes = plt.subplots(
        2, 1, figsize=figsize,
        sharex="col",
    )

    # ---------- BC panels (left column) ----------
    # Top: u_c avg over c; Bottom: u_k avg over k
    for row, fam in enumerate(utilities):
        ax = axes[row]
        df_plot_base = prep_panel_df(df_base, fam)
        df_plot_slight = prep_panel_df(df_slight, fam)
        df_plot_strong = prep_panel_df(df_strong, fam)
        df_plot_extreme = prep_panel_df(df_extreme, fam)

        make_grouped_boxplot_on_ax(
            ax=ax,
            dfs=[df_plot_base, df_plot_slight, df_plot_strong, df_plot_extreme],
            group_labels=["base", "slight", "strong", "extreme"],
            metrics=METRICS,
            metric_labels=METRICS_LABELS,
            colors=[cols(0.2), cols(0.4), cols(0.6), cols(0.8)],
            title=titles.get(fam, fam),
            ylim=[-1.05, 1.05],
        )
    # ---------- legend (shared) ----------
    fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.text(
        -0.01,            # x-position (figure coordinates)
        0.5,             # centered vertically
        r"alignment (Kendall's $\tau$)",
        va="center",
        rotation="vertical",
    )
    fname = f"sensitivity_analysis_{task}.pdf"

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)

    return fig


def main(task: str):
    plot_alignment(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="binary_classification",
        help="Can be 'binary_classification' and 'regression'."
    )

    args = parser.parse_args()
    main(**vars(args))
