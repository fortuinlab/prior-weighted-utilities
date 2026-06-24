import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def set_paper_style():
    """Matplotlib style used across all paper figures."""
    plt.rcParams.update({
        # --- font: Times-like serif ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # --- sizes / line widths ---
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,

        # editable text in vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_boxplot_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: list[str],
    metric_labels: list[str],
    colors: list[tuple[float, float, float]],
    title: str,
    ylim: list[float] = None,
    value_col: str = "kendall_tau",
):
    x = np.arange(len(metrics)) + 1
    data = [df[df["metric"] == m][value_col].values for m in metrics]

    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=30)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(y=0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)


def make_grouped_boxplot_on_ax(
    ax: plt.Axes,
    dfs: list[pd.DataFrame],
    group_labels: list[str],
    metrics: list[str],
    metric_labels: list[str],
    colors: list[str],
    title: str,
    ylim: list[float] | None = None,
    value_col: str = "kendall_tau",
):
    assert len(dfs) == len(colors) == len(group_labels)

    x = np.arange(len(metrics)) + 1  # centers for each metric group

    n_groups = len(dfs)
    group_width = 0.8                 # total width occupied per metric (all boxes)
    box_width = group_width / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * box_width

    # draw one boxplot call per df (gives you independent coloring & control)
    for j, (dfj, color, glab) in enumerate(zip(dfs, colors, group_labels)):
        data_j = [dfj[dfj["metric"] == m][value_col].values for m in metrics]
        pos_j = x + offsets[j]

        bp = ax.boxplot(
            data_j,
            positions=pos_j,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops=dict(color="black", linewidth=2),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    # axes cosmetics
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=30)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(y=0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)


def load_corr_by_repeat(
    results_root: Path,
    filename: str = "kendall_by_repeat.csv",
) -> pd.DataFrame:
    """
    Loads a per-repeat correlation CSV for one or more datasets and concatenates them.
    Adds a 'dataset' column.
    """
    datasets = [p.name for p in results_root.iterdir() if p.is_dir()]
    dfs = []
    for d in datasets:
        fp = results_root / d / filename
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        df = pd.read_csv(fp)
        df["dataset"] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prep_panel_df(
    df: pd.DataFrame,
    fam_prefix: str,
    value_col: str = "kendall_tau",
) -> pd.DataFrame:
    """
    1) Select utilities belonging to a given family (by prefix).
    2) Average the chosen alignment measure over utilities within the family,
       for each (dataset, repeat, metric) combination.
    """
    df_plot = df[df["utility"].str.startswith(fam_prefix)].copy()
    df_plot = df_plot.groupby(
        ["dataset", "repeat", "metric"], as_index=False
    )[value_col].mean()
    df_plot = df_plot.dropna(subset=[value_col])
    return df_plot
