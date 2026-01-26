import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def set_paper_style_icmlish():
    """ICML-ish Matplotlib style used across all paper figures."""
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
    colors: list[str],
    title: str,
    ylim: list[float] = None,
):
    x = np.arange(len(metrics)) + 1
    data = [df[df["metric"] == m]["kendall_tau"].values for m in metrics]

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


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def load_corr_by_repeat(results_root: Path):
    """
    Loads kendall_by_repeat.csv for one or more datasets and concatenates them.
    Adds a 'dataset' column.
    """
    datasets = [p.name for p in results_root.iterdir() if p.is_dir()]
    dfs = []
    for d in datasets:
        fp = results_root / d / "kendall_by_repeat.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        df = pd.read_csv(fp)
        df["dataset"] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), datasets


def dataset_filename(datasets) -> str:
    datasets = _as_list(datasets)
    return "_".join(str(d) for d in datasets)


def prep_panel_df(df: pd.DataFrame, fam_prefix: str) -> pd.DataFrame:
    """
    1) select utilities in family
    2) average over utilities within family per (dataset, repeat, metric)
    3) optionally aggregate across datasets:
        - pool: keep (dataset, repeat, metric)
        - mean_per_repeat: average over datasets per (repeat, metric)
    """
    df_plot = df[df["utility"].str.startswith(fam_prefix)].copy()

    # family average (u_c over sampled c's etc.)
    df_plot = df_plot.groupby(["dataset", "repeat", "metric"], as_index=False)["kendall_tau"].mean()

    return df_plot
