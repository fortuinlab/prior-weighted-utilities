from pathlib import Path
import sys
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style_icmlish,
)

fig_dir = root_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
set_paper_style_icmlish()

df_windpower_profit_cumsum = pd.read_csv(
    root_dir / "experiments/case_studies/electricity_market/results/profit_cumsum.csv", index_col=0
)
df_windpower_profit_cumsum.index = pd.to_datetime(df_windpower_profit_cumsum.index)

fig, ax = plt.subplots(
    1, 3, figsize=(7.1, 2.3),
    gridspec_kw={"width_ratios": [0.6, 0.2, 0.2]},
)
models = df_windpower_profit_cumsum.columns

cols = sns.color_palette()
colors = [cols[0]] + [cols[8]] + [cols[1]] + [cols[4]] + [cols[2]]

for model, color in zip(models, colors):
    day_slice = df_windpower_profit_cumsum["2024-12-12 00:00:00":"2024-12-12 15:00:00"]
    x = day_slice.index
    y = day_slice[model].values

    # plot the line
    ax[0].plot(x, y, color=color, linewidth=1.5, label=model)

    # compute differences
    dy = np.diff(y)
    # pad to match original length
    dy = np.insert(dy, 0, np.nan)

    # indices of each condition
    up = dy > 0
    same = dy == 0
    down = dy < 0

    # plot each marker type
    ax[0].scatter(x[same], y[same], marker='o', color=color, s=70, zorder=3)
    ax[0].scatter(x[down], y[down], marker='x', color=color, s=70, zorder=3)
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))  # start at 12h, every 12h
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Hh'))
ax[0].set_ylabel("Cumulative profit")
ax[0].tick_params(axis="both")


# --- Legend for model lines ---
# collect handles and labels from the line plots
line_handles, line_labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles=line_handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.31, -0.0),
    columnspacing=0.8,
)

same_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=8, label='Abstainment')
down_handle = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                            markersize=8, label='Loss')

fig.legend(
    handles=[same_handle, down_handle],
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.31, -0.08),
    columnspacing=0.8,
)


th_kaggle = np.load(root_dir / "experiments/case_studies/credit_approval/results/kaggle/thresholds.npy")
th_kaggle = th_kaggle[(th_kaggle < 1.0) & (th_kaggle > 0.0)]
th_pakdd = np.load(root_dir / "experiments/case_studies/credit_approval/results/pakdd/thresholds.npy")
th_pakdd = th_pakdd[(th_pakdd < 1.0) & (th_pakdd > 0.0)]

# Add histogram to ax[1]
ax[1].hist(th_kaggle, bins=20, alpha=0.6, label='Kaggle', color=sns.color_palette()[0], edgecolor='black')
ax[1].hist(th_pakdd, bins=20, alpha=0.6, label='PAKDD', color=sns.color_palette()[1], edgecolor='black')
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Frequency')
ax[1].tick_params(axis="both")
ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
bin_handles, bin_labels = ax[1].get_legend_handles_labels()
fig.legend(
    handles=bin_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.68, -0.08),
    columnspacing=0.8,
)

p2p_k_fracs = np.load(root_dir / "experiments/case_studies/p2p_lending/results/p2p_k_fracs.npy")

ax[2].hist(p2p_k_fracs * 100, bins=20, alpha=0.6, label='Lending Club', color=sns.color_palette()[2], edgecolor='black')
ax[2].set_xlabel(r"$k/n$ (%)")
ax[2].set_ylabel('Frequency')
ax[2].tick_params(axis="both")
ax[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
bin_handles, bin_labels = ax[2].get_legend_handles_labels()
fig.legend(
    handles=bin_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.905, -0.08),
    columnspacing=0.8,
)

fname = "cumsum_thresholds_ks.pdf"

fig.tight_layout(w_pad=1.2, h_pad=0.8)
fig.savefig(fig_dir / fname, bbox_inches="tight")
plt.close(fig)
