import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import beta
from pathlib import Path
import seaborn as sns

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style_icmlish,
)

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


def beta_mode(a, b):
    if a > 1 and b > 1:
        return (a - 1) / (a + b - 2)
    if a <= 1 and b > 1:
        return 0.0
    if a > 1 and b <= 1:
        return 1.0
    return 0.0


def plot_beta_density():
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style_icmlish()
    x = np.linspace(1e-6, 1 - 1e-6, 2000)

    fig, ax = plt.subplots(
        1, 3, figsize=(6.8, 2), sharey=True
    )

    y_c_base = beta.pdf(x, 2, 10)
    y_c_slight = beta.pdf(x, 3, 9)
    y_c_strong = beta.pdf(x, 10, 10)
    y_c_extreme = beta.pdf(x, 10, 2)

    y_k_base = beta.pdf(x, 1.2, 20.8)
    y_k_slight = beta.pdf(x, 1.5, 25.5)
    y_k_strong = beta.pdf(x, 2, 10)
    y_k_extreme = beta.pdf(x, 10, 10)

    y_gamma_base = beta.pdf(x, 2, 6)
    y_gamma_slight = beta.pdf(x, 2, 10)
    y_gamma_strong = beta.pdf(x, 5, 7)
    y_gamma_extreme = beta.pdf(x, 9, 3)

    ax[0].plot(x, y_c_base, linewidth=2.0, label="base", color=cols(0.2))
    ax[0].plot(x, y_c_slight, linewidth=2.0, label="slight", color=cols(0.4))
    ax[0].plot(x, y_c_strong, linewidth=2.0, label="strong", color=cols(0.6))
    ax[0].plot(x, y_c_extreme, linewidth=2.0, label="extreme", color=cols(0.8))
    ax[0].set_xlabel(r"$\theta \in \{c,\lambda\}$")
    ax[0].set_ylabel(r"$\pi(\theta)$")

    ax[1].plot(x, y_k_base, linewidth=2.0, label="base", color=cols(0.2))
    ax[1].plot(x, y_k_slight, linewidth=2.0, label="slight", color=cols(0.4))
    ax[1].plot(x, y_k_strong, linewidth=2.0, label="strong", color=cols(0.6))
    ax[1].plot(x, y_k_extreme, linewidth=2.0, label="extreme", color=cols(0.8))
    ax[1].set_xlabel(r"$k/n$")
    ax[1].set_ylabel(r"$\pi(k/n)$")

    ax[2].plot(x, y_gamma_base, linewidth=2.0, label="base", color=cols(0.2))
    ax[2].plot(x, y_gamma_slight, linewidth=2.0, label="slight", color=cols(0.4))
    ax[2].plot(x, y_gamma_strong, linewidth=2.0, label="strong", color=cols(0.6))
    ax[2].plot(x, y_gamma_extreme, linewidth=2.0, label="extreme", color=cols(0.8))
    ax[2].set_xlabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\pi(\gamma)$")

    # ======================================================
    # Shared cleanup
    # ======================================================
    ymax_left = max(y_c_base.max(), y_c_slight.max(), y_c_strong.max(), y_c_extreme.max())
    ymax_center = max(y_k_base.max(), y_k_slight.max(), y_k_strong.max(), y_k_extreme.max())
    ymax_right = max(y_gamma_base.max(), y_gamma_slight.max(), y_gamma_strong.max(), y_gamma_extreme.max())
    ymax = max(ymax_left, ymax_center, ymax_right)

    for a in ax:
        a.set_xlim(-0.01, 1)
        a.set_ylim(0, 1.05 * ymax)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(direction="out", length=3, width=0.8)
        a.set_xticks(np.linspace(0, 1, 6))

        # BOTH vertical and horizontal gridlines
        a.grid(
            True,
            which="major",
            axis="both",
            linestyle=":",
            linewidth=0.6,
            alpha=0.5,
        )

    fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    fig.savefig(fig_dir / "prior_misspecification.pdf", bbox_inches="tight")
    plt.close(fig)


plot_beta_density()
