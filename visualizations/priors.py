import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
from pathlib import Path
import seaborn as sns

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style_NeurIPSish,
)


def beta_mode(a, b):
    if a > 1 and b > 1:
        return (a - 1) / (a + b - 2)
    if a <= 1 and b > 1:
        return 0.0
    if a > 1 and b <= 1:
        return 1.0
    return 0.0


def plot_beta_density(
    alpha_c, beta_c,
    alpha_k_bc, beta_k_bc,

    alpha_lambda, beta_lambda,
    alpha_k_r, beta_k_r,
    alpha_gamma, beta_gamma,

    n=2000,
):
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style_NeurIPSish()
    set_paper_style_NeurIPSish()
    x = np.linspace(1e-6, 1 - 1e-6, n)

    # One row, two columns
    fig, axes = plt.subplots(
        1, 2,
        # figsize=(6.8, 2),
        # figsize = (6.8, 3.8)
        figsize=(5.5, 1.6),
        sharey=True
    )

    # ======================================================
    # LEFT PANEL
    # ======================================================
    ax = axes[0]

    y_c = beta.pdf(x, alpha_c, beta_c)
    y_k = beta.pdf(x, alpha_k_bc, beta_k_bc)

    m_c = beta_mode(alpha_c, beta_c)
    m_k = beta_mode(alpha_k_bc, beta_k_bc)

    cols = sns.color_palette()

    (lc,) = ax.plot(x, y_c, linewidth=2.0, label=r"$\theta=c$", color=cols[0])
    (lk,) = ax.plot(x, y_k, linewidth=2.0, label=r"$\theta=k/n$", color=cols[1])
    ax.axvline(0, linewidth=2.0, color=lk.get_color(), ymax=0.2)

    ax.axvline(m_c, linestyle=":", linewidth=1.5, color=lc.get_color())
    ax.axvline(m_k, linestyle=":", linewidth=1.5, color=lk.get_color())

    ax.set_xlabel(r"utility parameter $\theta$")
    ax.set_ylabel(r"$\pi(\theta)$")

    # ---- add "mode" entry to legend without drawing a line ----
    mode_handle = Line2D([], [], linestyle=":", linewidth=1.5, color="black", label="mode")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [mode_handle],
              labels=labels + ["mode"],
              frameon=False, loc="upper right", handlelength=2.2)

    # ======================================================
    # RIGHT PANEL
    # ======================================================
    ax = axes[1]

    y_lambda = beta.pdf(x, alpha_lambda, beta_lambda)
    y_k_r = beta.pdf(x, alpha_k_r, beta_k_r)
    y_gamma = beta.pdf(x, alpha_gamma, beta_gamma)

    m_lambda = beta_mode(alpha_lambda, beta_lambda)
    m_k_r = beta_mode(alpha_k_r, beta_k_r)
    m_gamma = beta_mode(alpha_gamma, beta_gamma)

    (lr1,) = ax.plot(
        x, y_lambda, linewidth=2.0,
        label=r"$\theta=\lambda$", color=cols[0]
    )
    (lr2,) = ax.plot(
        x, y_k_r, linewidth=2.0,
        label=r"$\theta=k/n$", color=cols[1]
    )
    (lr3,) = ax.plot(
        x, y_gamma, linewidth=2.0,
        label=r"$\theta=\gamma$", color=cols[2]
    )
    ax.axvline(0, linewidth=2.0, color=lr2.get_color(), ymax=0.2)

    ax.axvline(m_lambda, linestyle=":", linewidth=1.5, color=lr1.get_color())
    ax.axvline(m_k_r, linestyle=":", linewidth=1.5, color=lr2.get_color())
    ax.axvline(m_gamma, linestyle=":", linewidth=1.5, color=lr3.get_color())
    ax.set_xlabel(r"utility parameter $\theta$")

    # ---- add "mode" entry to legend without drawing a line ----
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [mode_handle],
              labels=labels + ["mode"],
              frameon=False, loc="upper right", handlelength=2.2)

    # ======================================================
    # Shared cleanup
    # ======================================================
    ymax_left = max(y_c.max(), y_k.max())
    ymax_right = max(y_lambda.max(), y_k_r.max(), y_gamma.max())
    ymax = max(ymax_left, ymax_right)

    for ax in axes:
        ax.set_xlim(-0.01, 1)
        ax.set_ylim(0, 1.05 * ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.set_xticks(np.linspace(0, 1, 6))

        # BOTH vertical and horizontal gridlines
        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle=":",
            linewidth=0.6,
            alpha=0.5,
        )

    fig.tight_layout(w_pad=1.2)

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    fig.savefig(fig_dir / "priors.pdf", bbox_inches="tight")
    plt.close(fig)


plot_beta_density(
    # left panel
    alpha_c=2, beta_c=10,
    alpha_k_bc=1.2, beta_k_bc=20.8,

    # right panel
    alpha_lambda=2, beta_lambda=10,
    alpha_k_r=1.2, beta_k_r=20.8,
    alpha_gamma=2, beta_gamma=6,

    n=2000,
)
