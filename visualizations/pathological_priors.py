import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.utils import (   # noqa: E402
    set_paper_style,
)


def plot_pathological_priors(
    eps=0.05,
    y_clip=12.0,
    n=4000,
):
    """
    Visualize the pathological priors implied by NLL, BS, Acc (binary decision)
    and NLL, MSE (selective prediction), as listed in Table 1.

    Left panel (binary decision, theta = c in [0, 1]):
      NLL : pi(c) = 1 / ( c (1 - c) )         (U-shaped, diverges at 0 and 1)
      BS  : pi(c) = 2                         (uniform / flat)
      Acc : pi(c) = 2 * delta_{0.5}(c)        (Dirac spike at 0.5)

    Right panel (selective prediction, theta = lambda):
      NLL : pi(lambda) = eps / lambda^2  for lambda > eps > 0   (heavy left edge)
      MSE : pi(lambda) = delta_{infty}(lambda)                  (Dirac at infinity)

    The right panel uses x in [0, 1] as a *symbolic* axis: x = 0 corresponds
    to lambda = 0, the rightmost tick is relabeled as "infinity" to mark the
    location of the MSE Dirac mass.
    """
    fig_dir = root_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    cols = sns.color_palette()
    col_nll = cols[0]
    col_bs_or_mse = cols[1]
    col_acc = cols[2]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(5.5, 2.4),
        sharey=True,
    )

    # ======================================================
    # LEFT PANEL: binary decision priors over c in [0, 1]
    # ======================================================
    ax = axes[0]

    # NLL: 1 / ( c (1 - c) ).  Diverges at 0 and 1; clip values for plotting.
    c = np.linspace(1e-4, 1 - 1e-4, n)
    y_nll = 1.0 / (c * (1.0 - c))
    y_nll_plot = np.minimum(y_nll, y_clip)

    # Mask out the points where the curve is being clipped so the line ends
    # cleanly *inside* the visible region rather than tracing the cap.
    mask = y_nll <= y_clip
    ax.plot(
        c[mask], y_nll_plot[mask],
        linewidth=2.0, color=col_nll,
        label=r"$\pi_{\text{NLL}}$",
    )
    ax.annotate(
        "", xy=(0.0, y_clip), xytext=(0.0, y_clip - 2.0),
        arrowprops=dict(arrowstyle="-|>", color=col_nll, lw=2.0),
        annotation_clip=False, zorder=5,
    )
    ax.annotate(
        "", xy=(1.0, y_clip), xytext=(1.0, y_clip - 2.0),
        arrowprops=dict(arrowstyle="-|>", color=col_nll, lw=2.0),
        annotation_clip=False, zorder=5,
    )

    # BS: constant 2.
    ax.plot(
        [0, 1], [2, 2],
        linewidth=2.0, color=col_bs_or_mse,
        label=r"$\pi_{\text{BS}}$",
    )

    # Acc: Dirac at 0.5.  Draw as an upward arrow (spike).
    ax.annotate(
        "", xy=(0.5, y_clip), xytext=(0.5, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=col_acc, lw=2.0),
        annotation_clip=False, zorder=5,
    )
    # Add a phantom line so it appears in the legend.
    ax.plot(
        [], [],
        linewidth=2.0, color=col_acc,
        label=r"$\pi_{\text{Acc}}$",
    )

    ax.set_xlabel(r"utility parameter $\theta=c$")
    ax.set_ylabel(r"$\pi(\theta)$")
    ax.set_title("Binary decision", fontsize=plt.rcParams["axes.titlesize"])

    # Legend below the panel, three columns in one row.
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        handlelength=1.6,
        columnspacing=1.2,
        fontsize=plt.rcParams["legend.fontsize"],
    )

    # ======================================================
    # RIGHT PANEL: selective prediction priors over lambda
    # ======================================================
    # x in [0, 1] is a symbolic axis for lambda; rightmost tick = infinity.
    ax = axes[1]

    # NLL: pi(lambda) = eps / lambda^2 for lambda > eps.
    # We treat the x-axis directly as lambda values in [0, 1].
    lam = np.linspace(1e-4, 1.0, n)
    y_nll_r = eps / (lam ** 2)
    y_nll_r_plot = np.minimum(y_nll_r, y_clip)
    # Only show the curve on its support lambda > eps; below eps the prior is 0.
    support = lam > eps
    inside = (y_nll_r <= y_clip) & support

    ax.plot(
        lam[inside], y_nll_r_plot[inside],
        linewidth=2.0, color=col_nll,
        label=r"$\pi_{\text{NLL}}$",
    )
    # Arrow at lambda = eps to indicate the curve continues upward (clipped).
    ax.annotate(
        "", xy=(eps, y_clip), xytext=(eps, y_clip - 2.0),
        arrowprops=dict(arrowstyle="-|>", color=col_nll, lw=2.0),
        annotation_clip=False, zorder=5,
    )

    # MSE: Dirac at lambda = infinity, drawn as a spike at the right edge.
    ax.annotate(
        "", xy=(1.0, y_clip), xytext=(1.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=col_bs_or_mse, lw=2.0),
        annotation_clip=False, zorder=5,
    )
    ax.plot(
        [], [],
        linewidth=2.0, color=col_bs_or_mse,
        label=r"$\pi_{\text{MSE}}$",
    )

    ax.set_xlabel(r"utility parameter $\theta=\lambda$")
    ax.set_title("Selective prediction",
                 fontsize=plt.rcParams["axes.titlesize"])

    # Legend below the panel, two columns in one row.
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        handlelength=1.6,
        columnspacing=1.2,
        fontsize=plt.rcParams["legend.fontsize"],
    )

    # ======================================================
    # Shared cleanup
    # ======================================================
    for ax in axes:
        ax.set_ylim(0, y_clip)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.grid(
            True, which="major", axis="both",
            linestyle=":", linewidth=0.6, alpha=0.5,
        )

    # Left panel: small left extension so c=0 doesn't sit on the y-axis.
    axes[0].set_xlim(-0.03, 1.0)
    axes[0].set_xticks(np.linspace(0, 1, 6))

    # Right panel: keep the original tight x-limits.
    axes[1].set_xlim(-0.01, 1.0)
    right_ticks = np.linspace(0, 1, 6)
    axes[1].set_xticks(right_ticks)
    right_labels = [f"{t:g}" for t in right_ticks[:-1]] + [r"$\infty$"]
    axes[1].set_xticklabels(right_labels)

    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    out_path = fig_dir / "pathological_priors.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


plot_pathological_priors(
    eps=0.05,
    y_clip=12.0,
    n=4000,
)
