import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import beta, pearsonr

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.utils import (  # noqa: E402
    compute_metric_correlation,
    top1_agreement,
    topk_agreement,
)


# ---------------------------------------------------------------------------
# Multivariate metrics
# ---------------------------------------------------------------------------
def _cholesky_with_jitter(
    Sigma_i: np.ndarray, base_jitter: float = 1e-6, max_tries: int = 6
) -> np.ndarray:
    """Try Cholesky on Sigma_i with progressively larger jitter until it
    succeeds. Scales the jitter by the trace of Sigma to make it unitful.

    Raises LinAlgError if still failing at max_tries (extreme rare case).
    """
    D = Sigma_i.shape[0]
    # scale the jitter by the magnitude of the matrix so it does something
    # useful even when Sigma has tiny entries
    trace_scale = max(float(np.trace(Sigma_i)) / D, 1.0)
    jitter = base_jitter * trace_scale

    eye = np.eye(D)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(Sigma_i + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Last resort: force symmetric PSD by eigen-clipping
    w, V = np.linalg.eigh(0.5 * (Sigma_i + Sigma_i.T))
    w_clipped = np.clip(w, jitter, None)
    Sigma_psd = (V * w_clipped) @ V.T
    return np.linalg.cholesky(Sigma_psd)


def multivariate_gaussian_nll(
    y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, jitter: float = 1e-6
) -> float:
    """Mean negative log-likelihood of y_i under N(mu_i, Sigma_i).

    Uses Cholesky for numerical stability with adaptive jitter for
    near-singular Sigma (e.g., diagonal covariance models on sparse
    targets where a dimension may collapse to zero variance).
    """
    n, D = y.shape

    nll_i = np.empty(n)
    const = 0.5 * D * np.log(2.0 * np.pi)
    for i in range(n):
        L = _cholesky_with_jitter(Sigma[i], base_jitter=jitter)
        # log|Sigma| = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.log(np.diag(L)).sum()
        # z = L^{-1} (y - mu);  (y-mu)^T Sigma^{-1} (y-mu) = ||z||^2
        z = solve_triangular(L, y[i] - mu[i], lower=True)
        nll_i[i] = const + 0.5 * log_det + 0.5 * float(z @ z)
    return float(nll_i.mean())


def mean_squared_error_multivariate(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean over instances of ||y_i - mu_i||^2 (sum over output dims)."""
    return float(((y - mu) ** 2).sum(axis=1).mean())


def energy_score(
    y: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_samples: int = 100,
    seed: int = 0,
) -> float:
    """Monte Carlo estimator of the energy score for Gaussian predictive
    distributions.

    ES(F, y) = E||Y - y|| - 0.5 * E||Y - Y'||
    with Y, Y' iid from F = N(mu, Sigma).
    """
    rng = np.random.default_rng(seed)
    n, D = y.shape

    # Per-instance Cholesky with adaptive jitter for near-singular Sigma.
    L = np.empty_like(Sigma)
    for i in range(n):
        L[i] = _cholesky_with_jitter(Sigma[i])

    # Draw n_samples from each N(mu_i, Sigma_i).  Shape: (n_samples, n, D)
    eps1 = rng.standard_normal((n_samples, n, D))
    eps2 = rng.standard_normal((n_samples, n, D))
    samples_1 = mu[None] + np.einsum("ijk,sik->sij", L, eps1)
    samples_2 = mu[None] + np.einsum("ijk,sik->sij", L, eps2)

    # E||Y - y||
    term1 = np.linalg.norm(samples_1 - y[None], axis=-1).mean(axis=0)  # (n,)
    # 0.5 * E||Y - Y'||
    term2 = 0.5 * np.linalg.norm(samples_1 - samples_2, axis=-1).mean(axis=0)  # (n,)

    return float((term1 - term2).mean())


# ---------------------------------------------------------------------------
# Multivariate selective-prediction utility + PWU
# ---------------------------------------------------------------------------
def selective_prediction_utility_multivariate(
    mu: np.ndarray, Sigma: np.ndarray, y: np.ndarray, lambda_factor: float,
    y_var_per_dim: Optional[np.ndarray] = None,
):
    """Coordinate-wise selective prediction utility, averaged over output
    dimensions.

    For each dimension d, the per-instance decision is the univariate
    selective prediction rule applied to the marginal (mu_{i,d}, Sigma_{i,dd}):
    trade (pay standardized squared error) if Sigma_{i,dd} <= lambda * Var(y_d);
    otherwise abstain (pay lambda).

    Costs are standardized by dividing by Var(y_d) so each dimension
    contributes on the same unit scale — otherwise a single high-variance
    dimension dominates the averaged utility on multi-output datasets
    where output dimensions differ in scale.

    This is equivalent to solving the selective prediction problem on the
    standardized targets y_d / sqrt(Var(y_d)). The per-dim decision rule
    (trade iff Sigma_{i,dd} / Var(y_d) <= lambda) is unchanged — only
    the payoff is rescaled, which does not affect the Bayes act.

    lambda_factor is the dimensionless lambda value sampled from the Beta
    prior; it is multiplied internally by each dimension's empirical
    variance y_var_per_dim (computed from y if not provided).
    """
    n, D = y.shape
    if y_var_per_dim is None:
        y_var_per_dim = y.var(axis=0, ddof=0)  # (D,)

    # Floor variances to avoid division-by-zero on a degenerate dimension
    # (e.g., a training fold where one target dim is constant).
    eps = 1e-12
    y_var_safe = np.maximum(y_var_per_dim, eps)

    # Marginal predictive variances: (n, D)
    sigma_dd = np.diagonal(Sigma, axis1=1, axis2=2)  # (n, D)

    # Per-dimension abstention threshold (unstandardized, same units as
    # sigma_dd): (D,)
    abst_cost_per_dim = lambda_factor * y_var_safe

    # (n, D) trade mask
    trade = sigma_dd <= abst_cost_per_dim[None, :]

    # Standardized costs: dividing squared error by Var(y_d) puts every
    # dimension on the same scale. After division, the abstention cost in
    # standardized space is exactly lambda (a scalar), not lambda * Var(y_d).
    cost_mse_standardized = ((y - mu) ** 2) / y_var_safe[None, :]  # (n, D)
    cost = np.where(trade, cost_mse_standardized, lambda_factor)

    # Average over instances and dimensions.
    return -float(cost.mean()), int((~trade).sum())


def selective_prediction_pwu_multivariate(
    mu: np.ndarray,
    Sigma: np.ndarray,
    y: np.ndarray,
    alpha_param: float,
    beta_param: float,
    n_mc: int = 10000,
) -> float:
    """Monte Carlo estimate of the coordinate-wise selective-prediction PWU.

    lambda_factor ~ Beta(a, b), scaled per-dimension by each dim's
    empirical variance inside the utility.
    """
    rng = np.random.default_rng(42)

    y_var_per_dim = y.var(axis=0, ddof=0)  # (D,)

    lambda_factor_vec = rng.beta(alpha_param, beta_param, size=n_mc)

    costs = np.empty(n_mc)
    for i in range(n_mc):
        costs[i] = -selective_prediction_utility_multivariate(
            mu, Sigma, y,
            lambda_factor=float(lambda_factor_vec[i]),
            y_var_per_dim=y_var_per_dim,
        )[0]
    return float(costs.mean())


# ---------------------------------------------------------------------------
# Per-model metric computation
# ---------------------------------------------------------------------------
def _compute_metrics_for_one_model(
    dataset: str,
    model: str,
    repeat: int,
    fold: int,
    y_true: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    alpha_param_lambda: float,
    beta_param_lambda: float,
    lambda_factor_vec: np.ndarray,
    lambda_factor_vec_name: np.ndarray,
) -> Dict[str, float]:
    PWU_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multivariate_regression/results/{dataset}/pwus/{model}"
    )
    PWU_PATH.mkdir(parents=True, exist_ok=True)

    nll = multivariate_gaussian_nll(y_true, mu, Sigma)
    mse = mean_squared_error_multivariate(y_true, mu)
    es = energy_score(y_true, mu, Sigma, n_samples=100, seed=0)

    pwu_fp = PWU_PATH / f"repeat_{repeat}_fold_{fold}_sp_pwu.npy"
    if pwu_fp.exists():
        sp_p = float(np.load(pwu_fp))
    else:
        sp_p = selective_prediction_pwu_multivariate(
            mu, Sigma, y_true, alpha_param_lambda, beta_param_lambda
        )
        np.save(pwu_fp, sp_p)

    out = {
        "NLL ↓": float(nll),
        "MSE ↓": float(mse),
        "ES ↓": float(es),
        "SP-PWU ↓": float(sp_p),
    }

    # Sampled utilities — one per lambda-factor.  Per-dim variance scaling
    # is now handled inside the utility function itself.
    y_var_per_dim = y_true.var(axis=0, ddof=0)  # (D,)
    for i in range(len(lambda_factor_vec)):
        u, _ = selective_prediction_utility_multivariate(
            mu, Sigma, y_true,
            lambda_factor=float(lambda_factor_vec[i]),
            y_var_per_dim=y_var_per_dim,
        )
        out[f"u_lambda, lambda={lambda_factor_vec_name[i]:.2f} ↑"] = float(u)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(dataset: Optional[str] = "energy"):
    PRED_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multivariate_regression/predictions/{dataset}"
    )

    # configure utilities — same Beta(2, 10) prior on lambda-factor as univariate
    alpha_param_lambda = 2.0
    beta_param_lambda = 10.0
    n_lambda_samples = 5

    rng = np.random.default_rng(0)
    lambda_factor_vec = beta.rvs(
        alpha_param_lambda, beta_param_lambda,
        size=n_lambda_samples, random_state=rng,
    )
    lambda_factor_vec_name = np.round(lambda_factor_vec, 2)

    RESULT_PATH = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multivariate_regression/results/{dataset}"
    )
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(PRED_PATH.glob("repeat_*/fold_*/predictions.npz"))
    if len(pred_files) == 0:
        print(f"ERROR: No predictions.npz files found under {PRED_PATH}")
        sys.exit(1)

    rows = []
    for fp in pred_files:
        repeat = int(fp.parents[1].name.replace("repeat_", ""))
        fold = int(fp.parents[0].name.replace("fold_", ""))

        meta_fp = fp.with_name("meta.csv")
        if not meta_fp.exists():
            raise FileNotFoundError(f"Missing meta.csv next to {fp}")
        meta = pd.read_csv(meta_fp)
        y_cols = sorted(
            [c for c in meta.columns if c.startswith("y_true_")],
            key=lambda c: int(c.replace("y_true_", "")),
        )
        y_true = meta[y_cols].to_numpy().astype(float)

        with np.load(fp) as npz:
            model_names = sorted({
                k[: -len("_mu")] for k in npz.keys() if k.endswith("_mu")
            })
            for model in model_names:
                mu_key, Sigma_key = f"{model}_mu", f"{model}_Sigma"
                if mu_key not in npz or Sigma_key not in npz:
                    print(
                        f"WARNING: {fp} missing {mu_key} or {Sigma_key}, skipping."
                    )
                    continue
                mu = npz[mu_key].astype(float)
                Sigma = npz[Sigma_key].astype(float)

                if mu.shape[0] != y_true.shape[0]:
                    raise ValueError(
                        f"Row mismatch in {fp}:{model}: mu {mu.shape} vs "
                        f"y_true {y_true.shape}"
                    )

                metrics = _compute_metrics_for_one_model(
                    dataset=dataset,
                    model=model,
                    repeat=repeat,
                    fold=fold,
                    y_true=y_true,
                    mu=mu,
                    Sigma=Sigma,
                    alpha_param_lambda=alpha_param_lambda,
                    beta_param_lambda=beta_param_lambda,
                    lambda_factor_vec=lambda_factor_vec,
                    lambda_factor_vec_name=lambda_factor_vec_name,
                )
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model,
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        print(f"ERROR: No prediction files loadable in {PRED_PATH}")
        sys.exit(1)

    # ---- Aggregate over folds ----
    metric_cols = [
        c for c in metrics_df.columns if c not in {"repeat", "fold", "model"}
    ]
    metric_values_by_repeat = metrics_df.groupby(["repeat", "model"], as_index=False)[
        metric_cols
    ].mean()
    metric_values_by_repeat.to_csv(
        RESULT_PATH / "metric_values_by_repeat.csv", index=False
    )

    # ---- Ranking + alignment per repeat ----
    smaller_is_better = {
        "NLL ↓": True,
        "MSE ↓": True,
        "ES ↓": True,
        "SP-PWU ↓": True,
    }
    for i in range(len(lambda_factor_vec)):
        smaller_is_better[
            f"u_lambda, lambda={lambda_factor_vec_name[i]:.2f} ↑"
        ] = False

    def clean_metric_name(name: str) -> str:
        return name.replace("↑", "").replace("↓", "").strip()

    base_metrics = [
        clean_metric_name(m) for m in ["NLL ↓", "MSE ↓", "ES ↓", "SP-PWU ↓"]
    ]
    utilities = [
        clean_metric_name(f"u_lambda, lambda={c:.2f} ↑")
        for c in lambda_factor_vec_name
    ]

    ranking_rows = []
    tau_rows = []

    repeats = sorted(metric_values_by_repeat["repeat"].unique().tolist())
    for r in repeats:
        sub = metric_values_by_repeat[
            metric_values_by_repeat["repeat"] == r
        ].set_index("model")

        rank_dict = {}
        for metric, asc in smaller_is_better.items():
            order = sub[metric].sort_values(ascending=asc).index.tolist()
            rank_dict[clean_metric_name(metric)] = order

        row = {"repeat": r}
        for metric, order in rank_dict.items():
            row[metric] = ">".join(order)
        ranking_rows.append(row)

        for m in base_metrics:
            for u in utilities:
                tau = compute_metric_correlation(
                    ranking_m=rank_dict[m], ranking_u=rank_dict[u]
                )
                t1 = top1_agreement(
                    ranking_m=rank_dict[m], ranking_u=rank_dict[u]
                )
                t3 = topk_agreement(
                    ranking_m=rank_dict[m], ranking_u=rank_dict[u], k=3
                )
                tau_rows.append(
                    {
                        "repeat": r,
                        "utility": u,
                        "metric": m,
                        "kendall_tau": tau,
                        "top1": t1,
                        "top3": t3,
                    }
                )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(RESULT_PATH / "model_rankings_by_repeat.csv", index=False)
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(RESULT_PATH / "kendall_by_repeat.csv", index=False)

    # ---- Summary over repeats ----
    def summarize_alignment(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        g = df.groupby(["utility", "metric"])[value_col]
        return g.agg(
            mean="mean",
            median="median",
            q05=lambda x: np.quantile(x, 0.05),
            q95=lambda x: np.quantile(x, 0.95),
        ).reset_index()

    for value_col in ["kendall_tau", "top1", "top3"]:
        summary = summarize_alignment(tau_df, value_col)
        summary.to_csv(
            RESULT_PATH / f"{value_col}_summary_over_repeats.csv", index=False
        )

    # ---- Pearson on raw metric/utility values ----
    pearson_rows = []
    clean_to_orig = {clean_metric_name(m): m for m in smaller_is_better.keys()}

    for r in repeats:
        sub = metric_values_by_repeat[metric_values_by_repeat["repeat"] == r]
        for m in base_metrics:
            sign = -1.0 if smaller_is_better[clean_to_orig[m]] else 1.0
            m_values = sign * sub[clean_to_orig[m]].to_numpy()
            for u in utilities:
                u_values = sub[clean_to_orig[u]].to_numpy()
                corr, pval = pearsonr(m_values, u_values)
                pearson_rows.append(
                    {
                        "repeat": r,
                        "utility": u,
                        "metric": m,
                        "pearson_r": corr,
                        "pearson_pval": pval,
                    }
                )

    pearson_df = pd.DataFrame(pearson_rows)
    pearson_df.to_csv(RESULT_PATH / "pearson_by_repeat.csv", index=False)
    pearson_summary = summarize_alignment(pearson_df, "pearson_r")
    pearson_summary.to_csv(
        RESULT_PATH / "pearson_summary_over_repeats.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate regression evaluation.")
    parser.add_argument(
        "--dataset", type=str, default="energy", help="Which dataset to use."
    )
    args = parser.parse_args()
    main(**vars(args))
