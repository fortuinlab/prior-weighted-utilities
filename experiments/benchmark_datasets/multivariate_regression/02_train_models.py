import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal
from ngboost.scores import MLE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

# Multivariate regression experiment suite (Appendix H):
# five models, two deep, all producing a full-covariance Gaussian report.
ALL_MODELS = ["LinReg", "RF", "NGB", "MLP", "DeepEnsemble"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Heteroscedastic MLP with Cholesky-parametrized covariance
# ---------------------------------------------------------------------------
class HeteroscedasticMultivariateMLP(nn.Module):
    """Two-layer MLP that outputs μ ∈ R^D and a Cholesky factor L ∈ R^{D×D}
    with positive diagonals, so Σ = L Lᵀ + jitter·I is PD.
    """

    def __init__(self, d_in: int, d_out: int, hidden=(128, 128), jitter: float = 1e-6):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, d_out)
        # Cholesky factor: D diagonal (passed through softplus) + D(D-1)/2 off-diagonals
        self.d_out = d_out
        self.n_diag = d_out
        self.n_off = d_out * (d_out - 1) // 2
        self.L_head = nn.Linear(last, self.n_diag + self.n_off)
        self.softplus = nn.Softplus()
        self.jitter = jitter

        # Precompute lower-triangular index masks (excluding diagonal) once
        tril_rows, tril_cols = torch.tril_indices(d_out, d_out, offset=-1)
        self.register_buffer("tril_rows", tril_rows, persistent=False)
        self.register_buffer("tril_cols", tril_cols, persistent=False)

    def _assemble_L(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (B, n_diag + n_off) -> L: (B, D, D) lower triangular, pos diag."""
        B = raw.size(0)
        diag = self.softplus(raw[:, : self.n_diag]) + 1e-6
        off = raw[:, self.n_diag :]
        L = torch.zeros(B, self.d_out, self.d_out, device=raw.device, dtype=raw.dtype)
        # Diagonal
        idx = torch.arange(self.d_out, device=raw.device)
        L[:, idx, idx] = diag
        # Strict lower-triangular off-diagonals
        if self.n_off > 0:
            L[:, self.tril_rows, self.tril_cols] = off
        return L

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = self.mu_head(h)
        L = self._assemble_L(self.L_head(h))
        return mu, L

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multivariate Gaussian NLL (per-sample, averaged over batch).

        With Σ = L Lᵀ + jitter·I, and using L's positive diagonal,
        log det Σ ≈ 2·sum(log diag(L))  (ignoring the tiny jitter contribution).
        """
        mu, L = self.forward(x)
        resid = (y - mu).unsqueeze(-1)  # (B, D, 1)
        # Solve L z = resid  => z = L^{-1} resid; then resid^T Σ^{-1} resid ≈ z^T z
        # (jitter negligible for the quadratic term at our scales)
        z = torch.linalg.solve_triangular(L, resid, upper=False)  # (B, D, 1)
        quad = (z.squeeze(-1) ** 2).sum(dim=-1)  # (B,)
        # log det Σ = 2 sum log diag(L)  (L is lower-triangular with positive diag)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        return (logdet + quad).mean()


# ---------------------------------------------------------------------------
# Helper: diagonal-covariance wrapper for per-dim univariate models
# ---------------------------------------------------------------------------
def _diag_sigma(stds: np.ndarray) -> np.ndarray:
    """stds: (n, D) -> (n, D, D) diagonal covariance matrices."""
    n, D = stds.shape
    Sigma = np.zeros((n, D, D), dtype=np.float64)
    idx = np.arange(D)
    Sigma[:, idx, idx] = stds**2
    return Sigma


# ---------------------------------------------------------------------------
# Individual model functions — all return (mu: (n, D), Sigma: (n, D, D))
# ---------------------------------------------------------------------------
def lin_reg(X_tr, y_tr, X_te):
    """Per-dimension OLS with a constant predictive variance per output
    dimension (diagonal Σ). The predictive variance is the residual variance
    sigma_d^2 = RSS_d / (n - p), which is the standard Gaussian-noise OLS
    estimate. We intentionally drop the (1 + x^T (X^T X)^{-1} x) leverage
    term used by the univariate script — for fold-level variation and
    categorical one-hots, the Gram matrix can be ill-conditioned and SVD
    can fail; the constant-variance version is numerically stable and
    still gives a well-defined Gaussian report."""
    D = y_tr.shape[1]
    n_te = X_te.shape[0]
    mus = np.zeros((n_te, D), dtype=np.float64)
    stds = np.zeros_like(mus)

    n_tr, p = X_tr.shape
    df = max(1, n_tr - (p + 1))  # +1 for the intercept

    for d in range(D):
        linreg = LinearRegression()
        linreg.fit(X_tr, y_tr[:, d])
        mus[:, d] = linreg.predict(X_te)

        y_tr_pred = linreg.predict(X_tr)
        rss = float(np.sum((y_tr[:, d] - y_tr_pred) ** 2))
        sigma2 = rss / df
        stds[:, d] = np.sqrt(max(sigma2, 1e-12))

    return mus, _diag_sigma(stds)


def ran_for(X_tr, y_tr, X_te, seed):
    """Per-dimension Random Forest (diagonal Σ)."""
    D = y_tr.shape[1]
    mus = np.zeros((X_te.shape[0], D), dtype=np.float64)
    stds = np.zeros_like(mus)

    for d in range(D):
        rf = RandomForestRegressor(
            n_estimators=200, oob_score=True, bootstrap=True,
            random_state=seed + d, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr[:, d])
        mus[:, d] = rf.predict(X_te)

        tree_preds = np.stack([t.predict(X_te) for t in rf.estimators_], axis=1)
        sd_ep = tree_preds.std(axis=1, ddof=1)
        mask = np.isfinite(rf.oob_prediction_)
        resid = y_tr[mask, d] - rf.oob_prediction_[mask]
        sd_al = float(np.sqrt(np.mean(resid**2))) if resid.size else 0.0
        stds[:, d] = np.sqrt(sd_ep**2 + sd_al**2)

    return mus, _diag_sigma(stds)


def nat_gra_boo(X_tr, y_tr, X_te, seed):
    """NGBoost with native MultivariateNormal — full-covariance output."""
    n_samples = X_tr.shape[0]
    n_est = 500 if n_samples < 1000 else 1000
    D = y_tr.shape[1]

    ngb = NGBRegressor(
        Dist=MultivariateNormal(D),
        Score=MLE,
        n_estimators=n_est,
        learning_rate=0.03,
        col_sample=0.8,
        random_state=seed,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)
    dist = ngb.pred_dist(X_te)

    # NGBoost's MultivariateNormal exposes `mean()` as a method and `cov`
    # as a property. Shapes: mu (n, D), Sigma (n, D, D).
    mu = np.asarray(dist.mean()).astype(np.float64)
    Sigma = np.asarray(dist.cov).astype(np.float64)
    return mu, Sigma


def het_mlp(X_tr, y_tr, X_te, seed, device: torch.device):
    """Heteroscedastic MLP with Cholesky-parametrized full covariance."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=seed
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    d_in = X_tr.shape[1]
    d_out = y_tr.shape[1]
    model = HeteroscedasticMultivariateMLP(d_in, d_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_samples = X_tr.shape[0]
    batch_size = min(512, max(32, n_samples // 10))
    max_epochs = 3000 if n_samples < 1000 else 5000
    patience, bad = 20, 0
    best_val = float("inf")
    best_state = None
    loss_evo = []

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(X_tr_t.size(0), device=device)
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = model.loss(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_evo.append(loss.detach().item())

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(X_val_t, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mu, L = model(X_te_t)
        Sigma = L @ L.transpose(-1, -2)
    mu = mu.cpu().numpy().astype(np.float64)
    Sigma = Sigma.cpu().numpy().astype(np.float64)
    return mu, Sigma, loss_evo


def deep_ensemble(X_tr, y_tr, X_te, seed, device: torch.device, n_members: int = 5):
    """Ensemble of heteroscedastic multivariate MLPs. The combined predictive
    distribution is a mixture of Gaussians; we report its mean and covariance.

    For a mixture 1/M Σ N(μ_m, Σ_m):
      μ = (1/M) Σ μ_m
      Σ = (1/M) Σ Σ_m + (1/M) Σ (μ_m - μ)(μ_m - μ)^T
    """
    all_mus, all_sigmas = [], []
    for m in range(n_members):
        member_seed = seed * 100 + m

        X_tr_m, X_val_m, y_tr_m, y_val_m = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=member_seed
        )

        d_in = X_tr_m.shape[1]
        d_out = y_tr_m.shape[1]
        model = HeteroscedasticMultivariateMLP(d_in, d_out).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_tr_t = torch.tensor(X_tr_m, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr_m, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_m, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_m, dtype=torch.float32, device=device)

        n_samples = X_tr_m.shape[0]
        batch_size = min(512, max(32, n_samples // 10))
        max_epochs = 3000 if n_samples < 1000 else 5000
        patience, bad = 20, 0
        best_val = float("inf")
        best_state = None

        for _ in range(max_epochs):
            model.train()
            perm = torch.randperm(X_tr_t.size(0), device=device)
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                opt.zero_grad()
                loss = model.loss(xb, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = model.loss(X_val_t, y_val_t).item()

            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_m, L_m = model(X_te_t)
            Sigma_m = L_m @ L_m.transpose(-1, -2)
        all_mus.append(mu_m.cpu().numpy())
        all_sigmas.append(Sigma_m.cpu().numpy())

    mus = np.stack(all_mus)  # (M, n, D)
    sigmas = np.stack(all_sigmas)  # (M, n, D, D)

    mu_bar = mus.mean(axis=0)  # (n, D)
    # Within-component mean: (1/M) Σ Σ_m
    sigma_within = sigmas.mean(axis=0)  # (n, D, D)
    # Between-component: (1/M) Σ (μ_m - μ̄)(μ_m - μ̄)^T
    diffs = mus - mu_bar[None]  # (M, n, D)
    sigma_between = np.einsum("mni,mnj->nij", diffs, diffs) / mus.shape[0]
    Sigma = sigma_within + sigma_between

    return mu_bar.astype(np.float64), Sigma.astype(np.float64)


# ---------------------------------------------------------------------------
# y-space inverse transform for (mu, Sigma)
# ---------------------------------------------------------------------------
def _inverse_transform(mu_s, Sigma_s, y_scaler: StandardScaler):
    """Invert y-standardization for a multivariate Gaussian report.

    If ỹ = (y - m) / s (elementwise), then y = s ⊙ ỹ + m, so
      μ_y = s ⊙ μ_{ỹ} + m
      Σ_y = S Σ_{ỹ} S      with S = diag(s)
    """
    scale = y_scaler.scale_  # (D,)
    mean = y_scaler.mean_
    mu = mu_s * scale + mean
    S = np.diag(scale)
    Sigma = S @ Sigma_s @ S  # broadcasts over the batch dim
    return mu, Sigma


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------
def _fit_single_model(
    model_name: str,
    X_tr, y_tr, X_te,
    X_tr_scaled, X_te_scaled,
    X_tr_gp, X_te_gp, y_tr_gp,
    y_scaler,
    seed, device, out_path,
):
    """Dispatch to the right training function.

    Returns (mu, Sigma) in original y-space, or (mu, Sigma, loss_evo) for MLP.
    """
    if model_name == "LinReg":
        return lin_reg(X_tr_scaled, y_tr, X_te_scaled)

    elif model_name == "RF":
        return ran_for(X_tr, y_tr, X_te, seed)

    elif model_name == "NGB":
        return nat_gra_boo(X_tr, y_tr, X_te, seed)

    elif model_name == "MLP":
        mu_s, Sigma_s, loss_evo = het_mlp(X_tr_gp, y_tr_gp, X_te_gp, seed, device)
        mu, Sigma = _inverse_transform(mu_s, Sigma_s, y_scaler)
        return mu, Sigma, loss_evo

    elif model_name == "DeepEnsemble":
        mu_s, Sigma_s = deep_ensemble(X_tr_gp, y_tr_gp, X_te_gp, seed, device)
        mu, Sigma = _inverse_transform(mu_s, Sigma_s, y_scaler)
        return mu, Sigma

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    dataset: Optional[str] = None,
    repeat: Optional[int] = None,
    fold: Optional[int] = None,
):
    seed = 10 * repeat + fold

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")

    data_path = Path(
        str(root_dir)
        + f"/data/benchmark_datasets/multivariate_regression/{dataset}"
    )
    # y is expected to be a parquet with D columns (one per output dim)
    X = pd.read_parquet(data_path / "X.parquet")
    y = pd.read_parquet(data_path / "y.parquet")

    with open(data_path / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    split = splits[repeat][fold - 1]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    X_tr_df = X.iloc[train_idx].copy()
    y_tr_df = y.iloc[train_idx].copy()
    X_te_df = X.iloc[test_idx].copy()
    y_te_df = y.iloc[test_idx].copy()

    X_tr = X_tr_df.to_numpy()
    X_te = X_te_df.to_numpy()
    y_tr = y_tr_df.to_numpy()  # (n_tr, D)
    y_te = y_te_df.to_numpy()  # (n_te, D)

    # Drop columns that are constant or entirely NaN in the training fold.
    # Constant columns make StandardScaler divide by zero (producing NaN and
    # breaking downstream models); all-NaN columns also yield NaN after
    # scaling. Either can slip through for rare ordinal-encoded categories
    # (flare) or for sensor columns that are missing everywhere (air's
    # NMHC(GT), which is almost entirely −200 in the raw data).
    col_std = np.nanstd(X_tr, axis=0, ddof=0)
    keep_mask = np.isfinite(col_std) & (col_std > 0)
    if not keep_mask.all():
        dropped = int((~keep_mask).sum())
        print(f"  Dropping {dropped} constant/all-NaN feature column(s) for this fold.")
        X_tr = X_tr[:, keep_mask]
        X_te = X_te[:, keep_mask]
    if X_tr.shape[1] == 0:
        raise RuntimeError(
            f"All feature columns were dropped for {dataset} "
            f"(repeat={repeat}, fold={fold}). Check preprocessing."
        )

    # Standardize X once (used by LinReg directly; deep models use X_tr_gp below)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Standardize X and y for MLP / DeepEnsemble
    x_scaler = StandardScaler()
    X_tr_gp = x_scaler.fit_transform(X_tr)
    X_te_gp = x_scaler.transform(X_te)

    y_scaler = StandardScaler()
    y_tr_gp = y_scaler.fit_transform(y_tr)

    out_path = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multivariate_regression/predictions/{dataset}/repeat_{repeat:04d}/fold_{fold:02d}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # ------ full run: train all models + predict ------
    print(f"[{dataset}] repeat={repeat} fold={fold} seed={seed}")

    results = {}
    for model_name in ALL_MODELS:
        print(f"Fitting {model_name}...")
        result = _fit_single_model(
            model_name,
            X_tr, y_tr, X_te,
            X_tr_scaled, X_te_scaled,
            X_tr_gp, X_te_gp, y_tr_gp,
            y_scaler,
            seed, device, out_path,
        )
        if isinstance(result, tuple) and len(result) == 3:
            mu, Sigma, _loss_evo = result
        else:
            mu, Sigma = result

        results[f"{model_name}_mu"] = mu.astype(np.float64)
        results[f"{model_name}_Sigma"] = Sigma.astype(np.float64)

    np.savez(out_path / "predictions.npz", **results)

    # Metadata: one column per output dim for y_true
    D = y_te.shape[1]
    meta = {
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "test_idx": test_idx,
    }
    for d in range(D):
        meta[f"y_true_{d}"] = y_te[:, d]
    pd.DataFrame(meta).to_csv(out_path / "meta.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="energy")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
