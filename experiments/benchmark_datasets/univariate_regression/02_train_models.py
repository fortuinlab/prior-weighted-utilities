import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tabpfn import TabPFNRegressor

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

# All model names, used for CLI validation and the full-run column order
ALL_MODELS = [
    "LinReg", "RF", "NGB", "GP", "MLP", "TabPFN",
    "FTTransformer", "SAINT", "ResNetMLP", "DeepEnsemble",
]


def get_device() -> torch.device:
    """Pick CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Existing model components
# ---------------------------------------------------------------------------
class HeteroscedasticMLP(torch.nn.Module):
    def __init__(self, d_in, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            last = h
        self.backbone = torch.nn.Sequential(*layers)
        self.mu_head = torch.nn.Linear(last, 1)
        self.rho_head = torch.nn.Linear(last, 1)  # rho -> sigma via softplus
        self.softplus = torch.nn.Softplus()  # ensures σ > 0

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma

    def loss(self, x, y):
        # neg log-likelihood (Gaussian likelihood)
        # −log N(y; μ, σ²) up to constant: 2 log σ + ((y−μ)/σ)²
        mu, sigma = self(x)
        return (2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2).mean()


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, ard_dims):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------------------------------------------------------------------------
# FT-Transformer  (Gorishniy et al., 2021) — heteroscedastic regression
# ---------------------------------------------------------------------------
class _FTTokenizer(nn.Module):
    """Tokenize numerical features: each feature gets a learned linear embedding."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.bias, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight[None] + self.bias[None]


class FTTransformerRegModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = _FTTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mu_head = nn.Linear(d_token, 1)
        self.rho_head = nn.Linear(d_token, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        tokens = self.tokenizer(x)  # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, D)
        out = self.transformer(tokens)  # (B, F+1, D)
        h = out[:, 0]  # CLS token output
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


# ---------------------------------------------------------------------------
# SAINT  (Somepalli et al., 2021) — heteroscedastic regression
# ---------------------------------------------------------------------------
class _IntersampleAttentionLayer(nn.Module):
    """Transpose batch & feature dims so standard MHA attends across samples."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — we want attention across B for each token position.
        B, T, D = x.shape
        xt = self.norm(x).permute(1, 0, 2).contiguous()  # (T, B, D)
        # MHA with batch_first=True: treats T as batch, B as seq
        attn_out, _ = self.attn(xt, xt, xt)  # (T, B, D)
        out = attn_out.permute(1, 0, 2)  # (B, T, D)
        return x + self.dropout(out)


class SAINTRegModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = _FTTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": nn.TransformerEncoderLayer(
                            d_model=d_token,
                            nhead=n_heads,
                            dim_feedforward=d_token * 4,
                            dropout=dropout,
                            activation="gelu",
                            batch_first=True,
                        ),
                        "inter_attn": _IntersampleAttentionLayer(
                            d_token, n_heads, dropout
                        ),
                    }
                )
            )
        self.norm = nn.LayerNorm(d_token)
        self.mu_head = nn.Linear(d_token, 1)
        self.rho_head = nn.Linear(d_token, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            tokens = layer["self_attn"](tokens)
            tokens = layer["inter_attn"](tokens)
        h = self.norm(tokens[:, 0])
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


class ResNetMLPRegModel(nn.Module):
    """ResNet-like MLP (Gorishniy et al., 2021) — heteroscedastic regression."""

    def __init__(self, n_features: int, d_hidden: int = 128, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                )
            )
        self.norm = nn.BatchNorm1d(d_hidden)
        self.mu_head = nn.Linear(d_hidden, 1)
        self.rho_head = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


# ---------------------------------------------------------------------------
# Gaussian NLL loss (shared by all new heteroscedastic models)
# ---------------------------------------------------------------------------
def _gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL: 2 log σ + ((y − μ) / σ)², averaged over batch."""
    return (2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2).mean()


# ---------------------------------------------------------------------------
# Generic PyTorch training loop for heteroscedastic regression models
# ---------------------------------------------------------------------------
def _train_torch_regressor(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 256,
    patience: int = 20,
) -> nn.Module:
    """Train a heteroscedastic PyTorch regression model with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_tr)
    perm = np.random.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)

    X_val, y_val = X_t[val_idx].to(device), y_t[val_idx].to(device)
    train_ds = TensorDataset(X_t[tr_idx], y_t[tr_idx])
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(tr_idx)),
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = _gaussian_nll(mu, sigma, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            mu_val, sigma_val = model(X_val)
            val_loss = _gaussian_nll(mu_val, sigma_val, y_val).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model


def _predict_torch_regressor(
    model: nn.Module, X_te: np.ndarray, device: torch.device, batch_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mu and sigma from a trained heteroscedastic model."""
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    mus, sigmas = [], []
    for i in range(0, len(X_te_t), batch_size):
        xb = X_te_t[i : i + batch_size].to(device)
        with torch.no_grad():
            mu, sigma = model(xb)
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
    return (
        np.concatenate(mus).astype(np.float64).ravel(),
        np.concatenate(sigmas).astype(np.float64).ravel(),
    )


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------
def lin_reg(X_tr, y_tr, X_te):
    linreg = LinearRegression()
    linreg.fit(X_tr, y_tr)
    y_pred = linreg.predict(X_te)

    # Add intercept column to scaled features
    Xb_train = np.c_[np.ones(len(X_tr)), X_tr]
    Xb_test = np.c_[np.ones(len(X_te)), X_te]

    # Residual variance estimate
    y_tr_pred = linreg.predict(X_tr)
    rss = np.sum((y_tr - y_tr_pred) ** 2)
    df = len(y_tr) - Xb_train.shape[1]  # degrees of freedom
    if df <= 0:
        df = max(1, df)
    sigma2 = float(rss / df)

    # (X^T X)^(-1)
    XtX_inv = np.linalg.pinv(Xb_train.T @ Xb_train)

    # Predictive variance for each test point: σ² * (1 + xᵀ (XᵀX)^(-1) x)
    pred_var = sigma2 * (1.0 + np.einsum("ij,jk,ik->i", Xb_test, XtX_inv, Xb_test))
    y_std = np.sqrt(pred_var)

    return y_pred, y_std


def ran_for(X_tr, y_tr, X_te, seed):
    rf = RandomForestRegressor(
        n_estimators=200, oob_score=True, bootstrap=True, random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)

    tree_preds = np.stack([t.predict(X_te) for t in rf.estimators_], axis=1)
    sd_epistemic = tree_preds.std(axis=1, ddof=1)
    mask = np.isfinite(rf.oob_prediction_)
    resid = y_tr[mask] - rf.oob_prediction_[mask]
    sigma_aleatoric = float(np.sqrt(np.mean(resid**2)))
    y_std = np.sqrt(sd_epistemic**2 + sigma_aleatoric**2)

    return y_pred, y_std


def nat_gra_boo(X_tr, y_tr, X_te, seed):
    n_samples = X_tr.shape[0]
    n_est = 500 if n_samples < 1000 else 1000

    ngb = NGBRegressor(
        Dist=Normal,
        Score=MLE,
        n_estimators=n_est,
        learning_rate=0.03,
        col_sample=0.8,
        random_state=seed,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    dist = ngb.pred_dist(X_te)
    return dist.loc, dist.scale


def gee_pee(X_tr, y_tr, X_te, seed, device: torch.device):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    n_tr = X_tr_t.shape[0]
    M = min(512, max(16, int(0.5 * np.sqrt(n_tr)) * 16))
    rs = np.random.RandomState(seed)
    inds = rs.choice(X_tr_t.shape[0], size=min(M, X_tr_t.shape[0]), replace=False)
    Z = X_tr_t[inds].clone().to(device)

    model = GPRegressionModel(Z, ard_dims=X_tr_t.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    batch_size = min(1024, max(32, n_tr // 4))
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=5e-3
    )

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr_t.shape[0])
    loss_evo = []
    epochs = 100
    print("Training SVGP...")
    with gpytorch.settings.cholesky_jitter(1e-4):
        for _ in tqdm(range(epochs), disable=False):
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(xb)
                loss = -mll(output, yb)
                loss_evo.append(loss.detach().item())
                loss.backward()
                optimizer.step()
    print("Done.")

    model.eval()
    likelihood.eval()
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        gpytorch.settings.cholesky_jitter(1e-4),
    ):
        pred_dist = likelihood(model(X_te_t))
        y_pred = pred_dist.mean.detach().cpu().numpy().astype(np.float64).ravel()
        y_std = pred_dist.stddev.detach().cpu().numpy().astype(np.float64).ravel()

    return y_pred, y_std, loss_evo


def het_mlp(X_tr, y_tr, X_te, seed, device: torch.device):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=seed
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    d_in = X_tr.shape[1]
    model = HeteroscedasticMLP(d_in).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_samples = X_tr.shape[0]
    batch_size = min(512, max(32, n_samples // 10))
    max_epochs = 3000 if n_samples < 1000 else 5000

    best_val = float("inf")
    best_state = None
    patience, bad = 20, 0

    loss_evo = []

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(X_tr_t.size(0), device=device)
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = model.loss(xb, yb)
            loss_evo.append(loss.detach().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(X_val_t, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        mu_s, sigma_s = model(X_te_t)
        mu_s = mu_s.cpu().numpy()
        sigma_s = sigma_s.cpu().numpy()

    return mu_s, sigma_s, loss_evo


def tab_pfn_reg(X_tr, y_tr, X_te, seed):
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in df.columns:
            col = df[c].tolist()
            first = next((v for v in col if v is not None and v is not pd.NA), None)
            if isinstance(first, (int, float, np.number)):
                out[c] = [float(v) if v is not None and v is not pd.NA else np.nan for v in col]
            else:
                out[c] = ["__nan__" if v is None or v is pd.NA else str(v) for v in col]
        return pd.DataFrame(out)

    X_tr = normalize_df(X_tr)
    X_te = normalize_df(X_te)
    y_tr_clean = np.asarray([float(v) for v in y_tr.tolist()], dtype=np.float64)

    if len(X_tr) > 10_000:
        rs = np.random.RandomState(seed)
        idx = rs.choice(len(X_tr), 10_000, replace=False)
        X_tr = X_tr.iloc[idx]
        y_tr_clean = y_tr_clean[idx]

    reg = TabPFNRegressor(device="cpu")
    reg.fit(X_tr, y_tr_clean)

    y_pred = reg.predict(X_te, output_type="mean")

    quantiles = reg.predict(X_te, output_type="quantiles", quantiles=[0.1587, 0.8413])
    y_std = (quantiles[1] - quantiles[0]) / 2.0
    y_std = np.clip(y_std, 1e-12, None)

    return y_pred, y_std


def ft_transformer_reg(X_tr, y_tr, X_te, seed, device):
    """FT-Transformer (Gorishniy et al., 2021) — heteroscedastic regression."""
    n_features = X_tr.shape[1]
    model = FTTransformerRegModel(
        n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1,
    )
    model = _train_torch_regressor(model, X_tr, y_tr, device, seed)
    return _predict_torch_regressor(model, X_te, device)


def saint_reg(X_tr, y_tr, X_te, seed, device):
    """SAINT (Somepalli et al., 2021) — heteroscedastic regression."""
    n_features = X_tr.shape[1]
    model = SAINTRegModel(
        n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1,
    )
    model = _train_torch_regressor(model, X_tr, y_tr, device, seed)
    return _predict_torch_regressor(model, X_te, device)


def resnet_mlp_reg(X_tr, y_tr, X_te, seed, device):
    """ResNet-MLP (Gorishniy et al., 2021) — heteroscedastic regression."""
    n_features = X_tr.shape[1]
    model = ResNetMLPRegModel(n_features=n_features, d_hidden=128, n_blocks=3, dropout=0.1)
    model = _train_torch_regressor(model, X_tr, y_tr, device, seed)
    return _predict_torch_regressor(model, X_te, device)


def deep_ensemble_reg(X_tr, y_tr, X_te, seed, device, n_members=5):
    """Deep Ensemble (Lakshminarayanan et al., 2017) of heteroscedastic MLPs."""
    all_mus, all_sigmas = [], []
    for m in range(n_members):
        member_seed = seed * 100 + m

        X_tr_m, X_val_m, y_tr_m, y_val_m = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=member_seed
        )

        d_in = X_tr_m.shape[1]
        model = HeteroscedasticMLP(d_in).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_tr_t = torch.tensor(X_tr_m, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr_m, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_m, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_m, dtype=torch.float32, device=device)

        n_samples = X_tr_m.shape[0]
        batch_size = min(512, max(32, n_samples // 10))
        max_epochs = 3000 if n_samples < 1000 else 5000

        best_val = float("inf")
        best_state = None
        patience, bad = 20, 0

        for _ in range(max_epochs):
            model.train()
            perm = torch.randperm(X_tr_t.size(0), device=device)
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                opt.zero_grad()
                loss = model.loss(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = model.loss(X_val_t, y_val_t).item()

            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_m, sigma_m = model(X_te_t)
        all_mus.append(mu_m.cpu().numpy())
        all_sigmas.append(sigma_m.cpu().numpy())

    # Mixture of Gaussians: combined mean and variance
    mus = np.stack(all_mus)        # (M, n_test)
    sigmas = np.stack(all_sigmas)  # (M, n_test)
    mu_bar = mus.mean(axis=0)
    # Var = E[σ²] + E[μ²] − (E[μ])²  (law of total variance)
    var_bar = (sigmas**2).mean(axis=0) + (mus**2).mean(axis=0) - mu_bar**2
    sigma_bar = np.sqrt(np.clip(var_bar, 1e-12, None))

    return mu_bar.ravel(), sigma_bar.ravel()


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------
def _fit_single_model(
    model_name: str,
    X_tr, y_tr, X_te,
    X_tr_df, y_tr_df, X_te_df,
    X_tr_scaled, X_te_scaled,
    X_tr_gp, X_te_gp, y_tr_gp,
    y_scaler, preprocess_gp_mlp,
    seed, device, out_path,
):
    """Dispatch to the right training function for *model_name*.

    Returns (y_pred, y_std) — or (y_pred, y_std, loss_evo) for GP/MLP.
    All predictions are in *original* y-space.
    """
    if model_name == "LinReg":
        return lin_reg(X_tr_scaled, y_tr, X_te_scaled)

    elif model_name == "RF":
        return ran_for(X_tr, y_tr, X_te, seed)

    elif model_name == "NGB":
        return nat_gra_boo(X_tr, y_tr, X_te, seed)

    elif model_name == "GP":
        y_pred_scld, y_std_scld, loss_evo = gee_pee(
            X_tr_gp, y_tr_gp, X_te_gp, seed, device
        )
        y_pred = y_scaler.inverse_transform(y_pred_scld.reshape(-1, 1)).ravel()
        y_std = y_std_scld * y_scaler.scale_[0]
        return y_pred, y_std, loss_evo

    elif model_name == "MLP":
        y_pred_scld, y_std_scld, loss_evo = het_mlp(
            X_tr_gp, y_tr_gp, X_te_gp, seed, device
        )
        y_pred = y_scaler.inverse_transform(y_pred_scld.reshape(-1, 1)).ravel()
        y_std = y_std_scld * y_scaler.scale_[0]
        return y_pred, y_std, loss_evo

    elif model_name == "TabPFN":
        return tab_pfn_reg(X_tr_df, y_tr_df, X_te_df, seed)

    elif model_name in ("FTTransformer", "SAINT", "ResNetMLP"):
        fn = {
            "FTTransformer": ft_transformer_reg,
            "SAINT": saint_reg,
            "ResNetMLP": resnet_mlp_reg,
        }[model_name]
        # Train in standardized y-space, then invert
        mu_scld, sigma_scld = fn(X_tr_gp, y_tr_gp, X_te_gp, seed, device)
        y_pred = y_scaler.inverse_transform(mu_scld.reshape(-1, 1)).ravel()
        y_std = sigma_scld * y_scaler.scale_[0]
        return y_pred, y_std

    elif model_name == "DeepEnsemble":
        # Train in standardized y-space, then invert
        mu_scld, sigma_scld = deep_ensemble_reg(
            X_tr_gp, y_tr_gp, X_te_gp, seed, device
        )
        y_pred = y_scaler.inverse_transform(mu_scld.reshape(-1, 1)).ravel()
        y_std = sigma_scld * y_scaler.scale_[0]
        return y_pred, y_std

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
    # Deterministic default model-seed per run
    seed = 10 * repeat + fold

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")

    data_path = Path(
        str(root_dir) + f"/data/benchmark_datasets/univariate_regression/{dataset}"
    )
    # Load dataset once
    X = pd.read_parquet(data_path / "X.parquet")
    y = pd.read_parquet(data_path / "y.parquet")["y"]
    # Load split indices
    with open(data_path / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    split = splits[repeat][fold - 1]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    X_tr_df = X.iloc[train_idx].copy()
    y_tr_df = y.iloc[train_idx].copy()
    X_te_df = X.iloc[test_idx].copy()
    y_te_df = y.iloc[test_idx].copy()
    X_tr, y_tr, X_te = (
        X_tr_df.to_numpy(),
        y_tr_df.to_numpy().ravel(),
        X_te_df.to_numpy(),
    )
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    preprocess_gp_mlp = Pipeline(steps=[("scale", StandardScaler())])

    # Standardize X and y for GP/MLP/Transformer models
    X_tr_gp = preprocess_gp_mlp.fit_transform(X_tr)
    X_te_gp = preprocess_gp_mlp.transform(X_te)
    y_scaler = StandardScaler()
    y_tr_gp = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

    # Output path: one directory per repeat/fold
    out_path = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/univariate_regression/predictions/{dataset}/repeat_{repeat:04d}/fold_{fold:02d}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # ------ full run: train all models + predict ------
    print(f"[{dataset}] repeat={repeat} fold={fold} seed={seed}")

    results_mu = {}
    results_std = {}

    for model_name in ALL_MODELS:
        print(f"Fitting {model_name}...")
        result = _fit_single_model(
            model_name,
            X_tr, y_tr, X_te,
            X_tr_df, y_tr_df, X_te_df,
            X_tr_scaled, X_te_scaled,
            X_tr_gp, X_te_gp, y_tr_gp,
            y_scaler, preprocess_gp_mlp,
            seed, device, out_path,
        )

        # Handle GP/MLP which return (mu, std, loss_evo)
        if isinstance(result, tuple) and len(result) == 3:
            y_pred, y_std_pred, loss_evo = result
            # Save training curves
            plt.figure()
            plt.plot(loss_evo)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(out_path / f"{model_name}-training.png")
            plt.close()
        else:
            y_pred, y_std_pred = result

        results_mu[model_name] = np.squeeze(y_pred)
        results_std[model_name] = np.squeeze(y_std_pred)

    # Save predictions
    df_y_pred = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,
            "y_true": y_te_df.values,
            **results_mu,
        }
    )
    df_y_pred.to_csv(out_path / "predictions_mu.csv", index=False)

    df_y_std = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,
            "y_true": y_te_df.values,
            **results_std,
        }
    )
    df_y_std.to_csv(out_path / "predictions_std.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="air")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
