import sys
import os
from pathlib import Path

import gpytorch
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tabpfn import TabPFNRegressor

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

L = 36  # history length

ALL_MODELS = [
    "LinReg", "RF", "NGB", "GP", "MLP", "TabPFN",
    "FTTransformer", "SAINT", "ResNetMLP", "DeepEnsemble",
]


# =====================================================================
# Model definitions
# =====================================================================

# --- GP model ---
class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, ard_dims):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, q, learn_inducing_locations=True
        )
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --- MLP (heteroscedastic, original) ---
class MLP_MuSigma(nn.Module):
    def __init__(self, d_in, hidden=(128, 128), dropout=0.0):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, 1)
        self.rho_head = nn.Linear(last, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


# --- HeteroscedasticMLP (with .loss(), used by DeepEnsemble) ---
class HeteroscedasticMLP(nn.Module):
    def __init__(self, d_in, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, 1)
        self.rho_head = nn.Linear(last, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma

    def loss(self, x, y):
        mu, sigma = self(x)
        return (2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2).mean()


# --- FT-Transformer (Gorishniy et al., 2021) — heteroscedastic regression ---
class _FTTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.bias, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight[None] + self.bias[None]


class FTTransformerRegModel(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = _FTTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mu_head = nn.Linear(d_token, 1)
        self.rho_head = nn.Linear(d_token, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        h = out[:, 0]
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


# --- SAINT (Somepalli et al., 2021) — heteroscedastic regression ---
class _IntersampleAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        xt = self.norm(x).permute(1, 0, 2).contiguous()
        attn_out, _ = self.attn(xt, xt, xt)
        out = attn_out.permute(1, 0, 2)
        return x + self.dropout(out)


class SAINTRegModel(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = _FTTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.TransformerEncoderLayer(
                    d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                    dropout=dropout, activation="gelu", batch_first=True,
                ),
                "inter_attn": _IntersampleAttentionLayer(d_token, n_heads, dropout),
            }))
        self.norm = nn.LayerNorm(d_token)
        self.mu_head = nn.Linear(d_token, 1)
        self.rho_head = nn.Linear(d_token, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
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


# --- ResNet-like MLP (Gorishniy et al., 2021) — heteroscedastic regression ---
class ResNetMLPRegModel(nn.Module):
    def __init__(self, n_features, d_hidden=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(d_hidden), nn.Linear(d_hidden, d_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
            ))
        self.norm = nn.BatchNorm1d(d_hidden)
        self.mu_head = nn.Linear(d_hidden, 1)
        self.rho_head = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


# =====================================================================
# Shared losses and training utilities
# =====================================================================
def nll_gaussian(y, mu, sigma):
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma)**2).mean()


def _gaussian_nll(mu, sigma, y):
    """Gaussian NLL: 2 log σ + ((y − μ) / σ)², averaged over batch."""
    return (2.0 * torch.log(sigma) + ((y - mu) / sigma) ** 2).mean()


def _train_torch_regressor(
    model, X_tr, y_tr, device, seed_val,
    epochs=200, lr=1e-3, weight_decay=1e-5, batch_size=256, patience=20,
):
    """Train a heteroscedastic PyTorch regression model with early stopping."""
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    n = len(X_tr)
    perm = np.random.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)

    X_val, y_val = X_t[val_idx].to(device), y_t[val_idx].to(device)
    train_ds = TensorDataset(X_t[tr_idx], y_t[tr_idx])
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, len(tr_idx)),
        shuffle=True, pin_memory=(device.type == "cuda"),
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


def _predict_torch_regressor(model, X_te, device, batch_size=2048):
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


# =====================================================================
# Main
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    PATH_PREPROCESSED = Path(str(root_dir) + "/data/case_studies/electricity_market/preprocessed")
    df = pd.read_csv(PATH_PREPROCESSED / "prices_generation.csv", index_col=0, parse_dates=True)
    X = np.load(PATH_PREPROCESSED / "X.npy")       # shape [n_days, L]
    Y = np.load(PATH_PREPROCESSED / "Y.npy")       # shape [n_days, 24]

    # prep data
    n = X.shape[0]
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)
    X_design = np.c_[np.ones((n, 1)), X_scaled]  # add intercept term
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    dec_days = pd.date_range("2024-12-01", "2024-12-31", freq="D", tz="UTC")

    # helper: extract lookback window for a given day
    def _get_lookback(d, scaled=True):
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_lookback = window.values[-L:].astype(float).reshape(1, -1)
        if scaled:
            return scaler.transform(x_lookback).astype(np.float32)
        return x_lookback

    # ================================================================
    # Train all models (24 per-hour models each)
    # ================================================================

    # --- Linear Regression ---
    print("Training LinReg...")
    linreg_models = {}
    for k in range(24):
        y_k = Y[:, k]
        model = LinearRegression(fit_intercept=False)
        model.fit(X_design, y_k)
        residuals = y_k - model.predict(X_design)
        p = X_design.shape[1]
        sigma2 = (residuals @ residuals) / (n - p)
        linreg_models[k] = {"coef": model.coef_, "sigma2": float(sigma2)}

    def linreg_forecast(d):
        x_lookback_scaled = _get_lookback(d, scaled=True).ravel()
        x_star = np.concatenate([[1.0], x_lookback_scaled])
        h = float(x_star @ XtX_inv @ x_star)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            beta = linreg_models[k]["coef"]
            mus[k] = float(x_star @ beta)
            sds[k] = float(np.sqrt(linreg_models[k]["sigma2"] * (1.0 + h)))
        return mus, sds

    # --- Random Forest ---
    print("Training RF...")
    rf_models = {}
    for k in range(24):
        y_k = Y[:, k].ravel()
        rf = RandomForestRegressor(
            n_estimators=200, oob_score=True, bootstrap=True, random_state=seed, n_jobs=-1
        )
        rf.fit(X, y_k)
        resid = y_k - rf.oob_prediction_
        sigma_aleatoric = float(np.sqrt(np.mean(resid**2)))
        rf_models[k] = {"rf": rf, "sigma_aleatoric": sigma_aleatoric}

    def rf_forecast(d):
        x_last = _get_lookback(d, scaled=False).astype(float)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            rf = rf_models[k]["rf"]
            mu = float(rf.predict(x_last)[0])
            tree_preds = np.array([est.predict(x_last)[0] for est in rf.estimators_])
            sd_epi = float(tree_preds.std(ddof=1)) if len(tree_preds) > 1 else 0.0
            sds[k] = float(np.sqrt(sd_epi**2 + rf_models[k]["sigma_aleatoric"]**2))
            mus[k] = mu
        return mus, sds

    # --- NGBoost ---
    print("Training NGB...")
    ngb_models = {}
    for k in range(24):
        y_k = Y[:, k].ravel()
        ngb = NGBRegressor(
            Dist=Normal, Score=MLE, n_estimators=1000, learning_rate=0.03,
            col_sample=0.8, random_state=seed, verbose=False, natural_gradient=True,
        )
        ngb.fit(X, y_k)
        ngb_models[k] = {"ngb": ngb}

    def ngb_forecast(d):
        x_last = _get_lookback(d, scaled=False).astype(float)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            dist = ngb_models[k]["ngb"].pred_dist(x_last)
            mus[k] = float(dist.loc.ravel()[0])
            sds[k] = float(dist.scale.ravel()[0])
        return mus, sds

    # --- Sparse GP ---
    print("Training GP...")
    X_train_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)
    Z_shared = X_train_t.clone()

    ckpt_dir = Path(str(root_dir) + f"/experiments/case_studies/electricity_market/models/gp_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    gp_artifacts = {"y_scalers": [], "ckpt_paths": []}

    for k in range(24):
        y_k = Y[:, k].astype(np.float32)
        y_scaler_k = StandardScaler()
        y_k_std = y_scaler_k.fit_transform(y_k.reshape(-1, 1)).ravel().astype(np.float32)
        y_train_t = torch.tensor(y_k_std, dtype=torch.float32)

        model = GPRegressionModel(Z_shared.clone(), ard_dims=X_train_t.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        ckpt_path = os.path.join(ckpt_dir, f'hour_{k}.pth')
        gp_artifacts["y_scalers"].append(y_scaler_k)
        gp_artifacts["ckpt_paths"].append(ckpt_path)

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        else:
            ds = TensorDataset(X_train_t, y_train_t)
            g = torch.Generator()
            g.manual_seed(seed)
            loader = DataLoader(ds, batch_size=1024, shuffle=True, generator=g)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                [{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=5e-3
            )
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_t.shape[0])
            for it in range(100):
                for xb, yb in loader:
                    optimizer.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = -mll(out, yb)
                    loss.backward()
                    optimizer.step()
            torch.save({
                'model_state_dict': model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
            }, ckpt_path)

    def _load_hour_gp(k):
        model = GPRegressionModel(Z_shared.clone(), ard_dims=X_train_t.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        checkpoint = torch.load(gp_artifacts["ckpt_paths"][k])
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        model.eval()
        likelihood.eval()
        return model, likelihood

    def gp_forecast(d):
        x_s = torch.tensor(_get_lookback(d, scaled=True), dtype=torch.float32)
        mus, sds = np.empty(24), np.empty(24)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for k in range(24):
                model, likelihood = _load_hour_gp(k)
                pred_dist = likelihood(model(x_s))
                mu_std = pred_dist.mean.cpu().numpy().ravel()[0]
                sd_std = pred_dist.stddev.cpu().numpy().ravel()[0]
                y_sc = gp_artifacts["y_scalers"][k]
                mus[k] = y_sc.inverse_transform([[mu_std]])[0, 0]
                sds[k] = sd_std * y_sc.scale_[0]
        return mus, sds

    # --- MLP (heteroscedastic, original) ---
    print("Training MLP...")
    d_in = X_scaled.shape[1]

    def _fit_one_hour_mlp(X_np, y_np, d_in_,
                          lr=1e-3, wd=1e-4, max_epochs=5000, batch_size=512,
                          hidden=(128, 128), dropout=0.0, val_frac=0.2, patience=20):
        n_ = X_np.shape[0]
        val_n = max(1, int(np.floor(val_frac * n_)))
        tr_end = n_ - val_n
        X_tr_np, y_tr_np = X_np[:tr_end], y_np[:tr_end]
        X_val_np, y_val_np = X_np[tr_end:], y_np[tr_end:]

        y_scaler_ = StandardScaler().fit(y_tr_np.reshape(-1, 1))
        y_tr_s = y_scaler_.transform(y_tr_np.reshape(-1, 1)).ravel()
        y_val_s = y_scaler_.transform(y_val_np.reshape(-1, 1)).ravel()

        X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_s, dtype=torch.float32)

        model_ = MLP_MuSigma(d_in_, hidden=hidden, dropout=dropout)
        opt = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=wd)

        best_val = float("inf")
        best_state = None
        bad = 0
        model_.train()
        for _ in range(max_epochs):
            perm = torch.randperm(X_tr_t.size(0))
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                opt.zero_grad(set_to_none=True)
                mu, sigma = model_(xb)
                loss = nll_gaussian(yb, mu, sigma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 5.0)
                opt.step()
            model_.eval()
            with torch.no_grad():
                mu_v, sig_v = model_(X_val_t)
                val_loss = nll_gaussian(y_val_t, mu_v, sig_v).item()
            model_.train()
            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
                best_state = {k_: v.detach().cpu().clone() for k_, v in model_.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is None:
            best_state = {k_: v.detach().cpu().clone() for k_, v in model_.state_dict().items()}
        return best_state, float(y_scaler_.mean_[0]), float(y_scaler_.scale_[0])

    mlp_models = {}
    mlp_y_stats = {}
    for k in range(24):
        y_k = Y[:, k].ravel().astype(float)
        state_k, y_mean_k, y_scale_k = _fit_one_hour_mlp(X_scaled, y_k, d_in)
        mlp_models[k] = state_k
        mlp_y_stats[k] = (y_mean_k, y_scale_k)

    mlp_template = MLP_MuSigma(d_in)
    mlp_template.eval()

    @torch.no_grad()
    def mlp_forecast(d):
        xb = torch.tensor(_get_lookback(d, scaled=True), dtype=torch.float32)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            mlp_template.load_state_dict(mlp_models[k], strict=True)
            mu_s, sigma_s = mlp_template(xb)
            mu_s, sigma_s = float(mu_s), float(sigma_s)
            y_mean_k, y_scale_k = mlp_y_stats[k]
            mus[k] = mu_s * y_scale_k + y_mean_k
            sds[k] = sigma_s * abs(y_scale_k)
        return mus, sds

    # --- TabPFN ---
    print("Training TabPFN...")
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    tabpfn_models = {}
    for k in range(24):
        y_k = Y[:, k].ravel().astype(np.float64)
        reg = TabPFNRegressor(device="cpu")
        reg.fit(X_df, y_k)
        tabpfn_models[k] = reg

    def tabpfn_forecast(d):
        x_last = _get_lookback(d, scaled=False).astype(float)
        x_last_df = pd.DataFrame(x_last, columns=[f"x{i}" for i in range(x_last.shape[1])])
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            reg = tabpfn_models[k]
            mus[k] = float(reg.predict(x_last_df, output_type="mean")[0])
            quantiles = reg.predict(x_last_df, output_type="quantiles", quantiles=[0.1587, 0.8413])
            sds[k] = max(float((quantiles[1][0] - quantiles[0][0]) / 2.0), 1e-12)
        return mus, sds

    # --- FTTransformer / SAINT / ResNetMLP (heteroscedastic, per-hour) ---
    def _train_heteroscedastic_per_hour(model_cls, model_name, model_kwargs):
        """Train 24 per-hour heteroscedastic models. Returns {k: (state_dict, y_mean, y_scale)}."""
        print(f"Training {model_name}...")
        artifacts = {}
        for k in range(24):
            y_k = Y[:, k].astype(np.float32)
            y_scaler_k = StandardScaler()
            y_k_std = y_scaler_k.fit_transform(y_k.reshape(-1, 1)).ravel().astype(np.float32)

            model_ = model_cls(**model_kwargs)
            model_ = _train_torch_regressor(
                model_, X_scaled.astype(np.float32), y_k_std, device, seed,
            )
            state = {k_: v.cpu().clone() for k_, v in model_.state_dict().items()}
            artifacts[k] = {
                "state": state,
                "y_mean": float(y_scaler_k.mean_[0]),
                "y_scale": float(y_scaler_k.scale_[0]),
            }
        return artifacts

    n_features = X_scaled.shape[1]

    ftt_artifacts = _train_heteroscedastic_per_hour(
        FTTransformerRegModel, "FTTransformer",
        dict(n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1),
    )
    saint_artifacts = _train_heteroscedastic_per_hour(
        SAINTRegModel, "SAINT",
        dict(n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1),
    )
    resnet_artifacts = _train_heteroscedastic_per_hour(
        ResNetMLPRegModel, "ResNetMLP",
        dict(n_features=n_features, d_hidden=128, n_blocks=3, dropout=0.1),
    )

    def _heteroscedastic_forecast(d, model_cls, model_kwargs, artifacts):
        x_s = _get_lookback(d, scaled=True).astype(np.float32)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            art = artifacts[k]
            model_ = model_cls(**model_kwargs).to(device)
            model_.load_state_dict(art["state"])
            mu_pred, sigma_pred = _predict_torch_regressor(model_, x_s, device)
            mus[k] = float(mu_pred[0]) * art["y_scale"] + art["y_mean"]
            sds[k] = float(sigma_pred[0]) * abs(art["y_scale"])
        return mus, sds

    ftt_kwargs = dict(n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1)
    saint_kwargs = dict(n_features=n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1)
    resnet_kwargs = dict(n_features=n_features, d_hidden=128, n_blocks=3, dropout=0.1)

    def ftt_forecast(d):
        return _heteroscedastic_forecast(d, FTTransformerRegModel, ftt_kwargs, ftt_artifacts)

    def saint_forecast(d):
        return _heteroscedastic_forecast(d, SAINTRegModel, saint_kwargs, saint_artifacts)

    def resnet_forecast(d):
        return _heteroscedastic_forecast(d, ResNetMLPRegModel, resnet_kwargs, resnet_artifacts)

    # --- DeepEnsemble (5 HeteroscedasticMLP members per hour) ---
    print("Training DeepEnsemble...")
    n_members = 5
    de_artifacts = {}
    for k in range(24):
        y_k = Y[:, k].astype(np.float32)
        y_scaler_k = StandardScaler()
        y_k_std = y_scaler_k.fit_transform(y_k.reshape(-1, 1)).ravel().astype(np.float32)

        member_states = []
        for m in range(n_members):
            member_seed = seed * 100 + m
            X_tr_m, X_val_m, y_tr_m, y_val_m = train_test_split(
                X_scaled.astype(np.float32), y_k_std, test_size=0.2, random_state=member_seed,
            )
            d_in_m = X_tr_m.shape[1]
            model_ = HeteroscedasticMLP(d_in_m).to(device)
            opt = torch.optim.Adam(model_.parameters(), lr=1e-3, weight_decay=1e-4)

            X_tr_t = torch.tensor(X_tr_m, dtype=torch.float32, device=device)
            y_tr_t = torch.tensor(y_tr_m, dtype=torch.float32, device=device)
            X_val_t = torch.tensor(X_val_m, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val_m, dtype=torch.float32, device=device)

            n_s = X_tr_m.shape[0]
            bs = min(512, max(32, n_s // 10))
            max_ep = 3000 if n_s < 1000 else 5000
            best_val = float("inf")
            best_state = None
            pat, bad = 20, 0

            for _ in range(max_ep):
                model_.train()
                perm = torch.randperm(X_tr_t.size(0), device=device)
                for i in range(0, X_tr_t.size(0), bs):
                    idx = perm[i : i + bs]
                    xb, yb = X_tr_t[idx], y_tr_t[idx]
                    opt.zero_grad()
                    loss = model_.loss(xb, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_.parameters(), 5.0)
                    opt.step()
                model_.eval()
                with torch.no_grad():
                    val_loss = model_.loss(X_val_t, y_val_t).item()
                if val_loss < best_val - 1e-6:
                    best_val, bad = val_loss, 0
                    best_state = {k_: v.detach().cpu().clone() for k_, v in model_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= pat:
                        break
            if best_state is not None:
                model_.load_state_dict(best_state)
            member_states.append({k_: v.cpu().clone() for k_, v in model_.state_dict().items()})

        de_artifacts[k] = {
            "members": member_states,
            "y_mean": float(y_scaler_k.mean_[0]),
            "y_scale": float(y_scaler_k.scale_[0]),
        }

    de_template = HeteroscedasticMLP(n_features)
    de_template.eval()

    @torch.no_grad()
    def de_forecast(d):
        x_s = torch.tensor(_get_lookback(d, scaled=True), dtype=torch.float32, device=device)
        mus, sds = np.empty(24), np.empty(24)
        for k in range(24):
            art = de_artifacts[k]
            all_mu, all_sigma = [], []
            for state in art["members"]:
                de_template.load_state_dict(state, strict=True)
                de_template.to(device)
                mu_m, sigma_m = de_template(x_s)
                all_mu.append(float(mu_m))
                all_sigma.append(float(sigma_m))
            all_mu = np.array(all_mu)
            all_sigma = np.array(all_sigma)
            mu_bar = all_mu.mean()
            var_bar = (all_sigma**2).mean() + (all_mu**2).mean() - mu_bar**2
            sigma_bar = np.sqrt(max(var_bar, 1e-12))
            mus[k] = mu_bar * art["y_scale"] + art["y_mean"]
            sds[k] = sigma_bar * abs(art["y_scale"])
        return mus, sds

    # ================================================================
    # Forecast loop over December days
    # ================================================================
    forecast_fns = {
        "LinReg": linreg_forecast,
        "RF": rf_forecast,
        "NGB": ngb_forecast,
        "GP": gp_forecast,
        "MLP": mlp_forecast,
        "TabPFN": tabpfn_forecast,
        "FTTransformer": ftt_forecast,
        "SAINT": saint_forecast,
        "ResNetMLP": resnet_forecast,
        "DeepEnsemble": de_forecast,
    }

    for repeat, d in enumerate(dec_days):
        print(f"Forecasting day {d.date()} (repeat={repeat})...")

        out_path = Path(
            str(root_dir)
            + f"/experiments/case_studies/electricity_market/predictions/repeat_{repeat:04d}/fold_{1:02d}"
        )
        out_path.mkdir(parents=True, exist_ok=True)

        y_true = df.loc[d: d + pd.Timedelta(hours=23), "Generation"].values.astype(float)
        prices_da = df.loc[d: d + pd.Timedelta(hours=23), "Day-Ahead Price"].values.astype(float)
        prices_balancing = df.loc[d: d + pd.Timedelta(hours=23), "Imbalance Price"].values.astype(float)

        mu_dict = {}
        sd_dict = {}
        for model_name in ALL_MODELS:
            mus_m, sds_m = forecast_fns[model_name](d)
            mu_dict[model_name] = mus_m
            sd_dict[model_name] = sds_m

        df_y_pred = pd.DataFrame({
            "repeat": repeat, "fold": 1, "seed": seed, "test_idx": None,
            "y_true": y_true,
            **mu_dict,
        })
        df_y_pred.to_csv(out_path / "predictions_mu.csv", index=False)

        df_y_std = pd.DataFrame({
            "repeat": repeat, "fold": 1, "seed": seed, "test_idx": None,
            "y_true": y_true,
            **sd_dict,
        })
        df_y_std.to_csv(out_path / "predictions_std.csv", index=False)

        df_prices = pd.DataFrame({
            "repeat": repeat, "fold": 1, "seed": seed, "test_idx": None,
            "Day-Ahead Price": prices_da, "Imbalance Price": prices_balancing,
        })
        df_prices.to_csv(out_path / "prices.csv", index=False)

    print("All done.")


if __name__ == "__main__":
    main()
