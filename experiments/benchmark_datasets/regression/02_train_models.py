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

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))


def get_device() -> torch.device:
    """Pick CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # fallback: ridge-like variance estimate
        df = max(1, df)  # or warn/skip variance calc
    sigma2 = float(rss / df)  # residual variance

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

    # Get per-tree predictions
    tree_preds = np.stack([t.predict(X_te) for t in rf.estimators_], axis=1)
    sd_epistemic = tree_preds.std(axis=1, ddof=1)
    mask = np.isfinite(rf.oob_prediction_)
    resid = y_tr[mask] - rf.oob_prediction_[mask]
    sigma_aleatoric = float(np.sqrt(np.mean(resid**2)))
    y_std = np.sqrt(sd_epistemic**2 + sigma_aleatoric**2)

    return y_pred, y_std


def nat_gra_boo(X_tr, y_tr, X_te, seed):
    # Adaptive n_estimators based on dataset size
    n_samples = X_tr.shape[0]
    if n_samples < 1000:  # smaller datasets
        n_est = 500
    else:  # larger datasets
        n_est = 1000

    ngb = NGBRegressor(
        Dist=Normal,  # predict Normal(μ, σ)
        Score=MLE,  # log-likelihood training
        n_estimators=n_est,
        learning_rate=0.03,
        col_sample=0.8,
        random_state=seed,
        verbose=False,
    )
    ngb.fit(X_tr, y_tr)

    dist = ngb.pred_dist(X_te)  # object with .loc (μ) and .scale (σ)
    return dist.loc, dist.scale


def gee_pee(X_tr, y_tr, X_te, seed, device: torch.device):
    # Keep training tensors on CPU for efficient DataLoader pinning; move batches to device.
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

    # Adaptive batch size based on dataset size
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

    # Variational ELBO (regression)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr_t.shape[0])
    loss_evo = []
    epochs = 100
    print("Training SVGP...")
    # Add jitter for numerical stability
    with gpytorch.settings.cholesky_jitter(1e-4):
        for _ in tqdm(
            range(epochs), disable=False
        ):  # disable=True if you don't want progress bar
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
    # small val split for early stopping
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

    # Adaptive training parameters based on dataset size
    n_samples = X_tr.shape[0]
    batch_size = min(512, max(32, n_samples // 10))
    max_epochs = 3000 if n_samples < 1000 else 5000

    best_val = float("inf")
    best_state = None
    patience, bad = 20, 0

    loss_evo = []

    for _ in range(max_epochs):
        # train
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

        # validate
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
        mu_s, sigma_s = model(X_te_t)  # μ, σ in *scaled-y* space
        mu_s = mu_s.cpu().numpy()
        sigma_s = sigma_s.cpu().numpy()

    return mu_s, sigma_s, loss_evo


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
        str(root_dir) + f"/data/benchmark_datasets/regression/{dataset}"
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

    # Output path: one directory per repeat/fold
    out_path = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/regression/predictions/{dataset}/repeat_{repeat:04d}/fold_{fold:02d}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # ------ train + predict ------
    print(f"[{dataset}] repeat={repeat} fold={fold} seed={seed}")

    # linear regression
    print("Fitting linear regression...")
    y_pred_linreg, y_std_linreg = lin_reg(X_tr_scaled, y_tr, X_te_scaled)

    # random forest
    print("Fitting random forest...")
    y_pred_rf, y_std_rf = ran_for(X_tr, y_tr, X_te, seed)

    # natural gradient boosting
    print("Fitting natural gradient boosting...")
    y_pred_ngb, y_std_ngb = nat_gra_boo(X_tr, y_tr, X_te, seed)

    # SVGP
    print("Fitting SVGP...")
    X_tr_gp = preprocess_gp_mlp.fit_transform(X_tr)
    X_te_gp = preprocess_gp_mlp.transform(X_te)
    # Standardize y for  GP training; invert later for metrics
    y_scaler = StandardScaler()
    y_tr_gp = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    # make GP predictions
    y_pred_gp_scld, y_std_gp_scld, gp_loss = gee_pee(
        X_tr_gp, y_tr_gp, X_te_gp, seed, device
    )
    # invert standardization
    y_pred_gp = y_scaler.inverse_transform(y_pred_gp_scld.reshape(-1, 1)).ravel()
    y_std_gp = (
        y_std_gp_scld * y_scaler.scale_[0]
    )  # predictive std back to original units
    plt.plot(gp_loss)
    plt.xlabel("Step")
    plt.ylabel("ELBO")
    plt.grid(True)
    plt.savefig(str(out_path) + "/GP-training.png")
    plt.close()

    # heteroscedastic MLP
    print("Fitting heteroscedastic MLP...")
    y_pred_mlp_scld, y_std_mlp_scld, mlp_loss = het_mlp(
        X_tr_gp, y_tr_gp, X_te_gp, seed, device
    )
    y_pred_mlp = y_scaler.inverse_transform(y_pred_mlp_scld.reshape(-1, 1)).ravel()
    y_std_mlp = (
        y_std_mlp_scld * y_scaler.scale_[0]
    )  # predictive std back to original units
    plt.plot(mlp_loss)
    plt.xlabel("Step")
    plt.ylabel("NLL")
    plt.grid(True)
    plt.savefig(str(out_path) + "/MLP-training.png")
    plt.close()

    # Save predictions + y_true + indices
    df_y_pred = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,  # aligns row-wise with predictions
            "y_true": y_te_df.values,
            "LinReg": np.squeeze(y_pred_linreg),
            "RF": np.squeeze(y_pred_rf),
            "NGB": np.squeeze(y_pred_ngb),
            "GP": np.squeeze(y_pred_gp),
            "MLP": np.squeeze(y_pred_mlp),
        }
    )
    df_y_pred.to_csv(out_path / "predictions_mu.csv", index=False)

    df_y_std = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,  # aligns row-wise with predictions
            "y_true": y_te_df.values,
            "LinReg": y_std_linreg,
            "RF": y_std_rf,
            "NGB": y_std_ngb,
            "GP": y_std_gp,
            "MLP": y_std_mlp,
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
