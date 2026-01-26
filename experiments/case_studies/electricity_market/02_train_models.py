import sys
import os
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

L = 36  # history length


# === GP model definition ===
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


# === MLP model definition ===
class MLP_MuSigma(torch.nn.Module):
    def __init__(self, d_in, hidden=(128, 128), dropout=0.0):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            if dropout > 0:
                layers += [torch.nn.Dropout(dropout)]
            last = h
        self.backbone = torch.nn.Sequential(*layers)
        self.mu_head = torch.nn.Linear(last, 1)
        self.rho_head = torch.nn.Linear(last, 1)   # rho -> sigma via softplus
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = self.softplus(self.rho_head(h).squeeze(-1)) + 1e-6
        return mu, sigma


def nll_gaussian(y, mu, sigma):
    # −log N(y; μ, σ²) up to constant: log σ + 0.5 * ((y−μ)/σ)²
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma)**2).mean()


def main():
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

    # initialize output dfs
    df_y_pred = pd.DataFrame(
        {
            "repeat": [i for i in dec_days.day.values for _ in range(24)],  # 1,...1,2,...2,...,31,...,31
            "fold": 1,                                                      # for compatibility
            "seed": seed,
            "test_idx":  None,                                              # for compatibility
            "y_true": np.zeros(31 * 24),
            "LinReg": np.zeros(31 * 24),
            "RF": np.zeros(31 * 24),
            "NGB": np.zeros(31 * 24),
            "GP": np.zeros(31 * 24),
            "MLP": np.zeros(31 * 24),
        }
    )
    df_y_std = pd.DataFrame(
        {
            "repeat": [i for i in dec_days.day.values for _ in range(24)],
            "fold": 1,
            "seed": seed,
            "test_idx": None,
            "y_true": np.zeros(31 * 24),
            "LinReg": np.zeros(31 * 24),
            "RF": np.zeros(31 * 24),
            "NGB": np.zeros(31 * 24),
            "GP": np.zeros(31 * 24),
            "MLP": np.zeros(31 * 24),
        }
    )

    # linear regression
    linreg_models = {}
    for k in range(24):
        y_k = Y[:, k]
        model = LinearRegression(fit_intercept=False)
        model.fit(X_design, y_k)
        residuals = y_k - model.predict(X_design)
        p = X_design.shape[1]
        sigma2 = (residuals @ residuals) / (n - p)
        linreg_models[k] = {"coef": model.coef_, "sigma2": float(sigma2)}

    def linreg_forecast(d, models, XtX_inv, L):
        """Return {k: (mu, sd)} for next-day hours k=0..23 using the last L hours up to 11:00 of day d."""

        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_lookback = window.values[-L:].astype(float)  # shape [L]
        x_lookback_scaled = scaler.transform(x_lookback.reshape(1, -1)).ravel()
        x_star = np.concatenate([[1.0], x_lookback_scaled])
        # leverage h = x^T (X^T X)^{-1} x
        h = float(x_star @ XtX_inv @ x_star)

        mus = []
        sds = []
        for k in range(24):
            beta_coef = models[k]["coef"]
            mu = float(x_star @ beta_coef)
            sigma2 = models[k]["sigma2"]
            var_pred = sigma2 * (1.0 + h)       # full predictive variance
            sd_pred = float(np.sqrt(var_pred))
            mus.append(mu)
            sds.append(sd_pred)
        return np.asarray(mus), np.asarray(sds)

    # random forest
    rf_models = {}
    for k in range(24):
        y_k = Y[:, k].ravel()

        rf = RandomForestRegressor(
            n_estimators=200, oob_score=True, bootstrap=True, random_state=seed, n_jobs=-1
        )
        rf.fit(X, y_k)

        resid = y_k - rf.oob_prediction_
        sigma_aleatoric = float(np.sqrt(np.mean(resid**2)))

        rf_models[k] = {
            "rf": rf,
            "sigma_aleatoric": sigma_aleatoric
        }

    def rf_forecast(d, rf_models, L):
        """
        Return arrays (mus, sds) for next-day hours k=0..23 using the last L hours up to 11:00 of day d.
        Uses per-tree variance (epistemic) + residual RMSE (aleatoric) for predictive std.
        """
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_last = window.values[-L:].astype(float).reshape(1, -1)  # shape [1, L]; trees don't need scaling

        mus = np.empty(24, dtype=float)
        sds = np.empty(24, dtype=float)

        for k in range(24):
            rf = rf_models[k]["rf"]
            mu = float(rf.predict(x_last)[0])

            # per-tree predictions for epistemic spread
            tree_preds = np.array([est.predict(x_last)[0] for est in rf.estimators_], dtype=float)
            sd_epistemic = float(tree_preds.std(ddof=1)) if len(tree_preds) > 1 else 0.0

            sigma_aleatoric = rf_models[k]["sigma_aleatoric"]
            sd_total = float(np.sqrt(sd_epistemic**2 + sigma_aleatoric**2))

            mus[k] = mu
            sds[k] = sd_total

        return mus, sds

    # NGBoost
    ngb_models = {}
    for k in range(24):
        y_k = Y[:, k].ravel()

        ngb = NGBRegressor(
            Dist=Normal,       # predictive distribution N(mu, sigma)
            Score=MLE,         # maximize log-likelihood
            n_estimators=1000,
            learning_rate=0.03,
            col_sample=0.8,
            random_state=seed,
            verbose=False,
            natural_gradient=True,
        )
        ngb.fit(X, y_k)

        ngb_models[k] = {"ngb": ngb}

    def ngb_forecast(d, ngb_models, L):
        """
        Return arrays (mus, sds) for next-day hours k=0..23 using the last L hours up to 11:00 of day d.
        Uses NGBoost's Normal predictive distribution (mu=loc, sigma=scale).
        """
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_last = window.values[-L:].astype(float).reshape(1, -1)  # [1, L]

        mus = np.empty(24, dtype=float)
        sds = np.empty(24, dtype=float)

        for k in range(24):
            ngb = ngb_models[k]["ngb"]
            dist = ngb.pred_dist(x_last)  # Normal with .loc and .scale
            mus[k] = float(dist.loc.ravel()[0])
            sds[k] = float(dist.scale.ravel()[0])

        return mus, sds

    # sparse GP
    X_train_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)
    Z_shared = X_train_t.clone()
    # === Train / load 24 hourly sparse GPs ===
    # === Train / load 24 hourly sparse GPs ===
    ckpt_dir = Path(str(root_dir) + f"/experiments/case_studies/electricity_market/models/gp_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Keep per-hour artifacts
    gp_artifacts = {
        "y_scalers": [],
        "ckpt_paths": [],
    }
    gp_artifacts = {
        "y_scalers": [],
        "ckpt_paths": [],
    }

    epochs = 100
    batch_size = 1024
    lr = 5e-3

    for k in range(24):
        # Standardize y per hour for numerical stability
        y_k = Y[:, k].astype(np.float32)
        y_scaler = StandardScaler()
        y_k_std = y_scaler.fit_transform(y_k.reshape(-1, 1)).ravel().astype(np.float32)

        y_train_t = torch.tensor(y_k_std, dtype=torch.float32)

        model = GPRegressionModel(Z_shared.clone(), ard_dims=X_train_t.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        ckpt_path = os.path.join(ckpt_dir, f'hour_{k}.pth')
        gp_artifacts["y_scalers"].append(y_scaler)
        gp_artifacts["ckpt_paths"].append(ckpt_path)

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

        else:
            ds = TensorDataset(X_train_t, y_train_t)
            g = torch.Generator()
            g.manual_seed(seed)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                [{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=lr
            )
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_t.shape[0])

            for it in range(epochs):
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

    # Helper to load a single hour on demand (keeps memory modest)
    def _load_hour_model(k):
        model = GPRegressionModel(Z_shared.clone(), ard_dims=X_train_t.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        checkpoint = torch.load(gp_artifacts["ckpt_paths"][k])
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        model.eval()
        likelihood.eval()
        return model, likelihood

    # === Forecast at 11:00 for next day using GP; returns (mu[24], sd[24]) on original scale ===
    def gp_forecast(d, L):
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_lookback = window.values[-L:].astype(np.float32).reshape(1, -1)  # [1, L]
        x_lookback_scaled = scaler.transform(x_lookback).astype(np.float32)
        x_lookback_t = torch.tensor(x_lookback_scaled, dtype=torch.float32)

        mus = np.empty(24, dtype=np.float64)
        sds = np.empty(24, dtype=np.float64)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for k in range(24):
                model, likelihood = _load_hour_model(k)
                pred_dist = likelihood(model(x_lookback_t))
                mu_std = pred_dist.mean.detach().cpu().numpy().astype(np.float64).ravel()[0]
                sd_std = pred_dist.stddev.detach().cpu().numpy().astype(np.float64).ravel()[0]

                # invert per-hour standardization
                y_scaler = gp_artifacts["y_scalers"][k]
                mu = y_scaler.inverse_transform([[mu_std]])[0, 0]
                sd = sd_std * y_scaler.scale_[0]

                mus[k] = mu
                sds[k] = sd

        return mus, sds

    # MLP
    def _fit_one_hour(
        X_np, y_np, d_in,
        lr=1e-3, wd=1e-4,
        max_epochs=5000, batch_size=512,
        hidden=(128, 128), dropout=0.0,
        val_frac=0.2, patience=20, improvement_tol=1e-6
    ):
        """
        X_np: [n, d_in]  (already scaled with x_scaler fitted on Jun-Nov)
        y_np: [n,]       (raw units, Jun-Nov, in chronological order)

        Returns: (best_state_dict, y_mean, y_scale)
        """
        n = X_np.shape[0]
        # --- chronological validation split: last val_frac for validation ---
        val_n = max(1, int(np.floor(val_frac * n)))
        tr_end = n - val_n
        X_tr_np, y_tr_np = X_np[:tr_end], y_np[:tr_end]
        X_val_np, y_val_np = X_np[tr_end:], y_np[tr_end:]

        # --- scale y using *training* portion only (to avoid leakage) ---
        y_scaler = StandardScaler().fit(y_tr_np.reshape(-1, 1))
        y_tr_s = y_scaler.transform(y_tr_np.reshape(-1, 1)).ravel()
        y_val_s = y_scaler.transform(y_val_np.reshape(-1, 1)).ravel()

        # tensors
        X_tr_t = torch.tensor(X_tr_np,  dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_s,   dtype=torch.float32)
        X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_s,  dtype=torch.float32)

        # model, opt
        model = MLP_MuSigma(d_in, hidden=hidden, dropout=dropout)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_val = float("inf")
        best_state = None
        bad = 0

        model.train()
        for _ in range(max_epochs):
            # ---- train (shuffle *within* training block only) ----
            perm = torch.randperm(X_tr_t.size(0))
            for i in range(0, X_tr_t.size(0), batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                opt.zero_grad(set_to_none=True)
                mu, sigma = model(xb)
                loss = nll_gaussian(yb, mu, sigma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            # ---- validate (chronological holdout: last 20%) ----
            model.eval()
            with torch.no_grad():
                mu_v, sig_v = model(X_val_t)
                val_loss = nll_gaussian(y_val_t, mu_v, sig_v).item()
            model.train()

            # ---- early stopping bookkeeping ----
            if val_loss < best_val - improvement_tol:
                best_val = val_loss
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        # fall back to last state if no improvement ever recorded
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        y_mean = float(y_scaler.mean_[0])
        y_scale = float(y_scaler.scale_[0])
        return best_state, y_mean, y_scale

    # ---------- fit (or load) the 24 hourly models on Jun–Nov ----------
    d_in = X_scaled.shape[1]

    mlp_models = {}
    y_stats = {}
    for k in range(24):
        y_k = Y[:, k].ravel().astype(float)
        state_k, y_mean_k, y_scale_k = _fit_one_hour(
            X_scaled, y_k, d_in,
            lr=1e-3, wd=1e-4,
            max_epochs=5000, batch_size=512,
            hidden=(128, 128), dropout=0.0
        )
        mlp_models[k] = state_k
        y_stats[k] = (y_mean_k, y_scale_k)

    # ---------- instantiate a template for inference ----------
    d_in = X.shape[1]
    mlp_template = MLP_MuSigma(d_in)
    mlp_template.eval()

    @torch.no_grad()
    def mlp_forecast(d, mlp_models, y_stats, x_scaler, L):
        """
        Return (mus, sds) for next-day hours k=0..23 using last L hours up to 11:00 of day d.
        """
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]
        x_lookback = window.values[-L:].astype(float).reshape(1, -1)  # [1, L]
        x_lookback_s = x_scaler.transform(x_lookback)
        xb = torch.tensor(x_lookback_s, dtype=torch.float32)

        mus = np.empty(24, dtype=float)
        sds = np.empty(24, dtype=float)

        for k in range(24):
            mlp_template.load_state_dict(mlp_models[k], strict=True)
            mu_s, sigma_s = mlp_template(xb)            # scaled-y space
            mu_s, sigma_s = float(mu_s.squeeze(0).cpu().numpy()), float(sigma_s.squeeze(0).cpu().numpy())
            y_mean_k, y_scale_k = y_stats[k]
            mus[k] = mu_s * y_scale_k + y_mean_k        # inverse target scaling
            sds[k] = sigma_s * abs(y_scale_k)

        return mus, sds

    for i, d in enumerate(dec_days):
        mus, sds = mlp_forecast(d, mlp_models, y_stats, scaler, L=L)
        start = i * 24
        end = (i + 1) * 24
        df_y_pred.iloc[start:end, df_y_pred.columns.get_loc("MLP")] = mus
        df_y_std.iloc[start:end, df_y_std.columns.get_loc("MLP")] = sds

    # predict
    prices_da = []
    prices_balancing = []
    for repeat, d in enumerate(dec_days):

        out_path = Path(
            str(root_dir)
            + f"/experiments/case_studies/electricity_market/predictions/repeat_{repeat:04d}/fold_{1:02d}"
        )
        out_path.mkdir(parents=True, exist_ok=True)

        y_true = df.loc[d: d + pd.Timedelta(hours=23), "Generation"].values.astype(float)
        prices_da = df.loc[d: d + pd.Timedelta(hours=23), "Day-Ahead Price"].values.astype(float)
        prices_balancing = df.loc[d: d + pd.Timedelta(hours=23), "Imbalance Price"].values.astype(float)

        # LinReg
        mus_linreg, sds_linreg = linreg_forecast(d, linreg_models, XtX_inv, L)

        # RF
        mus_rf, sds_rf = rf_forecast(d, rf_models, L)

        # NGB
        mus_ngb, sds_ngb = ngb_forecast(d, ngb_models, L)

        # GP
        mus_gp, sds_gp = gp_forecast(d, L)

        # MLP
        mus_mlp, sds_mlp = mlp_forecast(d, mlp_models, y_stats, scaler, L)

        # Save predictions + y_true + indices
        df_y_pred = pd.DataFrame(
            {
                "repeat": repeat,
                "fold": 1,
                "seed": seed,
                "test_idx": None,  # aligns row-wise with predictions
                "y_true": y_true,
                "LinReg": mus_linreg,
                "RF": mus_rf,
                "NGB": mus_ngb,
                "GP": mus_gp,
                "MLP": mus_mlp,
            }
        )
        df_y_pred.to_csv(out_path / "predictions_mu.csv", index=False)

        df_y_std = pd.DataFrame(
            {
                "repeat": repeat,
                "fold": 1,
                "seed": seed,
                "test_idx": None,  # aligns row-wise with predictions
                "y_true": y_true,
                "LinReg": sds_linreg,
                "RF": sds_rf,
                "NGB": sds_ngb,
                "GP": sds_gp,
                "MLP": sds_mlp,
            }
        )
        df_y_std.to_csv(out_path / "predictions_std.csv", index=False)

        df_prices = pd.DataFrame(
            {
                "repeat": repeat,
                "fold": 1,
                "seed": seed,
                "test_idx": None,  # aligns row-wise with predictions
                "Day-Ahead Price": prices_da,
                "Imbalance Price": prices_balancing,
            }
        )
        df_prices.to_csv(out_path / "prices.csv", index=False)


if __name__ == "__main__":
    main()
