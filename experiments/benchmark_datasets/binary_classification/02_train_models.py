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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tabpfn import TabPFNClassifier

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))


def get_device() -> torch.device:
    """Pick CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Variational GP model ---
class GPClassificationModel(gpytorch.models.ApproximateGP):
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


def log_reg(X_tr, y_tr, X_te, preprocessor):
    logreg = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
    clf_logreg = Pipeline([("prep", preprocessor), ("clf", logreg)])
    clf_logreg.fit(X_tr, y_tr)
    probs_test_logreg = clf_logreg.predict_proba(X_te)[:, 1]
    return probs_test_logreg


def ran_for(X_tr, y_tr, X_te, preprocessor, seed):
    rf = RandomForestClassifier(
        n_estimators=200, oob_score=False, bootstrap=True, random_state=seed, n_jobs=-1
    )
    clf_rf = Pipeline([("prep", preprocessor), ("rf", rf)])
    clf_rf.fit(X_tr, y_tr)
    probs_test_rf = clf_rf.predict_proba(X_te)[:, 1]
    return probs_test_rf


def gra_boo(X_tr, y_tr, X_te, preprocessor, seed):
    n_samples = X_tr.shape[0]
    n_est = 500 if n_samples < 1000 else 1000

    gb = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=0.03,
        max_depth=3,
        subsample=1.0,
        random_state=seed,
        verbose=False,
    )
    clf_gb = Pipeline([("prep", preprocessor), ("gb", gb)])
    clf_gb.fit(X_tr, y_tr)
    probs_test_gb = clf_gb.predict_proba(X_te)[:, 1]
    return probs_test_gb


def gee_pee(X_tr, y_tr, X_te, seed, device: torch.device):
    # Check for NaN/inf values
    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
        raise ValueError("X_tr contains NaN or infinite values")
    if np.any(np.isnan(X_te)) or np.any(np.isinf(X_te)):
        raise ValueError("X_te contains NaN or infinite values")
    if np.any(np.isnan(y_tr.values)) or np.any(np.isinf(y_tr.values)):
        raise ValueError("y_tr contains NaN or infinite values")

    # Torch tensors: keep training tensors on CPU for DataLoader pinning
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)
    # Put test tensors on target device for evaluation
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    n_tr = X_tr_t.shape[0]
    M = min(512, max(16, int(0.5 * np.sqrt(n_tr)) * 16))

    rs = np.random.RandomState(seed)
    inds = rs.choice(X_tr_t.shape[0], size=min(M, X_tr_t.shape[0]), replace=False)
    Z = X_tr_t[inds].clone().to(device)

    model = GPClassificationModel(Z, ard_dims=X_tr_t.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    batch_size = min(1024, max(32, n_tr // 4))
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
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
                optimizer.zero_grad()
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
        proba_test_gp = pred_dist.probs.cpu().numpy().astype(np.float64).ravel()

    return proba_test_gp, loss_evo


def mlp(X_tr, y_tr, X_te, preprocessor, seed):
    n_samples = X_tr.shape[0]
    batch_size = min(512, max(32, n_samples // 10))

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=batch_size,
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
        verbose=False,
    )
    pipe = Pipeline([("prep", preprocessor), ("mlp", mlp_model)])
    pipe.fit(X_tr, y_tr)
    proba_test_mlp = pipe.predict_proba(X_te)[:, 1]
    return proba_test_mlp


def tab_pfn(X_tr, y_tr, X_te, device: torch.device, seed: int):
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in df.columns:
            col = df[c].tolist()

            # detect numeric vs categorical by first non-missing value
            first = next((v for v in col if v is not None and v is not pd.NA), None)

            if isinstance(first, (int, float, np.number)):
                # numeric column → force float
                out[c] = [
                    float(v) if v is not None and v is not pd.NA else np.nan
                    for v in col
                ]
            else:
                # categorical column → force pure strings
                out[c] = [
                    "__nan__" if v is None or v is pd.NA else str(v)
                    for v in col
                ]

        return pd.DataFrame(out)

    X_tr = normalize_df(X_tr)
    X_te = normalize_df(X_te)

    # y must be pure ints
    y_tr = np.asarray([int(v) for v in y_tr.tolist()], dtype=np.int64)

    # enforce TabPFN regime
    if len(X_tr) > 10_000:
        rs = np.random.RandomState(seed)
        idx = rs.choice(len(X_tr), 10_000, replace=False)
        X_tr = X_tr.iloc[idx]
        y_tr = y_tr[idx]

    clf = TabPFNClassifier(device=device)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def main(
    dataset: Optional[str] = None,
    repeat: Optional[int] = None,
    fold: Optional[int] = None,
    only_tabpfn: bool = False,
):
    # Deterministic default model-seed per run
    seed = 10 * repeat + fold

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    data_path = Path(
        str(root_dir) + f"/data/benchmark_datasets/binary_classification/{dataset}"
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

    # Output path: one directory per repeat/fold
    out_path = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/binary_classification/predictions/{dataset}/repeat_{repeat:04d}/fold_{fold:02d}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- preprocessors (fit on training only via Pipeline) ----
    cat_cols = X_tr_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr_df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_tf = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    numeric_tf_trees = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )
    preprocess_trees = ColumnTransformer(
        [
            ("num", numeric_tf_trees, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    preprocess_gp_mlp = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    if only_tabpfn:
        existing_fp = out_path / "predictions.csv"
        if not existing_fp.exists():
            raise FileNotFoundError(f"Cannot append TabPFN — no existing {existing_fp}")
        df_pred = pd.read_csv(existing_fp)
        if "TabPFN" in df_pred.columns:
            print(f"TabPFN already present in {existing_fp}, skipping.")
            return

        print(f"[{dataset}] repeat={repeat} fold={fold} — fitting TabPFN only...")
        y_pred_tabpfn = tab_pfn(X_tr_df, y_tr_df, X_te_df, device, seed)
        df_pred["TabPFN"] = np.squeeze(y_pred_tabpfn)
        df_pred.to_csv(existing_fp, index=False)
        print("Done — TabPFN column appended.")
        return

    # ------ train + predict ------
    print(f"[{dataset}] repeat={repeat} fold={fold} seed={seed}")

    print("Fitting logistic regression...")
    y_pred_logreg = log_reg(X_tr_df, y_tr_df, X_te_df, preprocess)

    print("Fitting random forest...")
    y_pred_rf = ran_for(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)

    print("Fitting gradient boosting...")
    y_pred_gb = gra_boo(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)

    print("Fitting SVGP...")
    X_tr_gp = preprocess_gp_mlp.fit_transform(X_tr_df)
    X_te_gp = preprocess_gp_mlp.transform(X_te_df)
    y_pred_gp, gp_loss = gee_pee(X_tr_gp, y_tr_df, X_te_gp, seed, device)

    plt.plot(gp_loss)
    plt.xlabel("Step")
    plt.ylabel("ELBO")
    plt.grid(True)
    plt.savefig(out_path / "GP-training.png")
    plt.close()

    print("Fitting MLP...")
    y_pred_mlp = mlp(X_tr_df, y_tr_df, X_te_df, preprocess_gp_mlp, seed)

    print("Fitting TabPFN...")
    y_pred_tabpfn = tab_pfn(X_tr_df, y_tr_df, X_te_df, device, seed)

    # Save predictions + y_true + indices
    df_pred = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,  # aligns row-wise with predictions
            "y_true": y_te_df.values,
            "LogReg": np.squeeze(y_pred_logreg),
            "RF": np.squeeze(y_pred_rf),
            "GB": np.squeeze(y_pred_gb),
            "GP": np.squeeze(y_pred_gp),
            "MLP": np.squeeze(y_pred_mlp),
            "TabPFN": np.squeeze(y_pred_tabpfn),
        }
    )

    df_pred.to_csv(out_path / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="heartdisease")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--only-tabpfn", type=bool, default=False)

    args = parser.parse_args()
    main(**vars(args))
