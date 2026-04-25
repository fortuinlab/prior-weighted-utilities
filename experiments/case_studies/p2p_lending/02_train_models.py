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

# All model names, used for CLI validation and the full-run column order
ALL_MODELS = [
    "LogReg", "RF", "GB", "GP", "MLP", "TabPFN",
    "FTTransformer", "SAINT", "ResNetMLP", "DeepEnsemble",
]


def get_device() -> torch.device:
    """Pick CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Variational GP model
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FT-Transformer  (Gorishniy et al., 2021)
# Minimal self-contained implementation — no external rtdl dependency needed.
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


class FTTransformerModel(nn.Module):
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
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, D)
        out = self.transformer(tokens)  # (B, F+1, D)
        return self.head(out[:, 0]).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# SAINT  (Somepalli et al., 2021)
# Self-Attention + Inter-Sample Attention Transformer
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
        # x: (B, T, D) — attention across B for each token position.
        B, T, D = x.shape
        xt = self.norm(x).permute(1, 0, 2).contiguous()  # (T, B, D)
        # With batch_first=True, MHA treats T as batch, B as seq.
        attn_out, _ = self.attn(xt, xt, xt)  # (T, B, D)
        out = attn_out.permute(1, 0, 2)  # (B, T, D)
        return x + self.dropout(out)


class SAINTModel(nn.Module):
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
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            tokens = layer["self_attn"](tokens)
            tokens = layer["inter_attn"](tokens)
        return self.head(self.norm(tokens[:, 0])).squeeze(-1)


# ---------------------------------------------------------------------------
# ResNet-like MLP (Gorishniy et al., 2021)
# ---------------------------------------------------------------------------
class ResNetMLPModel(nn.Module):
    """ResNet-like MLP (Gorishniy et al., 2021)."""

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
        self.head = nn.Sequential(nn.BatchNorm1d(d_hidden), nn.Linear(d_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Generic PyTorch training loop for FTTransformer / SAINT / ResNetMLP
# ---------------------------------------------------------------------------
def _train_torch_classifier(
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
    """Train a PyTorch binary classifier (logit output) with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_tr)
    # 90/10 train/val split for early stopping
    perm = np.random.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)

    # Keep val tensors on CPU and chunk to device during evaluation, otherwise
    # SAINT's inter-sample attention (O(B²) memory) OOMs on large val splits.
    X_val_cpu, y_val_cpu = X_t[val_idx], y_t[val_idx]
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
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation — chunk at batch_size to avoid OOM (inter-sample attention)
        model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_val_cpu), batch_size):
                xb = X_val_cpu[i : i + batch_size].to(device, non_blocking=True)
                yb = y_val_cpu[i : i + batch_size].to(device, non_blocking=True)
                loss = loss_fn(model(xb), yb)
                total_loss += loss.item() * len(xb)
                total_n += len(xb)
        val_loss = total_loss / total_n

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


def _predict_torch_classifier(
    model: nn.Module, X_te: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    """Predict probabilities from a trained PyTorch binary classifier."""
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    preds = []
    for i in range(0, len(X_te_t), batch_size):
        xb = X_te_t[i : i + batch_size].to(device)
        with torch.no_grad():
            logits = model(xb)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds).astype(np.float64).ravel()


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------
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


def ft_transformer(X_tr, y_tr, X_te, seed, device):
    """FT-Transformer (Gorishniy et al., 2021)."""
    n_features = X_tr.shape[1]
    model = FTTransformerModel(
        n_features=n_features,
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    )
    y_np = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
    model = _train_torch_classifier(model, X_tr, y_np, device, seed)
    return _predict_torch_classifier(model, X_te, device)


def saint(X_tr, y_tr, X_te, seed, device):
    """SAINT (Somepalli et al., 2021)."""
    n_features = X_tr.shape[1]
    model = SAINTModel(
        n_features=n_features,
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    )
    y_np = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
    model = _train_torch_classifier(model, X_tr, y_np, device, seed)
    return _predict_torch_classifier(model, X_te, device)


def resnet_mlp(X_tr, y_tr, X_te, seed, device):
    """ResNet-MLP (Gorishniy et al., 2021)."""
    n_features = X_tr.shape[1]
    model = ResNetMLPModel(n_features=n_features, d_hidden=128, n_blocks=3, dropout=0.1)
    y_np = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
    model = _train_torch_classifier(model, X_tr, y_np, device, seed)
    return _predict_torch_classifier(model, X_te, device)


def deep_ensemble(X_tr, y_tr, X_te, preprocessor, seed, n_members=5):
    """Deep Ensemble (Lakshminarayanan et al., 2017) — average of n_members MLPs."""
    n_samples = X_tr.shape[0]
    batch_size = min(512, max(32, n_samples // 10))

    all_probs = []
    for m in range(n_members):
        member_seed = seed * 100 + m  # distinct init per member
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
            random_state=member_seed,
            verbose=False,
        )
        pipe = Pipeline([("prep", preprocessor), ("mlp", mlp_model)])
        pipe.fit(X_tr, y_tr)
        all_probs.append(pipe.predict_proba(X_te)[:, 1])

    return np.mean(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------
def _fit_single_model(
    model_name: str,
    X_tr_df, y_tr_df, X_te_df,
    preprocess, preprocess_trees, preprocess_gp_mlp,
    seed, device, out_path,
):
    """Dispatch to the right training function for *model_name*.

    Returns predicted probabilities (or a tuple for GP).
    """
    if model_name == "LogReg":
        return log_reg(X_tr_df, y_tr_df, X_te_df, preprocess)

    elif model_name == "RF":
        return ran_for(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)

    elif model_name == "GB":
        return gra_boo(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)

    elif model_name == "GP":
        X_tr_gp = preprocess_gp_mlp.fit_transform(X_tr_df)
        X_te_gp = preprocess_gp_mlp.transform(X_te_df)
        return gee_pee(X_tr_gp, y_tr_df, X_te_gp, seed, device)  # returns tuple

    elif model_name == "MLP":
        return mlp(X_tr_df, y_tr_df, X_te_df, preprocess_gp_mlp, seed)

    elif model_name == "TabPFN":
        return tab_pfn(X_tr_df, y_tr_df, X_te_df, device, seed)

    elif model_name in ("FTTransformer", "SAINT", "ResNetMLP"):
        # All three use the same preprocessed numerical input
        X_tr_np = preprocess_gp_mlp.fit_transform(X_tr_df)
        X_te_np = preprocess_gp_mlp.transform(X_te_df)
        fn = {"FTTransformer": ft_transformer, "SAINT": saint, "ResNetMLP": resnet_mlp}[
            model_name
        ]
        return fn(X_tr_np, y_tr_df, X_te_np, seed, device)

    elif model_name == "DeepEnsemble":
        return deep_ensemble(X_tr_df, y_tr_df, X_te_df, preprocess_gp_mlp, seed)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    repeat: Optional[int] = None,
    fold: Optional[int] = None,
    only: Optional[str] = None,
    force: bool = False,
):
    # Deterministic default model-seed per run
    seed = 10 * repeat + fold

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    data_path = Path(
        str(root_dir) + "/data/case_studies/p2p_lending/preprocessed"
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
        + f"/experiments/case_studies/p2p_lending/predictions/repeat_{repeat:04d}/fold_{fold:02d}"
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

    # ------------------------------------------------------------------
    # --only <ModelName>: append a single model to existing predictions
    # ------------------------------------------------------------------
    if only is not None:
        if only not in ALL_MODELS:
            raise ValueError(
                f"Unknown model '{only}'. Choose from: {ALL_MODELS}"
            )

        existing_fp = out_path / "predictions.csv"
        if not existing_fp.exists():
            raise FileNotFoundError(
                f"Cannot append {only} — no existing {existing_fp}"
            )
        df_pred = pd.read_csv(existing_fp)
        if only in df_pred.columns and not force:
            print(f"{only} already present in {existing_fp}, skipping. Use --force to overwrite.")
            return

        print(f"repeat={repeat} fold={fold} — fitting {only} only...")
        preds = _fit_single_model(
            only,
            X_tr_df, y_tr_df, X_te_df,
            preprocess, preprocess_trees, preprocess_gp_mlp,
            seed, device, out_path,
        )
        if only == "GP" and isinstance(preds, tuple):
            preds, _ = preds
        df_pred[only] = np.squeeze(preds)
        df_pred.to_csv(existing_fp, index=False)
        print(f"Done — {only} column appended.")
        return

    # ------ full run: train all models + predict ------
    print(f"repeat={repeat} fold={fold} seed={seed}")

    results = {}
    gp_loss = None

    for model_name in ALL_MODELS:
        print(f"Fitting {model_name}...")
        preds = _fit_single_model(
            model_name,
            X_tr_df, y_tr_df, X_te_df,
            preprocess, preprocess_trees, preprocess_gp_mlp,
            seed, device, out_path,
        )
        if model_name == "GP" and isinstance(preds, tuple):
            preds, gp_loss = preds
        results[model_name] = np.squeeze(preds)

    # Save GP training curve
    if gp_loss is not None:
        plt.plot(gp_loss)
        plt.xlabel("Step")
        plt.ylabel("ELBO")
        plt.grid(True)
        plt.savefig(out_path / "GP-training.png")
        plt.close()

    # Save predictions + y_true + indices
    df_pred = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,  # aligns row-wise with predictions
            "y_true": y_te_df.values,
            **results,
        }
    )

    df_pred.to_csv(out_path / "predictions.csv", index=False)
    X_te_df.to_csv(out_path / "features.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary classification model training."
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=ALL_MODELS,
        help="If set, only train this model and append to existing predictions.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing column when using --only.",
    )

    args = parser.parse_args()
    main(**vars(args))
