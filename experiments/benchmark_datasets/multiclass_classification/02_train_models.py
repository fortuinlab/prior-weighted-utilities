import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

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
from tabpfn import TabPFNClassifier

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

# All model names, used for CLI validation and the full-run column order.
# Multiclass experiment suite (Appendix G): five models, two of them deep.
ALL_MODELS = ["LogReg", "RF", "GB", "MLP", "TabPFN"]


def get_device() -> torch.device:
    """Pick CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------
def log_reg(X_tr, y_tr, X_te, preprocessor):
    logreg = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
    )
    clf_logreg = Pipeline([("prep", preprocessor), ("clf", logreg)])
    clf_logreg.fit(X_tr, y_tr)
    probs_test_logreg = clf_logreg.predict_proba(X_te)
    return probs_test_logreg


def ran_for(X_tr, y_tr, X_te, preprocessor, seed):
    rf = RandomForestClassifier(
        n_estimators=200, oob_score=False, bootstrap=True, random_state=seed, n_jobs=-1
    )
    clf_rf = Pipeline([("prep", preprocessor), ("rf", rf)])
    clf_rf.fit(X_tr, y_tr)
    probs_test_rf = clf_rf.predict_proba(X_te)
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
    probs_test_gb = clf_gb.predict_proba(X_te)
    return probs_test_gb


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
    proba_test_mlp = pipe.predict_proba(X_te)
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
    return clf.predict_proba(X_te)


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
    print(f"Using device: {device}")

    data_path = Path(
        str(root_dir) + f"/data/benchmark_datasets/multiclass_classification/{dataset}"
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

    # Number of classes (inferred from full y to be robust to rare classes
    # being absent in a given training fold)
    n_classes = int(np.asarray(y).max()) + 1

    # Output path: one directory per repeat/fold
    out_path = Path(
        str(root_dir)
        + f"/experiments/benchmark_datasets/multiclass_classification/predictions/{dataset}/repeat_{repeat:04d}/fold_{fold:02d}"
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

    preprocess_mlp = ColumnTransformer(
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

    # ------ full run: train all models + predict ------
    print(f"[{dataset}] repeat={repeat} fold={fold} seed={seed}")

    results = {}
    for model_name in ALL_MODELS:
        print(f"Fitting {model_name}...")
        preds = _fit_single_model(
            model_name,
            X_tr_df, y_tr_df, X_te_df,
            preprocess, preprocess_trees, preprocess_mlp,
            n_classes, seed, device, out_path,
        )
        results[model_name] = preds.astype(np.float64)

    # Save predictions + metadata.
    # Predictions are (n_test, K) simplex arrays; one array per model.
    np.savez(
        out_path / "predictions.npz",
        **results,
    )

    # Save metadata alongside (small CSV for easy inspection / join with utilities)
    df_meta = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "test_idx": test_idx,
            "y_true": y_te_df.values,
        }
    )
    df_meta.to_csv(out_path / "meta.csv", index=False)


def _fit_single_model(
    model_name: str,
    X_tr_df, y_tr_df, X_te_df,
    preprocess, preprocess_trees, preprocess_mlp,
    n_classes, seed, device, out_path,
):
    """Dispatch to the right training function for *model_name*.

    Returns predicted class probabilities of shape (n_test, n_classes).
    Models whose training data happens not to contain every class get their
    probability matrix expanded to the full class set (missing columns = 0).
    """
    if model_name == "LogReg":
        preds = log_reg(X_tr_df, y_tr_df, X_te_df, preprocess)
        classes = _classes_from_pipeline(preds, model_name, y_tr_df)

    elif model_name == "RF":
        preds = ran_for(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)
        classes = _classes_from_pipeline(preds, model_name, y_tr_df)

    elif model_name == "GB":
        preds = gra_boo(X_tr_df, y_tr_df, X_te_df, preprocess_trees, seed)
        classes = _classes_from_pipeline(preds, model_name, y_tr_df)

    elif model_name == "MLP":
        preds = mlp(X_tr_df, y_tr_df, X_te_df, preprocess_mlp, seed)
        classes = _classes_from_pipeline(preds, model_name, y_tr_df)

    elif model_name == "TabPFN":
        preds = tab_pfn(X_tr_df, y_tr_df, X_te_df, device, seed)
        # TabPFN re-indexes classes internally; we fall back to the sorted
        # unique training labels as its class ordering.
        classes = np.sort(np.unique(np.asarray(y_tr_df)))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return _align_to_full_classes(preds, classes, n_classes)


def _classes_from_pipeline(preds, model_name, y_tr_df):
    """Return the sorted unique classes actually seen during training.

    All sklearn classifiers used here expose `classes_` through the final
    estimator, and `predict_proba` columns are ordered by `classes_`.
    Since we don't keep the fitted pipeline around after the model
    function returns, we rely on the invariant that sklearn sorts
    `classes_` ascending — which matches `np.sort(np.unique(y_tr))`.
    """
    return np.sort(np.unique(np.asarray(y_tr_df)))


def _align_to_full_classes(
    preds: np.ndarray, classes: np.ndarray, n_classes: int
) -> np.ndarray:
    """Expand (n_test, |classes|) to (n_test, n_classes) with zero padding
    for classes absent from training. Leaves predictions untouched when
    all classes are present."""
    preds = np.asarray(preds)
    if preds.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return preds
    full = np.zeros((preds.shape[0], n_classes), dtype=preds.dtype)
    for j, c in enumerate(classes):
        full[:, int(c)] = preds[:, j]
    return full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
