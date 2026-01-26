import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import inspect
import sys
from collections import namedtuple

import numpy as np
import joblib
import six
import sklearn.ensemble._base as _sk_ensemble_base

# -----------------------------------------------------------------------------
# Compatibility monkey-patches for legacy costcla on modern Python / scikit-learn
# -----------------------------------------------------------------------------

# Old sklearn path: sklearn.ensemble.base -> new sklearn.ensemble._base
sys.modules["sklearn.ensemble.base"] = _sk_ensemble_base

# Old sklearn.externals.* -> modern packages
sys.modules["sklearn.externals.joblib"] = joblib
sys.modules["sklearn.externals.six"] = six
sys.modules["sklearn.externals.six.moves"] = six.moves

# Python 3.11+/3.12: inspect.getargspec removed -> shim via getfullargspec
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):
        fs = inspect.getfullargspec(func)
        return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)

    inspect.getargspec = getargspec  # type: ignore[attr-defined]

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]

# Now safe(ish) to import costcla dataset loader
from costcla.datasets import load_creditscoring2  # noqa: E402

root_dir = Path(__file__).resolve().parents[3]


def main():
    R = 100  # repetitions
    K = 5  # folds

    #  --- Kaggle Dataset --- #
    PATH = Path(str(root_dir) + "/data/case_studies/credit_approval/preprocessed/kaggle")
    PATH.mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(
        str(root_dir) + "/data/case_studies/credit_approval/raw/kaggle/cs-training.csv", index_col=0
    )
    df_test = pd.read_csv(
        str(root_dir) + "/data/case_studies/credit_approval/raw/kaggle/cs-test.csv", index_col=0
    )
    df = pd.concat([df_train, df_test])
    df.dropna(inplace=True, ignore_index=True)
    df.to_csv(
        str(root_dir) + "/data/case_studies/credit_approval/preprocessed/kaggle/features.csv",
        index=False,
    )
    X_df = df.drop(columns=["SeriousDlqin2yrs"]).copy()
    y_df = df["SeriousDlqin2yrs"].copy()
    y_df = y_df.astype(int)
    if isinstance(y_df, pd.DataFrame):
        y_df = y_df.iloc[:, 0]
    X = X_df.reset_index(drop=True)
    y = y_df.reset_index(drop=True)
    X.to_parquet(PATH / "X.parquet", index=False)
    y.to_frame("y").to_parquet(PATH / "y.parquet", index=False)
    all_splits = []
    for r in range(R):
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=r)
        splits_r = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
            splits_r.append(
                {
                    "repeat": r,
                    "fold": fold_idx,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        all_splits.append(splits_r)
    with open(PATH / "splits.pkl", "wb") as f:
        pickle.dump(all_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Kaggle Dataset] Saved X.parquet, y.parquet, splits.pkl to {PATH}")

    #  --- PAKDD Dataset --- #
    PATH = Path(str(root_dir) + "/data/case_studies/credit_approval/preprocessed/pakdd")
    PATH.mkdir(parents=True, exist_ok=True)
    data = load_creditscoring2()
    X_raw = data["data"]
    y_raw = data["target"]
    feature_names = list(data["feature_names"])
    X_df = pd.DataFrame(X_raw, columns=feature_names).copy()
    y_df = pd.Series(y_raw, name="y").astype(int)
    for col in X_df.columns:
        if X_df[col].dtype == object:
            sample = X_df[col].dropna()
            # If boolean-valued, cast to 0/1
            if not sample.empty and sample.map(type).isin([bool]).any():
                X_df[col] = X_df[col].astype(bool).astype(int)
            else:
                # Try numeric; if not possible, keep as-is (avoid FutureWarning errors='ignore')
                try:
                    X_df[col] = pd.to_numeric(X_df[col])
                except Exception:
                    pass
    if "PERSONAL_NET_INCOME" not in X_df.columns:
        raise KeyError(
            "Expected column PERSONAL_NET_INCOME not found in PAKDD features."
        )
    if "MATE_INCOME" not in X_df.columns:
        raise KeyError("Expected column MATE_INCOME not found in PAKDD features.")
    personal_income = pd.to_numeric(
        X_df["PERSONAL_NET_INCOME"], errors="coerce"
    ).fillna(0.0)
    mate_income = pd.to_numeric(X_df["MATE_INCOME"], errors="coerce").fillna(0.0)
    monthly_income = (personal_income + mate_income).astype(float)
    if "FLAG_OTHER_CARD_Y" in X_df.columns:
        other_card = pd.to_numeric(X_df["FLAG_OTHER_CARD_Y"], errors="coerce").fillna(
            0.0
        )
    else:
        other_card = pd.Series(0.0, index=X_df.index)

    if "QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION" in X_df.columns:
        additional_cards = pd.to_numeric(
            X_df["QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION"], errors="coerce"
        ).fillna(0.0)
    else:
        additional_cards = pd.Series(0.0, index=X_df.index)

    debt_proxy = (other_card + additional_cards).astype(float)
    income_floor = 100.0
    monthly_income_stable = monthly_income.clip(lower=income_floor)
    debt_ratio = (debt_proxy / monthly_income_stable).astype(float)
    X_df["MonthlyIncome"] = monthly_income
    X_df["DebtRatio"] = debt_ratio
    df_all = X_df.copy()
    df_all["y"] = y_df
    df_all.dropna(inplace=True, ignore_index=True)
    df_all.to_csv(
        str(root_dir) + "/data/case_studies/credit_approval/preprocessed/pakdd/features.csv",
        index=False,
    )
    X = df_all.drop(columns=["y"]).reset_index(drop=True)
    y = df_all["y"].astype(int).reset_index(drop=True)
    X.to_parquet(PATH / "X.parquet", index=False)
    y.to_frame("y").to_parquet(PATH / "y.parquet", index=False)
    all_splits = []
    for r in range(R):
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=r)
        splits_r = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
            splits_r.append(
                {
                    "repeat": r,
                    "fold": fold_idx,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        all_splits.append(splits_r)
    with open(PATH / "splits.pkl", "wb") as f:
        pickle.dump(all_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[PAKDD] Saved X.parquet, y.parquet, splits.pkl to {PATH}")


if __name__ == "__main__":
    main()
