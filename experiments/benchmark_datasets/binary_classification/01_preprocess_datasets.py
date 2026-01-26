import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo

root_dir = Path(__file__).resolve().parents[3]


def main():
    datasets = [
        "bank",
        "heartdisease",
        "ionosphere",
        "mushroom",
        "sonar",
    ]
    dataset_ids = {
        "bank": 222,
        "heartdisease": 45,
        "ionosphere": 52,
        "mushroom": 73,
        "sonar": 151,
    }

    R = 100  # repetitions
    K = 5  # folds

    for dataset in datasets:
        PATH = Path(
            str(root_dir)
            + f"/data/benchmark_datasets/binary_classification/{dataset}"
        )
        PATH.mkdir(parents=True, exist_ok=True)

        # --- fetch dataset ---
        raw_data = fetch_ucirepo(id=dataset_ids[dataset])

        X_df = raw_data.data.features.copy()
        y_df = raw_data.data.targets.copy()

        # --- dataset-specific cleaning / binarization ---
        if dataset == "bank":
            y_col = y_df.iloc[:, 0].astype(str).str.strip().str.lower()
            y_binary = y_col.map({"yes": 1, "no": 0})
            y_binary = y_binary.fillna(pd.to_numeric(y_col, errors="coerce")).fillna(
                y_col.map({"true": 1, "false": 0})
            )
            y_df = y_binary.astype(int)
            # -> y = 1 if client has subscribed else 0

            X_df = X_df.replace("unknown", np.nan)

            # keep your rename guard
            alts = {"day": "day_of_week"}
            for old, new in alts.items():
                if old in X_df.columns and new not in X_df.columns:
                    X_df = X_df.rename(columns={old: new})

        if dataset == "heartdisease":
            y_df = (y_df != 0).astype(int)
            X_df = X_df.replace("?", np.nan)
            # -> y = 1 if presence else 0

        if dataset == "ionosphere":
            y_col = y_df.iloc[:, 0].astype(str).str.strip().str.lower()
            y_df = (y_col == "b").astype(int)
            # -> y = 1 if "bad radar return" (no evidence of structure in the ionosphere) else 0

        if dataset == "mushroom":
            y_df = (y_df == "p").astype(int)
            # -> y = 1 if poisonous else 0

            X_df = X_df.replace("?", np.nan)

        if dataset == "sonar":
            y_col = y_df.iloc[:, 0].astype(str).str.strip().str.upper()
            y_df = (y_col == "M").astype(int)
            # -> y = 1 if object is a mine (metal cylinder), 0 if it is a rock

        # Ensure y is a Series
        if isinstance(y_df, pd.DataFrame):
            y_df = y_df.iloc[:, 0]

        # Reset indices so that train/test indices are stable and portable
        X = X_df.reset_index(drop=True)
        y = y_df.reset_index(drop=True)

        # --- Save full dataset once ---
        X.to_parquet(PATH / "X.parquet", index=False)
        y.to_frame("y").to_parquet(PATH / "y.parquet", index=False)

        # --- Create and store only indices for all splits ---
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

        print(f"[{dataset}] saved X.parquet, y.parquet, splits.pkl to {PATH}")


if __name__ == "__main__":
    main()
