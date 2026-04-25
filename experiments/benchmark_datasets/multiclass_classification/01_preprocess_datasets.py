import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

root_dir = Path(__file__).resolve().parents[3]


def main():
    datasets = [
        "iris",
        "wine",
        "drybean",
        "pendigits",
        "covertype",
    ]
    dataset_ids = {
        "iris": 53,  # Iris (K=3)
        "wine": 109,  # Wine (K=3)
        "drybean": 602,  # Dry Bean (K=7)
        "pendigits": 81,  # Pen-Based Recognition of Handwritten Digits (K=10)
        "covertype": 31,  # Covertype (K=7)
    }

    # Subsample cap: covertype has ~580k rows; cap at 15k so runs stay tractable
    # (matches the paper's suggestion) and all folds fit in memory for GB/MLP.
    subsample_to = {
        "covertype": 15_000,
    }

    R = 100  # repetitions
    K_folds = 5  # folds

    for dataset in datasets:
        PATH = Path(
            str(root_dir)
            + f"/data/benchmark_datasets/multiclass_classification/{dataset}"
        )
        PATH.mkdir(parents=True, exist_ok=True)

        # --- fetch dataset ---
        raw_data = fetch_ucirepo(id=dataset_ids[dataset])

        X_df = raw_data.data.features.copy()
        y_df = raw_data.data.targets.copy()

        # Ensure y is a Series to start
        if isinstance(y_df, pd.DataFrame):
            y_df = y_df.iloc[:, 0]

        # --- dataset-specific cleaning ---
        if dataset == "iris":
            # Labels are strings: "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            pass

        if dataset == "wine":
            # Labels already 1/2/3; LabelEncoder below maps to 0/1/2.
            pass

        if dataset == "drybean":
            # Labels are strings: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
            pass

        if dataset == "pendigits":
            # Labels are integers 0..9 already; LabelEncoder is a no-op here.
            pass

        if dataset == "covertype":
            # Labels are 1..7; LabelEncoder maps to 0..6.
            pass

        # --- integer-encode labels to 0..K-1 ---
        # LabelEncoder sorts labels and maps them to 0..K-1 deterministically,
        # so the encoding is reproducible across reruns.
        le = LabelEncoder()
        y_enc = le.fit_transform(np.asarray(y_df).ravel())
        y_df = pd.Series(y_enc, name="y", dtype=np.int64)

        # --- subsample (stratified) for very large datasets ---
        if dataset in subsample_to:
            target_n = subsample_to[dataset]
            if len(X_df) > target_n:
                rng = np.random.RandomState(0)
                # stratified subsample: sample proportionally per class
                y_arr = y_df.to_numpy()
                idx_out = []
                for c in np.unique(y_arr):
                    class_idx = np.where(y_arr == c)[0]
                    n_c = int(round(len(class_idx) * target_n / len(y_arr)))
                    n_c = max(1, min(n_c, len(class_idx)))
                    idx_out.append(rng.choice(class_idx, size=n_c, replace=False))
                idx_out = np.sort(np.concatenate(idx_out))
                X_df = X_df.iloc[idx_out].copy()
                y_df = y_df.iloc[idx_out].copy()

        # Reset indices so that train/test indices are stable and portable
        X = X_df.reset_index(drop=True)
        y = y_df.reset_index(drop=True)

        # --- Save full dataset once ---
        X.to_parquet(PATH / "X.parquet", index=False)
        y.to_frame("y").to_parquet(PATH / "y.parquet", index=False)

        # Save the label mapping for reference (original label -> encoded int)
        label_map = {str(orig): int(enc) for enc, orig in enumerate(le.classes_)}
        with open(PATH / "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        # --- Create and store only indices for all splits ---
        all_splits = []
        for r in range(R):
            kf = StratifiedKFold(n_splits=K_folds, shuffle=True, random_state=r)
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

        print(
            f"[{dataset}] n={len(X)}, K={len(le.classes_)}, "
            f"saved X.parquet, y.parquet, splits.pkl, label_map.pkl to {PATH}"
        )


if __name__ == "__main__":
    main()
