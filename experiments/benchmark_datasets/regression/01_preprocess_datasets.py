import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo

root_dir = Path(__file__).resolve().parents[3]


def main():
    datasets = [
        "air",
        "auto",
        "energy",
        "power",
        "wine",
    ]
    dataset_ids = {
        "air": 360,  # Air Quality
        "auto": 9,  # Auto MPG
        "energy": 242,  # Energy Efficiency
        "power": 294,  # Combined Cycle Power Plant
        "wine": 186,  # Wine Quality (white wine)
    }

    R = 100  # repetitions
    K = 5  # folds

    for dataset in datasets:
        PATH = Path(str(root_dir) + f"/data/benchmark_datasets/regression/{dataset}")
        PATH.mkdir(parents=True, exist_ok=True)
        # fetch dataset
        raw_data = fetch_ucirepo(id=dataset_ids[dataset])

        if dataset == "air":
            df = raw_data["data"]["features"].copy()
            y_df = df["CO(GT)"]
            X_df = df.drop(columns=["CO(GT)", "Date", "Time"])

            # replace -200 placeholders with NaN
            X_df = X_df.replace(-200, pd.NA)
            y_df = y_df.replace(-200, pd.NA)

            # drop rows with missing target
            mask = y_df.notna()
            X_df = X_df.loc[mask]
            y_df = y_df.loc[mask]

            # fill missing features (mean imputation)
            pd.set_option("future.no_silent_downcasting", True)
            X_df = X_df.fillna(X_df.mean())

        else:
            X_df = raw_data.data.features.copy()
            y_df = raw_data.data.targets.copy()

        if dataset == "auto":
            # Auto MPG dataset
            # Remove rows with missing horsepower (could be NaN or '?')
            if "horsepower" in X_df.columns:
                # Handle both string '?' and NaN missing values
                if X_df["horsepower"].dtype == "object":
                    # If horsepower is object type, filter out '?' and convert to numeric
                    mask = X_df["horsepower"] != "?"
                    X_df = X_df[mask]
                    y_df = y_df[mask]
                    X_df["horsepower"] = pd.to_numeric(
                        X_df["horsepower"], errors="coerce"
                    )

                # Remove rows with NaN horsepower values
                mask = X_df["horsepower"].notna()
                X_df = X_df[mask]
                y_df = y_df[mask]

            # Handle car name - could be dropped or encoded
            if "car name" in X_df.columns:
                X_df = X_df.drop(columns=["car name"])

        if dataset == "energy":
            y_df = y_df.drop(columns=["Y2"])

        if dataset == "power":
            # Combined Cycle Power Plant - should be ready to use
            pass

        if dataset == "wine":
            # Wine quality dataset - target should already be continuous
            pass  # No special preprocessing needed

        # Ensure y_df is a Series for consistent indexing
        if isinstance(y_df, pd.DataFrame):
            y_df = y_df.iloc[:, 0]  # Take first column if DataFrame

        # Reset indices so that train/test indices are stable and portable
        X = X_df.reset_index(drop=True)
        y = y_df.reset_index(drop=True)

        # --- Save full dataset once ---
        X.to_parquet(PATH / "X.parquet", index=False)
        y.to_frame("y").to_parquet(PATH / "y.parquet", index=False)

        # --- Create and store only indices for all splits ---
        all_splits = []
        for r in range(R):
            kf = KFold(n_splits=K, shuffle=True, random_state=r)
            splits_r = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
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
