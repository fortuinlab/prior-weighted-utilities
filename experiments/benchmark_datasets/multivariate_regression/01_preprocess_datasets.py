import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo

root_dir = Path(__file__).resolve().parents[3]


def main():
    datasets = [
        "energy",
        "sgemm",
        "flare",
        "air",
        "parkinsons",
    ]
    dataset_ids = {
        "energy": 242,       # Energy Efficiency — D=2 (Heating, Cooling Load)
        "flare": 89,         # Solar Flare — D=3 (common, moderate, severe flare counts)
        "air": 360,          # Air Quality — D=3 (CO(GT), NOx(GT), NO2(GT))
        "parkinsons": 189,   # Parkinsons Telemonitoring — D=2 (motor_UPDRS, total_UPDRS)
    }

    R = 100  # repetitions
    K = 5  # folds

    for dataset in datasets:
        PATH = Path(
            str(root_dir)
            + f"/data/benchmark_datasets/multivariate_regression/{dataset}"
        )
        PATH.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Dataset-specific extraction of X (features) and y (D-dim target)
        # ------------------------------------------------------------
        if dataset == "energy":
            raw_data = fetch_ucirepo(id=dataset_ids[dataset])
            # ucimlrepo already splits features/targets; targets are Y1, Y2.
            X_df = raw_data.data.features.copy()
            y_df = raw_data.data.targets.copy()
            # Defensive: ensure both target columns are present and keep order.
            expected = [c for c in y_df.columns if c.lower().startswith("y")]
            if len(expected) >= 2:
                y_df = y_df[expected[:2]].rename(
                    columns={expected[0]: "y0", expected[1]: "y1"}
                )
            else:
                # Fallback: take the first two columns in order
                y_df = y_df.iloc[:, :2].copy()
                y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]

        elif dataset == "sgemm":
            # SGEMM GPU Kernel Performance — D=4.
            #
            # ucimlrepo does not expose this dataset via its Python-import
            # whitelist, so fetch the raw zip directly from the UCI archive
            # and read sgemm_product.csv from inside.
            #
            # 14 kernel configuration parameters (10 ordinal + 4 binary)
            # predict 4 independent timing runs Run1..Run4 (in ms) of the
            # same kernel configuration.
            #
            # The raw dataset has 241,600 rows; we subsample to 10,000
            # for computational tractability. Per the UCI dataset
            # documentation, we log-transform the running times, which
            # is standard practice for this dataset.
            import io
            import urllib.request
            import zipfile

            url = (
                "https://archive.ics.uci.edu/static/public/440/"
                "sgemm+gpu+kernel+performance.zip"
            )
            with urllib.request.urlopen(url) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Find the main CSV (skip any macOS resource-fork files)
                csv_name = next(
                    n for n in zf.namelist()
                    if n.lower().endswith(".csv") and "__macosx" not in n.lower()
                )
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)

            # Columns: 14 features, then Run1..Run4 (targets)
            y_df = df.iloc[:, -4:].copy()
            X_df = df.iloc[:, :-4].copy()

            # Log-transform running times. Run times are positive (ms).
            y_df = np.log(y_df.apply(pd.to_numeric, errors="coerce"))

            # Ensure features are numeric
            X_df = X_df.apply(pd.to_numeric, errors="coerce")

            # Drop any rows with NaN (defensive; none documented)
            joint_mask = X_df.notna().all(axis=1) & y_df.notna().all(axis=1)
            X_df = X_df.loc[joint_mask].reset_index(drop=True)
            y_df = y_df.loc[joint_mask].reset_index(drop=True)

            # Uniform subsample to 10k for tractability
            target_n = 10_000
            if len(X_df) > target_n:
                rng = np.random.RandomState(0)
                idx = np.sort(rng.choice(len(X_df), target_n, replace=False))
                X_df = X_df.iloc[idx].reset_index(drop=True)
                y_df = y_df.iloc[idx].reset_index(drop=True)

            y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]

        elif dataset == "flare":
            # Solar Flare has categorical features (Zurich class, sunspot
            # distribution, etc.) — ordinal-encode them via pandas.
            raw_data = fetch_ucirepo(id=dataset_ids[dataset])
            X_df = raw_data.data.features.copy()
            y_df = raw_data.data.targets.copy()
            # Targets: common, moderate, severe flares in 24h.
            y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]
            y_df = y_df.iloc[:, :3]
            # Encode any non-numeric feature columns as integer codes.
            for col in X_df.columns:
                if X_df[col].dtype == "object" or str(X_df[col].dtype).startswith("category"):
                    X_df[col] = X_df[col].astype("category").cat.codes

        elif dataset == "air":
            # Air Quality: the raw dataframe contains both features and target
            # concentrations interleaved. Select a clean set of features and
            # promote three pollutant concentrations to a joint target.
            raw_data = fetch_ucirepo(id=dataset_ids[dataset])
            df = raw_data["data"]["features"].copy()

            target_cols = ["CO(GT)", "NOx(GT)", "NO2(GT)"]
            feature_cols = [c for c in df.columns if c not in target_cols + ["Date", "Time"]]

            X_df = df[feature_cols].copy()
            y_df = df[target_cols].copy()

            # Coerce to numeric first, then treat -200 as NaN. Using
            # `pd.to_numeric(errors="coerce")` keeps the column dtype as
            # float rather than falling back to object dtype, which would
            # otherwise propagate pd.NA into the final X.parquet.
            X_df = X_df.apply(pd.to_numeric, errors="coerce")
            y_df = y_df.apply(pd.to_numeric, errors="coerce")
            X_df = X_df.mask(X_df == -200)
            y_df = y_df.mask(y_df == -200)

            # Drop rows where ANY target is missing — joint prediction requires
            # all targets available.
            mask = y_df.notna().all(axis=1)
            X_df = X_df.loc[mask]
            y_df = y_df.loc[mask]

            # Fill missing features with column means. With proper float dtype,
            # this now fills every column (previously object-dtype columns
            # were silently skipped).
            X_df = X_df.fillna(X_df.mean(numeric_only=True))

            y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]

        elif dataset == "parkinsons":
            # Parkinsons Telemonitoring — D=2 (motor_UPDRS, total_UPDRS).
            #
            # 16 biomedical voice measures are the features; motor_UPDRS
            # and total_UPDRS are the targets. The raw dataframe also
            # contains subject#, age, sex, and test_time — these are
            # patient metadata / ID columns, not predictive features.
            # We drop them to keep the input space voice-feature-only.
            raw_data = fetch_ucirepo(id=dataset_ids[dataset])
            X_df = raw_data.data.features.copy()
            y_df = raw_data.data.targets.copy()

            # Normalize column names (strip whitespace, replace weird chars)
            X_df.columns = [
                str(c).strip().replace("\xa0", " ") for c in X_df.columns
            ]
            y_df.columns = [
                str(c).strip().replace("\xa0", " ") for c in y_df.columns
            ]

            # Drop ID / metadata columns from features if they're present.
            drop_candidates = ["subject#", "subject", "age", "sex", "test_time"]
            for col in list(X_df.columns):
                if col.lower() in drop_candidates:
                    X_df = X_df.drop(columns=[col])

            # Coerce everything to numeric (ucimlrepo should already give
            # floats here, but safe to enforce).
            X_df = X_df.apply(pd.to_numeric, errors="coerce")
            y_df = y_df.apply(pd.to_numeric, errors="coerce")

            # Drop any rows with NaN (the dataset is documented as clean,
            # but defensive).
            joint_mask = X_df.notna().all(axis=1) & y_df.notna().all(axis=1)
            X_df = X_df.loc[joint_mask]
            y_df = y_df.loc[joint_mask]

            y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Ensure y_df is a DataFrame with D columns named y0..y{D-1}
        if isinstance(y_df, pd.Series):
            y_df = y_df.to_frame("y0")
        if list(y_df.columns) != [f"y{i}" for i in range(y_df.shape[1])]:
            y_df.columns = [f"y{i}" for i in range(y_df.shape[1])]

        # Reset indices so that train/test indices are stable and portable
        X = X_df.reset_index(drop=True)
        y = y_df.reset_index(drop=True)

        # --- Save full dataset once ---
        X.to_parquet(PATH / "X.parquet", index=False)
        y.to_parquet(PATH / "y.parquet", index=False)

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

        print(
            f"[{dataset}] n={len(X)}, D={y.shape[1]}, "
            f"saved X.parquet, y.parquet, splits.pkl to {PATH}"
        )


if __name__ == "__main__":
    main()
