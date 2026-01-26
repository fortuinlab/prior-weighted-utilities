import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

root_dir = Path(__file__).resolve().parents[3]


def main():
    R = 100  # repetitions
    K = 5  # folds

    PATH = Path(
        str(root_dir) + "/data/case_studies/p2p_lending/preprocessed"
    )
    PATH.mkdir(parents=True, exist_ok=True)

    # --- fetch raw dataset ---
    col_selection = [
        "int_rate",
        "term",
        "acc_open_past_24mths",
        "dti",
        "loan_amnt",
        "avg_cur_bal",
        "emp_length",
        "fico_range_low",
        "fico_range_high",
        "home_ownership",
        "grade",
        "mort_acc",
        "revol_util",
        "revol_bal",
        "annual_inc",
        "mths_since_recent_inq",
        "earliest_cr_line",
        "delinq_2yrs",
        "loan_status",
    ]
    df = pd.read_csv(
        str(root_dir) + "/data/case_studies/p2p_lending/raw/accepted_2007_to_2018Q4.csv",
        usecols=col_selection,
    )

    # --- dataset cleaning ---
    df = df.dropna(how="all")
    # drop all rows with loan status current
    df = df[df["loan_status"] != "Current"]
    bad_status = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "In Grace Period",
        "Late (31-120 days)",
        "Late (16-30 days)",
    ]
    good_status = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
    df = df[df["loan_status"].isin(good_status + bad_status)]
    df["y"] = df["loan_status"].isin(good_status).astype(int)  # 1 for good loans
    df = df.drop(columns=["loan_status"])
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    def parse_emp_length(x):
        if pd.isna(x):
            return np.nan
        if "10+" in x:
            return 10
        if "<" in x:
            return 0
        return int(x.split()[0])

    df["emp_length"] = df["emp_length"].apply(parse_emp_length)
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    issue_date = pd.Timestamp("2018-01-01")  # choose dataset cutoff
    df["credit_history_yrs"] = (issue_date - df["earliest_cr_line"]).dt.days / 365.25
    df = df.drop(columns=["earliest_cr_line"])
    df["fico"] = 0.5 * (df["fico_range_low"] + df["fico_range_high"])
    df = df.drop(columns=["fico_range_low", "fico_range_high"])
    grade_map = {g: i for i, g in enumerate(list("ABCDEFG"), start=1)}
    df["grade"] = df["grade"].map(grade_map)
    df["home_ownership"] = (
        df["home_ownership"]
        .astype(str)
        .str.upper()
        .replace({"NONE": "OTHER", "ANY": "OTHER"})
    )
    df = df.dropna(how="all")

    df_sub = df[df["grade"] >= 5].copy()

    X_df = df_sub.drop(columns=["y"]).copy()
    y_df = df_sub["y"].copy()
    y_df = y_df.astype(int)

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

    print(f"Saved X.parquet, y.parquet, splits.pkl to {PATH}")


if __name__ == "__main__":
    main()
