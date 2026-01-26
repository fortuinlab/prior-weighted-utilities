import numpy as np
from pathlib import Path
import os
from entsoe import EntsoePandasClient

import pandas as pd

root_dir = Path(__file__).resolve().parents[3]


def main():
    PATH_RAW = Path(str(root_dir) + "/data/case_studies/electricity_market/raw")
    PATH_PREPROCESSED = Path(str(root_dir) + "/data/case_studies/electricity_market/preprocessed")
    PATH_PREPROCESSED.mkdir(parents=True, exist_ok=True)

    # balancing price data
    df_balancing_prices = pd.read_csv(PATH_RAW / "balancing_prices.csv", sep=";")
    df_balancing_prices["Datetime"] = pd.to_datetime(df_balancing_prices["Datetime"], utc=True)
    df_balancing_prices = df_balancing_prices[df_balancing_prices["Datetime"].dt.year == 2024]
    df_balancing_prices = df_balancing_prices[df_balancing_prices["Datetime"].dt.month >= 6]
    df_balancing_prices = df_balancing_prices[["Datetime", "Imbalance Price"]]
    df_balancing_prices = df_balancing_prices.set_index("Datetime").resample("h").mean()

    # day-ahead price data
    if not os.path.exists(PATH_RAW / "dayahead_prices.csv"):
        import warnings
        from bs4 import XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning, module="entsoe")
        client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
        start = pd.Timestamp('20240531', tz='Europe/Brussels')
        end = pd.Timestamp('20250101', tz='Europe/Brussels')
        country_code = 'BE'  # Belgium
        ts = client.query_day_ahead_prices(country_code, start=start, end=end)
        ts.to_csv(PATH_RAW / "dayahead_prices.csv")
    df_dayahead_prices = pd.read_csv(PATH_RAW / "dayahead_prices.csv", header=0, index_col=0)
    df_dayahead_prices.columns = ["Day-Ahead Price"]
    idx = pd.to_datetime(df_dayahead_prices.index, utc=True)
    df_dayahead_prices.index = idx
    df_dayahead_prices = df_dayahead_prices[(df_dayahead_prices.index.year == 2024) & (df_dayahead_prices.index.month >= 6)]

    # combine both price datasets
    df_prices = pd.concat([df_balancing_prices, df_dayahead_prices], axis=1)

    # wind power data
    if not os.path.exists(PATH_RAW / "wind_generation.csv"):
        client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
        start = pd.Timestamp('20240531', tz='Europe/Brussels')
        end = pd.Timestamp('20250102', tz='Europe/Brussels')
        country_code = 'BE'  # Belgium
        df_generation = client.query_generation_per_plant(country_code, start=start, end=end, psr_type="B18", include_eic=False)
        df_generation.to_csv(PATH_RAW / "wind_generation.csv")
    df_generation = pd.read_csv(PATH_RAW / "wind_generation.csv", header=None)
    header = [str(x).strip() for x in df_generation.iloc[0, 1:].tolist()]
    df_generation = df_generation.iloc[3:].reset_index(drop=True)
    df_generation.columns = ['datetime'] + header
    df_generation['datetime'] = pd.to_datetime(df_generation['datetime'], utc=True)
    df_generation = df_generation[['datetime', 'Belwind Phase 1']]
    df_generation.set_index('datetime', inplace=True)
    df_generation = df_generation[(df_generation.index.year == 2024) & (df_generation.index.month >= 6)]

    # combine prices and generation
    df = pd.concat([df_prices, df_generation], axis=1)
    df = df.rename(columns={"Belwind Phase 1": "Generation"})
    df.to_csv(PATH_PREPROCESSED / "prices_generation.csv")

    # preprocess
    L = 36  # history length
    X = []
    Y = []
    days = pd.date_range("2024-06-02", "2024-11-30", freq="D", tz="UTC")

    for d in days:
        # input window: from d-1 00:00 to d 11:00
        start = d - pd.Timedelta(days=1)
        end = d + pd.Timedelta(hours=11)
        window = df.loc[start:end, "Generation"]

        x = window.values[-L:]  # last L hours up to 11h
        # targets: generation at day d+1, all 24 hours
        target_day = d + pd.Timedelta(days=1)
        y = df.loc[target_day:target_day + pd.Timedelta(hours=23), "Generation"].values

        X.append(x)
        Y.append(y)

    X = np.asarray(X, dtype=float)       # shape [n_days, L]
    Y = np.asarray(Y, dtype=float)       # shape [n_days, 24]

    # save X, Y
    np.save(PATH_PREPROCESSED / "X.npy", X)
    np.save(PATH_PREPROCESSED / "Y.npy", Y)


if __name__ == "__main__":
    main()
