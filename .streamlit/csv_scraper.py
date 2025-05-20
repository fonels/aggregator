from functools import lru_cache
from pathlib import Path
import re
import pandas as pd

_METALS = ["gold", "silver", "platinum", "palladium"]

@lru_cache(maxsize=1)
def load_all_csv(folder: str = "data") -> dict[str, pd.DataFrame]:

    data: dict[str, pd.DataFrame] = {}

    for csv_file in Path(folder).rglob("*.csv"):
        stem = csv_file.stem.lower()

        metal = next((m for m in _METALS if re.search(m, stem)), None)
        if not metal:
            continue                                   

        df = pd.read_csv(
            csv_file,
            sep=";",
            parse_dates=["timestamp"],
            index_col="timestamp",
        ).sort_index()


        num_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        data[metal] = df[num_cols]

    if missing := [m for m in _METALS if m not in data]:
        raise FileNotFoundError(f"Не найдено CSV для: {', '.join(missing)}")

    return data


def get_period_df(df: pd.DataFrame, period: str) -> pd.DataFrame:

    period = period.lower()
    end = df.index.max()

    if period == "day":
        start, freq = end.floor("D"), "1h"
    elif period == "week":
        start, freq = end - pd.Timedelta(days=7), "6h"
    elif period == "month":
        start, freq = end - pd.DateOffset(months=1), "1d"
    elif period == "year":
        start, freq = end - pd.DateOffset(years=1), "1W"
    else:
        raise ValueError(f"Неизвестный период: {period}")

    sliced = df.loc[start:end]
    resampled = sliced["close"].resample(freq).last().dropna()

    return resampled.to_frame(name="price")
