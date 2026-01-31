import gdown
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import holidays

CLEAN_DIR = Path("../data/cleansed")
CLEAN_FILE = CLEAN_DIR / "all_flights_2018-2022_cleansed.parquet"

MODEL_READY_DIR = Path("../data/model_Ready")
MODEL_READY_FILE = MODEL_READY_DIR / "flights_model_ready.parquet"
df = pd.read_parquet(CLEAN_FILE)

#day of week
df['DayOfWeek'] = df['FlightDate'].dt.day_name()

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["DayOfWeek_num"] = df["DayOfWeek"].map(day_map)

# cyclincal sin cos 
df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek_num"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek_num"] / 7)

#standardize weather features
num_cols = ["Distance", "tmpf", "vsby", "sknt", "relh", "gust"]

scaler = StandardScaler()

scaled_values = scaler.fit_transform(df[num_cols])

scaled_col_names = [f"{c}_std" for c in num_cols]

df[scaled_col_names] = scaled_values


# normalized distance from the mean of airport origin
df["distance_origin_norm"] = (
    df["Distance"] - df.groupby("Origin")["Distance"].transform("mean")
)

#has rain binary column
df["has_precip"] = (df["p01i"] > 0).astype(int)


# holidays
us_holidays = holidays.US()

# Is Holiday (0/1)
df["is_holiday"] = df["FlightDate"].dt.date.apply(
    lambda x: 1 if x in us_holidays else 0
)

# Within ±3 days of a holiday
def near_holiday(date, holiday_calendar, window=3):
    for offset in range(-window, window + 1):
        if (date + pd.Timedelta(days=offset)) in holiday_calendar:
            return 1
    return 0

df["near_holiday_3d"] = df["FlightDate"].dt.date.apply(
    lambda x: near_holiday(pd.Timestamp(x), us_holidays)
)

df.to_parquet(MODEL_READY_FILE, index=False)