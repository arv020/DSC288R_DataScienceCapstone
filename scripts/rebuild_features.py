# rebuild_features.py
# takes final_flights_model_dataset.parquet and rebuilds targets_features_split
# with the new features from the datasets we merged (opsnet, delay causes, runways, etc)
#
# run: python rebuild_features.py
# needs: final_flights_model_dataset.parquet in same folder
# outputs: targets_features_split_v2.parquet, cols_v2.py

import numpy as np
import pandas as pd
import gc
import time
from sklearn.preprocessing import LabelEncoder

# ---- load data ----

t0 = time.time()
print("loading data...")
df = pd.read_parquet("final_flights_model_dataset.parquet")
print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")

df["FlightDate"] = pd.to_datetime(df["FlightDate"])

# drop 2020 (covid)
df = df[df["FlightDate"].dt.year != 2020].copy()
print(f"after dropping 2020: {len(df):,} rows")
print(f"columns: {df.columns.tolist()}")


# ---- targets ----

print("\ncreating targets...")

if "is_cancelled" in df.columns:
    df["target_cancelled"] = df["is_cancelled"].astype(int)
elif "Cancelled" in df.columns:
    df["target_cancelled"] = df["Cancelled"].astype(int)

if "is_delayed" in df.columns:
    df["target_delayed"] = df["is_delayed"].astype(int)
elif "DepDelay" in df.columns:
    df["target_delayed"] = ((df["DepDelay"] >= 15) & (df["target_cancelled"] == 0)).astype(int)

df["target_delayed_non_cancelled"] = df["target_delayed"]

# 0=on_time, 1=delayed, 2=cancelled
df["target"] = np.select(
    [df["target_cancelled"] == 1, df["target_delayed"] == 1],
    [2, 1],
    default=0
)

print(f"cancel rate: {df['target_cancelled'].mean():.4f}")
print(f"delay rate:  {df['target_delayed'].mean():.4f}")


# ---- temporal split (same as before: 2018-2019 train, 2021 val, 2022 test) ----

yr = df["FlightDate"].dt.year
df["split"] = np.select(
    [yr.isin([2018, 2019]), yr == 2021, yr == 2022],
    ["train", "val", "test"],
    default="drop"
)
df = df[df["split"] != "drop"].copy()

for s in ["train", "val", "test"]:
    print(f"  {s}: {(df['split'] == s).sum():,}")


# ---- feature engineering (same stuff we already had) ----

print("\nengineering features...")

# month/dow cyclical
month = df["Month"] if "Month" in df.columns else df["FlightDate"].dt.month
dow = df["DayOfWeek"] if "DayOfWeek" in df.columns else df["FlightDate"].dt.dayofweek

df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)
df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

# time of day
df["is_early_morning"] = df["dep_hour"].isin([5, 6, 7]).astype(int)
df["is_evening"] = (df["dep_hour"] >= 18).astype(int)

# seasonal
df["is_summer"] = month.isin([6, 7, 8]).astype(int)
df["is_holiday_season"] = (month.isin([11, 12]) | (month == 1)).astype(int)

# weather thresholds
df["high_wind"] = (df["sknt"].fillna(0) > 20).astype(int)
df["low_visibility"] = (df["vsby"].fillna(10) < 3).astype(int)
df["precip_light"] = ((df["p01i"].fillna(0) > 0) & (df["p01i"] <= 0.1)).astype(int)
df["precip_moderate"] = ((df["p01i"].fillna(0) > 0.1) & (df["p01i"] <= 0.5)).astype(int)
df["precip_heavy"] = (df["p01i"].fillna(0) > 0.5).astype(int)

# composite weather severity (0-1 scale)
df["weather_severity"] = (
    df["sknt"].fillna(0) / 30 +
    df["gust"].fillna(0) / 50 +
    (1 - df["vsby"].fillna(10) / 10) +
    df["p01i"].fillna(0).clip(upper=2) / 2
) / 4

# encode categoricals
ws_map = {"airport_hour_match": 0, "region_hour_match": 1, "still_missing": 2}
df["weather_source_enc"] = df["weather_source"].map(ws_map).fillna(2).astype(int) if "weather_source" in df.columns else 2

le_air = LabelEncoder()
df["Airline_enc"] = le_air.fit_transform(df["Airline"].astype(str)) if "Airline" in df.columns else 0

le_orig = LabelEncoder()
df["Origin_enc"] = le_orig.fit_transform(df["Origin"].astype(str)) if "Origin" in df.columns else 0

le_reg = LabelEncoder()
df["region_enc"] = le_reg.fit_transform(df["region"].astype(str).fillna("Unknown")) if "region" in df.columns else 0


# ---- target encoding (smoothed, from training data only) ----

print("target encoding...")
is_train = df["split"] == "train"
M_AIR = 10_000
M_APT = 5_000

global_cr = df.loc[is_train, "target_cancelled"].mean()
global_dr = df.loc[is_train, "target_delayed"].mean()

# airline rates
agg_air = df.loc[is_train].groupby("Airline_enc").agg(
    n=("target_cancelled", "size"),
    cs=("target_cancelled", "sum"),
    ds=("target_delayed", "sum"),
).reset_index()
agg_air["airline_cancel_rate"] = (agg_air["cs"] + M_AIR * global_cr) / (agg_air["n"] + M_AIR)
agg_air["airline_delay_rate"] = (agg_air["ds"] + M_AIR * global_dr) / (agg_air["n"] + M_AIR)

df = df.merge(agg_air[["Airline_enc", "airline_cancel_rate", "airline_delay_rate"]], on="Airline_enc", how="left")
df["airline_cancel_rate"] = df["airline_cancel_rate"].fillna(global_cr)
df["airline_delay_rate"] = df["airline_delay_rate"].fillna(global_dr)

# airport rates
agg_apt = df.loc[is_train].groupby("Origin_enc").agg(
    n=("target_cancelled", "size"),
    cs=("target_cancelled", "sum"),
    ds=("target_delayed", "sum"),
).reset_index()
agg_apt["airport_cancel_rate"] = (agg_apt["cs"] + M_APT * global_cr) / (agg_apt["n"] + M_APT)
agg_apt["airport_delay_rate"] = (agg_apt["ds"] + M_APT * global_dr) / (agg_apt["n"] + M_APT)

df = df.merge(agg_apt[["Origin_enc", "airport_cancel_rate", "airport_delay_rate"]], on="Origin_enc", how="left")
df["airport_cancel_rate"] = df["airport_cancel_rate"].fillna(global_cr)
df["airport_delay_rate"] = df["airport_delay_rate"].fillna(global_dr)


# ---- lag features (1-day per airport, same as before) ----

print("lag features...")
orig = "Origin" if "Origin" in df.columns else "Origin_enc"
df["_date"] = df["FlightDate"].dt.normalize()

daily = df.loc[is_train].groupby([orig, "_date"]).agg(
    n=("target_cancelled", "size"),
    nc=("target_cancelled", "sum"),
    nd=("target_delayed", "sum"),
).reset_index()
daily["cr"] = daily["nc"] / daily["n"]
daily["dr"] = daily["nd"] / daily["n"]
daily["_date"] = pd.to_datetime(daily["_date"])
daily = daily.sort_values([orig, "_date"])

daily["lag1_cancel_rate"] = daily.groupby(orig)["cr"].shift(1)
daily["lag1_delay_rate"] = daily.groupby(orig)["dr"].shift(1)
daily["lag1_volume"] = daily.groupby(orig)["n"].shift(1)
daily["cancelled_yesterday"] = (daily.groupby(orig)["nc"].shift(1) > 0).astype(float)

# shift forward 1 day so merge gives yesterday's stats
lag = daily[[orig, "_date", "lag1_cancel_rate", "lag1_delay_rate", "lag1_volume", "cancelled_yesterday"]].copy()
lag["_date"] = lag["_date"] + pd.Timedelta(days=1)

df = df.merge(lag, on=[orig, "_date"], how="left")
df["lag1_cancel_rate"] = df["lag1_cancel_rate"].fillna(0)
df["lag1_delay_rate"] = df["lag1_delay_rate"].fillna(0)
df["lag1_volume"] = df["lag1_volume"].fillna(df["lag1_volume"].median())
df["cancelled_yesterday"] = df["cancelled_yesterday"].fillna(0)

# hourly congestion
hourly = df.loc[is_train].groupby([orig, "_date", "dep_hour"]).size().reset_index(name="hourly_flights")
df = df.merge(hourly, on=[orig, "_date", "dep_hour"], how="left")
df["hourly_flights"] = df["hourly_flights"].fillna(1)


# ---- NEW STUFF: features from the merged datasets ----

print("\nadding new features from merged datasets...")

# airport operations (opsnet - daily)
if "airport_operations" in df.columns:
    df["airport_operations"] = df["airport_operations"].fillna(0)
    df["has_opsnet"] = (df["airport_operations"] > 0).astype(int)
    print(f"  airport_operations: mean={df['airport_operations'].mean():.0f}, nonzero={(df['airport_operations'] > 0).mean()*100:.1f}%")
else:
    df["airport_operations"] = 0
    df["has_opsnet"] = 0

# runways
if "num_runways" in df.columns:
    df["num_runways"] = df["num_runways"].fillna(df["num_runways"].median())
else:
    df["num_runways"] = 3

if "max_runway_length_ft" in df.columns:
    df["max_runway_length_ft"] = df["max_runway_length_ft"].fillna(df["max_runway_length_ft"].median())
else:
    df["max_runway_length_ft"] = 8000

# ops per runway
if "ops_per_runway" not in df.columns:
    df["ops_per_runway"] = df["airport_operations"] / df["num_runways"].clip(lower=1)
df["ops_per_runway"] = df["ops_per_runway"].fillna(0)

# elevation
if "airport_elevation" in df.columns:
    df["airport_elevation"] = df["airport_elevation"].fillna(df["airport_elevation"].median())
else:
    df["airport_elevation"] = 0

# delay cause features - LAGGED BY 1 MONTH to prevent leakage
# these come in at airport-year-month granularity, so we shift forward 1 month
# so a flight in march gets february's delay stats
print("  lagging delay cause features by 1 month...")

delay_cols = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
have_delay = [c for c in delay_cols if c in df.columns]

if have_delay:
    df["_yr"] = df["FlightDate"].dt.year
    df["_mo"] = df["FlightDate"].dt.month

    # get unique airport-year-month delay stats from training data
    lkup = df.loc[is_train].groupby([orig, "_yr", "_mo"]).agg(
        {c: "first" for c in have_delay}
    ).reset_index()

    # shift forward 1 month — so jan data becomes available for feb flights
    lkup["_merge_yr"] = lkup["_yr"]
    lkup["_merge_mo"] = lkup["_mo"] + 1
    # dec -> jan rollover
    rollover = lkup["_merge_mo"] > 12
    lkup.loc[rollover, "_merge_mo"] = 1
    lkup.loc[rollover, "_merge_yr"] += 1

    renames = {c: f"lag1m_{c}" for c in have_delay}
    lkup = lkup.rename(columns=renames)

    keep = [orig, "_merge_yr", "_merge_mo"] + list(renames.values())
    lkup = lkup[keep]

    df = df.merge(lkup, left_on=[orig, "_yr", "_mo"], right_on=[orig, "_merge_yr", "_merge_mo"], how="left")

    for c in renames.values():
        if c in df.columns:
            df[c] = df[c].fillna(0)
            print(f"    {c}: mean={df[c].mean():.1f}, coverage={((df[c] > 0).mean()*100):.1f}%")

    # cleanup temp cols
    df.drop(columns=["_merge_yr", "_merge_mo", "_yr", "_mo"], errors="ignore", inplace=True)
else:
    print("  no delay cause columns found, skipping")
    for c in delay_cols:
        df[f"lag1m_{c}"] = 0


# ---- select columns and save ----

print("\nselecting final columns...")

meta = ["FlightDate", "split", "target", "target_cancelled", "target_delayed", "target_delayed_non_cancelled"]

features = [
    # time
    "dep_hour",
    # distance
    "Distance",
    # weather raw
    "tmpf", "vsby", "sknt", "p01i", "relh", "gust",
    # cyclical
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # time of day flags
    "is_early_morning", "is_evening",
    # seasonal
    "is_summer", "is_holiday_season",
    # weather thresholds
    "high_wind", "low_visibility", "precip_light", "precip_moderate", "precip_heavy",
    # weather composite
    "weather_severity", "weather_source_enc",
    # categoricals
    "Airline_enc", "Origin_enc", "region_enc",
    # target encoding
    "airline_delay_rate", "airline_cancel_rate", "airport_delay_rate", "airport_cancel_rate",
    # lag (1 day)
    "lag1_delay_rate", "lag1_cancel_rate", "lag1_volume", "cancelled_yesterday",
    # congestion
    "hourly_flights",
    # NEW: airport infrastructure
    "airport_operations", "has_opsnet", "ops_per_runway",
    "num_runways", "max_runway_length_ft", "airport_elevation",
    # NEW: delay causes (lagged 1 month)
    "lag1m_carrier_delay", "lag1m_weather_delay", "lag1m_nas_delay",
    "lag1m_late_aircraft_delay", "lag1m_security_delay",
]

# only keep what actually exists (in case some columns didnt make it)
features = [c for c in features if c in df.columns]

out = df[meta + features].copy()
out.to_parquet("targets_features_split_v2.parquet", index=False)
print(f"saved: targets_features_split_v2.parquet ({len(out):,} rows x {len(meta) + len(features)} cols)")
print(f"features: {len(features)} total ({len(features) - 33} new)")

for s in ["train", "val", "test"]:
    print(f"  {s}: {(out['split'] == s).sum():,}")


# ---- write cols_v2.py ----

print("\nwriting cols_v2.py...")

with open("cols_v2.py", "w") as f:
    f.write("# cols_v2.py — updated with new features from merged datasets\n")
    f.write(f"# model A: {len(features)} features, model B: {len(features)} features\n\n")
    f.write(f"model_a_feature_cols = {features}\n\n")
    f.write(f"model_b_feature_cols = {features}\n")

print(f"saved: cols_v2.py (model A: {len(features)}, model B: {len(features)})")


# ---- done ----

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"\nnext steps:")
print(f"  1. in imbalance_fix_v2.py change the import to: from cols_v2 import ...")
print(f"  2. change the parquet read to: targets_features_split_v2.parquet")
print(f"  3. rerun: python imbalance_fix_v2.py")

del df, out
gc.collect()
