import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH  = BASE_DIR / '1_download_data' / 'cleansed' / 'final_flights_model_dataset.parquet'
OUTPUT_PATH = BASE_DIR / '1_download_data' / 'cleansed' / 'modeling_dataset.parquet'
COLS_PATH   = BASE_DIR / 'scripts' / 'feature_cols.py'

print('Input: ', INPUT_PATH)
print('Output:', OUTPUT_PATH)

df = pd.read_parquet(INPUT_PATH)


# targets

df['is_cancelled'] = df['Cancelled'].astype(int)
df['is_delayed']   = ((df['DepDelay'] >= 15) & (df['is_cancelled'] == 0)).astype(int)
df['is_on_time']   = ((df['is_cancelled'] == 0) & (df['is_delayed'] == 0)).astype(int)

# 0=on_time, 1=delayed, 2=cancelled
df['target'] = np.select(
    [df['is_cancelled'] == 1, df['is_delayed'] == 1],
    [2, 1],
    default=0
)

print(f'cancel rate: {df["is_cancelled"].mean():.4f}, delay rate: {df["is_delayed"].mean():.4f}')


# temporal split (2018-2019 train, 2021 val, 2022 test)

yr = df['FlightDate'].dt.year
df['split'] = np.select(
    [yr.isin([2018, 2019]), yr == 2021, yr == 2022],
    ['train', 'val', 'test'],
    default='drop'
)
df = df[df['split'] != 'drop'].copy()

for s in ['train', 'val', 'test']:
    print(f'  {s}: {(df["split"] == s).sum():,}')


# cyclical time features

month = df['Month'] if 'Month' in df.columns else df['FlightDate'].dt.month
dow   = df['DayOfWeek'] if 'DayOfWeek' in df.columns else df['FlightDate'].dt.dayofweek

df['month_sin'] = np.sin(2 * np.pi * month / 12)
df['month_cos'] = np.cos(2 * np.pi * month / 12)
df['dow_sin']   = np.sin(2 * np.pi * dow / 7)
df['dow_cos']   = np.cos(2 * np.pi * dow / 7)

# time of day
df['is_early_morning'] = df['dep_hour'].isin([5, 6, 7]).astype(int)
df['is_evening']       = (df['dep_hour'] >= 18).astype(int)

# seasonal
df['is_summer']         = month.isin([6, 7, 8]).astype(int)
df['is_holiday_season'] = (month.isin([11, 12]) | (month == 1)).astype(int)

# weather thresholds
df['high_wind']       = (df['sknt'].fillna(0) > 20).astype(int)
df['low_visibility']  = (df['vsby'].fillna(10) < 3).astype(int)
df['precip_light']    = ((df['p01i'].fillna(0) > 0) & (df['p01i'] <= 0.1)).astype(int)
df['precip_moderate'] = ((df['p01i'].fillna(0) > 0.1) & (df['p01i'] <= 0.5)).astype(int)
df['precip_heavy']    = (df['p01i'].fillna(0) > 0.5).astype(int)

# composite weather severity (0-1 scale)
df['weather_severity'] = (
    df['sknt'].fillna(0) / 30 +
    df['gust'].fillna(0) / 50 +
    (1 - df['vsby'].fillna(10) / 10) +
    df['p01i'].fillna(0).clip(upper=2) / 2
) / 4

# fill weather NaNs before saving
df['gust'] = df['gust'].fillna(0)
df['sknt'] = df['sknt'].fillna(0)
df['p01i'] = df['p01i'].fillna(0)
df['vsby'] = df['vsby'].fillna(10)
df['tmpf'] = df['tmpf'].fillna(df['tmpf'].median())
df['relh'] = df['relh'].fillna(df['relh'].median())

# encode categoricals
ws_map = {'airport_hour_match': 0, 'region_hour_match': 1, 'still_missing': 2}
df['weather_source_enc'] = df['weather_source'].map(ws_map).fillna(2).astype(int)

le_air  = LabelEncoder()
le_orig = LabelEncoder()
le_reg  = LabelEncoder()
df['Airline_enc'] = le_air.fit_transform(df['Airline'].astype(str))
df['Origin_enc']  = le_orig.fit_transform(df['Origin'].astype(str))
df['region_enc']  = le_reg.fit_transform(df['region'].astype(str).fillna('Unknown'))


# target encoding (smoothed, from training data only to prevent leakage)

is_train  = df['split'] == 'train'
M_AIR     = 10_000
M_APT     = 5_000
global_cr = df.loc[is_train, 'is_cancelled'].mean()
global_dr = df.loc[is_train, 'is_delayed'].mean()

# airline rates
agg_air = df.loc[is_train].groupby('Airline_enc').agg(
    n=('is_cancelled', 'size'),
    cs=('is_cancelled', 'sum'),
    ds=('is_delayed', 'sum'),
).reset_index()
agg_air['airline_cancel_rate'] = (agg_air['cs'] + M_AIR * global_cr) / (agg_air['n'] + M_AIR)
agg_air['airline_delay_rate']  = (agg_air['ds'] + M_AIR * global_dr) / (agg_air['n'] + M_AIR)

df = df.merge(agg_air[['Airline_enc', 'airline_cancel_rate', 'airline_delay_rate']], on='Airline_enc', how='left')
df['airline_cancel_rate'] = df['airline_cancel_rate'].fillna(global_cr)
df['airline_delay_rate']  = df['airline_delay_rate'].fillna(global_dr)

# airport rates
agg_apt = df.loc[is_train].groupby('Origin_enc').agg(
    n=('is_cancelled', 'size'),
    cs=('is_cancelled', 'sum'),
    ds=('is_delayed', 'sum'),
).reset_index()
agg_apt['airport_cancel_rate'] = (agg_apt['cs'] + M_APT * global_cr) / (agg_apt['n'] + M_APT)
agg_apt['airport_delay_rate']  = (agg_apt['ds'] + M_APT * global_dr) / (agg_apt['n'] + M_APT)

df = df.merge(agg_apt[['Origin_enc', 'airport_cancel_rate', 'airport_delay_rate']], on='Origin_enc', how='left')
df['airport_cancel_rate'] = df['airport_cancel_rate'].fillna(global_cr)
df['airport_delay_rate']  = df['airport_delay_rate'].fillna(global_dr)


# lag features (1-day per airport)

df['_date'] = df['FlightDate'].dt.normalize()

daily = df.loc[is_train].groupby(['Origin', '_date']).agg(
    n=('is_cancelled', 'size'),
    nc=('is_cancelled', 'sum'),
    nd=('is_delayed', 'sum'),
).reset_index()
daily['cr']    = daily['nc'] / daily['n']
daily['dr']    = daily['nd'] / daily['n']
daily['_date'] = pd.to_datetime(daily['_date'])
daily = daily.sort_values(['Origin', '_date'])

daily['lag1_cancel_rate']    = daily.groupby('Origin')['cr'].shift(1)
daily['lag1_delay_rate']     = daily.groupby('Origin')['dr'].shift(1)
daily['lag1_volume']         = daily.groupby('Origin')['n'].shift(1)
daily['cancelled_yesterday'] = (daily.groupby('Origin')['nc'].shift(1) > 0).astype(float)

# shift forward 1 day (yesterday's stats)
lag = daily[['Origin', '_date', 'lag1_cancel_rate', 'lag1_delay_rate', 'lag1_volume', 'cancelled_yesterday']].copy()
lag['_date'] = lag['_date'] + pd.Timedelta(days=1)

df = df.merge(lag, on=['Origin', '_date'], how='left')
df['lag1_cancel_rate']    = df['lag1_cancel_rate'].fillna(0)
df['lag1_delay_rate']     = df['lag1_delay_rate'].fillna(0)
df['lag1_volume']         = df['lag1_volume'].fillna(df['lag1_volume'].median())
df['cancelled_yesterday'] = df['cancelled_yesterday'].fillna(0)

# hourly congestion
hourly = df.loc[is_train].groupby(['Origin', '_date', 'dep_hour']).size().reset_index(name='hourly_flights')
df = df.merge(hourly, on=['Origin', '_date', 'dep_hour'], how='left')
df['hourly_flights'] = df['hourly_flights'].fillna(1)


# airport infrastructure features (from OPSNET and runways)

df['airport_operations']  = df['airport_operations'].fillna(0) if 'airport_operations' in df.columns else 0
df['has_opsnet']          = (df['airport_operations'] > 0).astype(int)
df['num_runways']         = df['num_runways'].fillna(df['num_runways'].median()) if 'num_runways' in df.columns else 3
df['max_runway_length_ft']= df['max_runway_length_ft'].fillna(df['max_runway_length_ft'].median()) if 'max_runway_length_ft' in df.columns else 8000
df['ops_per_runway']      = (df['airport_operations'] / df['num_runways'].clip(lower=1)).fillna(0)
df['airport_elevation']   = df['airport_elevation'].fillna(df['airport_elevation'].median()) if 'airport_elevation' in df.columns else 0


# delay cause features: lagged by 1 month to prevent leakage (flight in march gets feb's delay stats)

delay_cols = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
have_delay = [c for c in delay_cols if c in df.columns]

if have_delay:
    df['_yr'] = df['FlightDate'].dt.year
    df['_mo'] = df['FlightDate'].dt.month

    # get unique airport-year-month delay stats from training data only
    lkup = df.loc[is_train].groupby(['Origin', '_yr', '_mo']).agg(
        {c: 'first' for c in have_delay}
    ).reset_index()

    # shift forward 1 month so jan data is available for feb flights
    lkup['_merge_yr'] = lkup['_yr']
    lkup['_merge_mo'] = lkup['_mo'] + 1
    # december -> january rollover
    rollover = lkup['_merge_mo'] > 12
    lkup.loc[rollover, '_merge_mo'] = 1
    lkup.loc[rollover, '_merge_yr'] += 1

    renames = {c: f'lag1m_{c}' for c in have_delay}
    lkup = lkup.rename(columns=renames)
    lkup = lkup[['Origin', '_merge_yr', '_merge_mo'] + list(renames.values())]

    df = df.merge(lkup, left_on=['Origin', '_yr', '_mo'], right_on=['Origin', '_merge_yr', '_merge_mo'], how='left')

    for c in renames.values():
        df[c] = df[c].fillna(0)

    df.drop(columns=['_merge_yr', '_merge_mo', '_yr', '_mo'], errors='ignore', inplace=True)
else:
    for c in delay_cols:
        df[f'lag1m_{c}'] = 0


# select final columns and save

meta = ['FlightDate', 'split', 'target', 'is_cancelled', 'is_delayed', 'is_on_time']

features = [
    'dep_hour', 'Distance',
    'tmpf', 'vsby', 'sknt', 'p01i', 'relh', 'gust',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'is_early_morning', 'is_evening',
    'is_summer', 'is_holiday_season',
    'high_wind', 'low_visibility', 'precip_light', 'precip_moderate', 'precip_heavy',
    'weather_severity', 'weather_source_enc',
    'Airline_enc', 'Origin_enc', 'region_enc',
    'airline_delay_rate', 'airline_cancel_rate', 'airport_delay_rate', 'airport_cancel_rate',
    'lag1_delay_rate', 'lag1_cancel_rate', 'lag1_volume', 'cancelled_yesterday',
    'hourly_flights',
    'airport_operations', 'has_opsnet', 'ops_per_runway',
    'num_runways', 'max_runway_length_ft', 'airport_elevation',
    'lag1m_carrier_delay', 'lag1m_weather_delay', 'lag1m_nas_delay',
    'lag1m_late_aircraft_delay', 'lag1m_security_delay',
]

features = [c for c in features if c in df.columns]

out = df[meta + features].copy()
out.to_parquet(OUTPUT_PATH, index=False)
print(f'saved: {OUTPUT_PATH} ({len(out):,} rows, {len(features)} features)')


# write feature_cols.py

with open(COLS_PATH, 'w') as f:
    f.write('# feature_cols.py\n\n')
    f.write(f'model_a_feature_cols = {features}\n\n')
    f.write(f'model_b_feature_cols = {features}\n')

print(f'saved: {COLS_PATH}')
































