import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path('../1_download_data/cleansed/flight_weather_merged.parquet')

# features from EDA 
FEATURES = [
    'Airline', 'Origin', 'Month', 'DayOfWeek', 'dep_hour', 'Distance',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos',
    'tmpf', 'vsby', 'sknt', 'p01i', 'relh', 'gust'
]

def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    df['Year'] = df['FlightDate'].dt.year

    # dropping 2020 because COVID anomaly 
    df = df[df['Year'] != 2020].copy()

    # build targets
    df['is_cancelled'] = df['Cancelled'].astype(int)
    df['DepDelayMinutes'] = pd.to_numeric(
        df.get('DepDelayMinutes', 0), errors='coerce'
    ).fillna(0)
    # a flight is delayed if it's not cancelled and left 15+ min late
    df['is_delayed'] = ((df['is_cancelled'] == 0) & (df['DepDelayMinutes'] >= 15)).astype(int)
    # 3-class target: 0 = on time, 1 = delayed, 2 = cancelled
    df['target'] = np.where(df['is_cancelled'] == 1, 2,
                   np.where(df['is_delayed'] == 1, 1, 0))

    # temporal split: (and train: 2018/2019, val: 2021, test:2022)
    df['split'] = 'train'
    df.loc[df['Year'] == 2021, 'split'] = 'val'
    df.loc[df['Year'] == 2022, 'split'] = 'test'

    return df

def get_splits(df):
    features = [f for f in FEATURES if f in df.columns]

    train = df[df['split'] == 'train']
    val   = df[df['split'] == 'val']
    test  = df[df['split'] == 'test']

    # model A: all flights, predicting cancellation
    X_train_a = train[features]
    y_train_a = train['is_cancelled'].astype(int)
    X_val_a   = val[features]
    y_val_a   = val['is_cancelled'].astype(int)
    X_test_a  = test[features]
    y_test_a  = test['is_cancelled'].astype(int)

    # model B: non-cancelled only, predicting delay
    train_b   = train[train['is_cancelled'] == 0]
    val_b     = val[val['is_cancelled'] == 0]
    test_b    = test[test['is_cancelled'] == 0]
    X_train_b = train_b[features]
    y_train_b = train_b['is_delayed'].astype(int)
    X_val_b   = val_b[features]
    y_val_b   = val_b['is_delayed'].astype(int)
    X_test_b  = test_b[features]
    y_test_b  = test_b['is_delayed'].astype(int)

    # also return the full val/test targets for cascade evaluation
    y_val_true  = val['target'].astype(int).values
    y_test_true = test['target'].astype(int).values

    return (
        X_train_a, y_train_a, X_val_a, y_val_a, X_test_a, y_test_a,
        X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b,
        y_val_true, y_test_true, features
    )