import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
import json
import gc
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier
from feature_cols import model_b_feature_cols

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / '1_download_data' / 'cleansed' / 'modeling_dataset.parquet'
OUTPUT_PATH = BASE_DIR / 'scripts' / 'model_b_tuning_results.json'

cancel_col = 'is_cancelled'
delay_col  = 'is_delayed'
features_b = model_b_feature_cols

# load splits one at a time 
def load_split(split_name, cols):
    return pq.read_table(
        DATA_PATH,
        filters=[('split', '=', split_name)],
        columns=cols
    ).to_pandas()

meta_b = features_b + [delay_col, cancel_col, 'target']

# model B (non-cancelled flights)
# train set
train_df = load_split('train', meta_b)
x_train  = train_df.loc[train_df[cancel_col] == 0, features_b]
y_train  = train_df.loc[train_df[cancel_col] == 0, delay_col].astype(int)
del train_df; gc.collect()

# validation set
val_df      = load_split('val', meta_b)
x_val_b     = val_df.loc[val_df[cancel_col] == 0, features_b]
y_val_b     = val_df.loc[val_df[cancel_col] == 0, delay_col].astype(int)
x_val_full  = val_df[features_b] # used for cascade threshold sweep 
y_val_multi = val_df['target'].astype(int)
del val_df; gc.collect()

# negative/positive class ratio ---> scale_pos_weight
base_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f'train: {len(x_train):,} rows, delay rate: {y_train.mean()*100:.1f}%')
print(f'base_ratio: {base_ratio:.2f}')


def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 500, 2000, step=100),
        'max_depth':        trial.suggest_int('max_depth', 4, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 50, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma':            trial.suggest_float('gamma', 1e-6, 1.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
        'max_delta_step':   trial.suggest_int('max_delta_step', 0, 6),
        # scale_pos_weight multiplied by a tunable factor so optuna can search around the base ratio
        'scale_pos_weight': base_ratio * trial.suggest_float('spw_mult', 0.5, 3.0),
    }

    model = XGBClassifier(
        **params,
        objective='binary:logistic',
        early_stopping_rounds=50,  
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_val_b, y_val_b)],
        verbose=False,
    )

    prauc = average_precision_score(y_val_b, model.predict_proba(x_val_b)[:, 1])

    # goal: to get best macro F1 score
    p_val    = model.predict_proba(x_val_full)[:, 1]
    best_mf1 = -1.0
    for tb in np.linspace(0.10, 0.60, 26):
        mf1 = f1_score(y_val_multi, (p_val >= tb).astype(int), average='macro')
        if mf1 > best_mf1:
            best_mf1 = mf1
            
    # picks best metrics + iteration 
    trial.set_user_attr('prauc', round(prauc, 4))
    trial.set_user_attr('best_mf1', round(best_mf1, 4))
    trial.set_user_attr('best_iteration', model.best_iteration)

    # optimise a weighted combo so we don't sacrifice one metric for the other
    return 0.5 * prauc + 0.5 * best_mf1

# using optuna for 25 different combos 
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=25, show_progress_bar=True)

# convert spw_mult back to actual scale_pos_weight for xgboost
best_params = study.best_trial.params
best_params['scale_pos_weight'] = base_ratio * best_params.pop('spw_mult')

print(f'\nbest trial: #{study.best_trial.number}')
print(f'  PR-AUC:   {study.best_trial.user_attrs["prauc"]}')
print(f'  best_mf1: {study.best_trial.user_attrs["best_mf1"]}')

results = {
    'best_params':    best_params,
    'best_score':     study.best_value,
    'prauc':          study.best_trial.user_attrs['prauc'],
    'best_mf1':       study.best_trial.user_attrs['best_mf1'],
    'best_iteration': study.best_trial.user_attrs['best_iteration'],
}
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f'saved: {OUTPUT_PATH}')
































