import optuna # library for bayesian search
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, f1_score
from column_groups import model_a_feature_cols, model_b_feature_cols
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = "targets_features_split.parquet"

def load_data():
    df = pd.read_parquet(DATA_PATH)

    # getting and splitting the data 
    train_mask = df["split"].eq("train")
    val_mask = df["split"].eq("val")
    
    # target cols
    cancel_col = "is_cancelled" 
    delay_col = "is_delayed" 
    
    return df, train_mask, val_mask, cancel_col, delay_col

# function to calc how imabalanced the dataset is
def calculate_scale_pos_weight(y):
    y = np.asarray(y).astype(int)
    return float((1 - y).sum() / max(y.sum(), 1))


# primary metric: pr-auc --> cancellations very rare
def model_a_hyperparams(trial, x_train, y_train, x_val, y_val):
    
    # calc weight of cancelled flights vs. non-cancelled
    imbalance_ratio = calculate_scale_pos_weight(y_train)

    # optuna guess ranges for each param
    param = {
        "n_estimators": 6000, 
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": imbalance_ratio,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_jobs": -1,
        "random_state": 42
    }

    model = XGBClassifier(**param)
    
    # training with early stopping 
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # grab the probabilities and calculate the pr-auc score
    preds_proba = model.predict_proba(x_val)[:, 1]
    pr_auc = average_precision_score(y_val, preds_proba)
    
    return pr_auc
# primary metric: f1 --> delays are more common
def model_b_hyperparams(trial, x_train, y_train, x_val, y_val):

    # calc ratio of delayed vs on-time flights
    imbalance_ratio = calculate_scale_pos_Weight(y_train)

    param = {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 50.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": imbalance_ratio,
        "objective": "binary:logistic",
        "n_jobs": -1,
        "random_state": 42
    }

    model = XGBClassifier(**param)
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # convert probabilities to 0 or 1 (cutoff threshold: 0.5)
    preds_proba = model.predict_proba(x_val)[:, 1]
    preds_binary = (preds_proba >= 0.5).astype(int)
    f1 = f1_score(y_val, preds_binary, zero_division=0)
    
    return f1

def main():
    df, train_mask, val_mask, cancel_col, delay_col = load_data()
    
    # prep data for model a
    x_train_a = df.loc[train_mask, model_a_feature_cols]
    y_train_a = df.loc[train_mask, cancel_col].astype(int)
    x_val_a = df.loc[val_mask, model_a_feature_cols]
    y_val_a = df.loc[val_mask, cancel_col].astype(int)
    
    # run optuna for model a
    study_a = optuna.create_study(direction="maximize", study_name="Model_A_Optimization")
    study_a.optimize(
        lambda trial: model_a_hyperparams(trial, x_train_a, y_train_a, x_val_a, y_val_a),
        n_trials=30 
    )
    
    print("Best Model A PR-AUC:", study_a.best_value)
    print("Best Model A Params:", study_a.best_params)

    # prep data for model b (only non-cancelled flights)
    train_b_mask = train_mask & (df[cancel_col] == 0)
    val_b_mask = val_mask & (df[cancel_col] == 0)
    
    x_train_b = df.loc[train_b_mask, model_b_feature_cols]
    y_train_b = df.loc[train_b_mask, delay_col].astype(int)
    x_val_b = df.loc[val_b_mask, model_b_feature_cols]
    y_val_b = df.loc[val_b_mask, delay_col].astype(int)
    
    # run optuna for model b
    study_b = optuna.create_study(direction="maximize", study_name="Model_B_Optimization")
    study_b.optimize(
        lambda trial: model_b_hyperparams(trial, x_train_b, y_train_b, x_val_b, y_val_b),
        n_trials=30
    )
    
    print("Best Model B F1-Score:", study_b.best_value)
    print("Best Model B Params:", study_b.best_params)

if __name__ == "__main__":
    main()
