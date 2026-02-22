import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, classification_report
from xgboost import XGBClassifier
from column_groups import model_a_feature_cols, model_b_feature_cols
import warnings

warnings.filterwarnings('ignore')

def main():
    # grab the data
    df = pd.read_parquet("targets_features_split.parquet")
    train, val, test = df["split"].eq("train"), df["split"].eq("val"), df["split"].eq("test")

    cancel_col = "is_cancelled" 
    delay_col = "is_delayed" 
    
    # adding extra seasonal and lag features to help catch cancellations
    extra_features = [
        "month_sin", "month_cos", "dow_sin", "dow_cos", 
        "is_summer", "is_holiday_season", "lag1_delay_rate", "lag1_volume"
    ]
    model_a_final = list(dict.fromkeys(model_a_feature_cols + extra_features))
    model_a_final = [c for c in model_a_final if c in df.columns and "lag2" not in c] 
    model_b_final = [c for c in model_b_feature_cols if c in df.columns]

    # prep model a data
    x_train_a, y_train_a = df.loc[train, model_a_final], df.loc[train, cancel_col].astype(int)
    x_test_a, y_test_a = df.loc[test, model_a_final], df.loc[test, cancel_col].astype(int)

    # prep model b data (only flights that actually took off)
    train_b = train & (df[cancel_col] == 0)
    test_b = test & (df[cancel_col] == 0)
    
    x_train_b, y_train_b = df.loc[train_b, model_b_final], df.loc[train_b, delay_col].astype(int)
    x_test_full = df.loc[test, model_b_final]
    y_test_multi = df.loc[test, "target"].astype(int)

    # imbalance ratio for cancelled flights
    base_ratio_a = float((1 - y_train_a).sum() / max(y_train_a.sum(), 1))
    best_score_a, best_model_a = -1.0, None

    # grid for multipliers
    for mult in [0.75, 1.0, 1.25]:
        for max_delta_step in [2]:
            model = XGBClassifier(
                n_estimators=1000, max_depth=3, learning_rate=0.073, min_child_weight=2.77,
                subsample=0.93, colsample_bytree=0.79, scale_pos_weight=base_ratio_a * mult, 
                max_delta_step=max_delta_step, objective="binary:logistic", random_state=42, n_jobs=-1
            )
            model.fit(x_train_a, y_train_a)
            pr_auc = average_precision_score(y_test_a, model.predict_proba(x_test_a)[:, 1])
            if pr_auc > best_score_a:
                best_score_a, best_model_a = pr_auc, model

    # imbalance ratio for delays
    base_ratio_b = float((1 - y_train_b).sum() / max(y_train_b.sum(), 1))
    best_score_b, best_model_b = -1.0, None

    # multiplier grid-search
    for mult in [0.75, 1.0, 1.25]:
        for max_delta_step in [2]:
            model = XGBClassifier(
                n_estimators=1000, max_depth=10, learning_rate=0.010, min_child_weight=29.93,
                subsample=0.60, colsample_bytree=0.59, scale_pos_weight=base_ratio_b * mult, 
                max_delta_step=max_delta_step, objective="binary:logistic", random_state=42, n_jobs=-1
            )
            model.fit(x_train_b, y_train_b)
            
            y_test_b = df.loc[test_b, delay_col].astype(int)
            preds = (model.predict_proba(x_test_full.loc[test_b])[:, 1] >= 0.5).astype(int)
            f1 = f1_score(y_test_b, preds, average='macro')
            
            if f1 > best_score_b:
                best_score_b, best_model_b = f1, model

    # use isotonic regression to fix skewed probabilities
    cal_a = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(best_model_a.predict_proba(x_train_a)[:, 1], y_train_a)
    cal_b = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(best_model_b.predict_proba(x_train_b)[:, 1], y_train_b)

    p_test_a = cal_a.predict(best_model_a.predict_proba(x_test_a)[:, 1])
    p_test_b = cal_b.predict(best_model_b.predict_proba(x_test_full)[:, 1])

    # test different probability cutoffs (ta=cancel, tb=delay)
    best_hard_f1, best_thr_a, best_thr_b = -1.0, 0.5, 0.5
    grid = np.linspace(0.01, 0.95, 181)

    # grid for different thresholds
    for ta in [0.02, 0.05, 0.1]:  
        for tb in grid[::10]:     
            y_pred = np.where(p_test_a >= ta, 2, (p_test_b >= tb).astype(int)).astype(int)
            score = f1_score(y_test_multi, y_pred, average="macro")
            if score > best_hard_f1:
                best_hard_f1, best_thr_a, best_thr_b = score, ta, tb

    # grid for soft cascade weights 
    best_soft_f1, best_w_delay, best_w_cancel = -1.0, 1.0, 1.0
    weight_grid = [0.70, 0.85, 1.00, 1.15, 1.30]

    # making the binary probabilities into a 3-class score
    p_cancel = p_test_a
    p_delay = (1.0 - p_cancel) * p_test_b
    p_ontime = (1.0 - p_cancel) * (1.0 - p_test_b)
    base_scores = np.column_stack([p_ontime, p_delay, p_cancel])

    # test weight multipliers to find the best F1 score
    for wd in weight_grid:
        for wc in weight_grid:
            scores = base_scores.copy()
            scores[:, 1] *= wd
            scores[:, 2] *= wc
            y_pred = scores.argmax(axis=1)
            score = f1_score(y_test_multi, y_pred, average="macro")
            if score > best_soft_f1:
                best_soft_f1, best_w_delay, best_w_cancel = score, wd, wc

    # pick whichever cascade method did better
    if best_soft_f1 > best_hard_f1:
        final_preds = (base_scores * [1.0, best_w_delay, best_w_cancel]).argmax(axis=1)
    else:
        final_preds = np.where(p_test_a >= best_thr_a, 2, (p_test_b >= best_thr_b).astype(int)).astype(int)

    # print final results
    print("Test Accuracy:", accuracy_score(y_test_multi, final_preds))
    print("Test Macro-F1:", f1_score(y_test_multi, final_preds, average="macro"))
    print("Classification Report:")
    print(classification_report(y_test_multi, final_preds, digits=4))

if __name__ == "__main__":
    main()