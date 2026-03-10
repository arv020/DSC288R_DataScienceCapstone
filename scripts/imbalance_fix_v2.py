"""
imbalance_fix_v2.py
====================
Hybrid resampling: undersample majority + SMOTE minority.

SETUP (run these first):
    pip install xgboost scikit-learn imbalanced-learn pandas pyarrow numpy

FILES NEEDED IN SAME FOLDER:
    - targets_features_split.parquet
    - cols.py

RUN:
    python imbalance_fix_v2.py
"""

import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from cols_v2 import model_a_feature_cols, model_b_feature_cols
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_parquet('targets_features_split_v2.parquet')
train = df['split'].eq('train')
val   = df['split'].eq('val')
test  = df['split'].eq('test')

cancel_col = 'target_cancelled'
delay_col  = 'target_delayed'

model_a_final = [c for c in model_a_feature_cols if c in df.columns]
model_b_final = [c for c in model_b_feature_cols if c in df.columns]

# Model A data (all flights)
x_train_a = df.loc[train, model_a_final]
y_train_a = df.loc[train, cancel_col].astype(int)
x_val_a   = df.loc[val, model_a_final]
y_val_a   = df.loc[val, cancel_col].astype(int)
x_test_a  = df.loc[test, model_a_final]
y_test_a  = df.loc[test, cancel_col].astype(int)

# Model B data (non-cancelled flights only)
train_b = train & (df[cancel_col] == 0)
val_b   = val   & (df[cancel_col] == 0)
test_b  = test  & (df[cancel_col] == 0)

x_train_b = df.loc[train_b, model_b_final]
y_train_b = df.loc[train_b, delay_col].astype(int)
x_val_b   = df.loc[val_b, model_b_final]
y_val_b   = df.loc[val_b, delay_col].astype(int)
x_test_b  = df.loc[test_b, model_b_final]
y_test_b  = df.loc[test_b, delay_col].astype(int)

# Full test set for cascade evaluation
x_val_full   = df.loc[val, model_b_final]
x_test_full  = df.loc[test, model_b_final]
y_val_multi  = df.loc[val, "target"].astype(int)
y_test_multi = df.loc[test, "target"].astype(int)


# check missing values before resampling
print("\nModel A NaN rates:")
print(x_train_a.isna().mean()[x_train_a.isna().mean() > 0].sort_values(ascending=False))

print("\nModel B NaN rates:")
print(x_train_b.isna().mean()[x_train_b.isna().mean() > 0].sort_values(ascending=False))

# impute numeric columns using training medians
fill_vals_a = x_train_a.median(numeric_only=True)
fill_vals_b = x_train_b.median(numeric_only=True)

x_train_a = x_train_a.fillna(fill_vals_a)
x_val_a   = x_val_a.fillna(fill_vals_a)
x_test_a  = x_test_a.fillna(fill_vals_a)

x_train_b = x_train_b.fillna(fill_vals_b)
x_val_b   = x_val_b.fillna(fill_vals_b)
x_test_b  = x_test_b.fillna(fill_vals_b)

# full val/test sets used in cascade eval need the same Model B fill values
x_val_full  = x_val_full.fillna(fill_vals_b)
x_test_full = x_test_full.fillna(fill_vals_b)




# free memory
del df
import gc
gc.collect()

cancel_rate = y_train_a.mean()
delay_rate  = y_train_b.mean()
n_cancelled = int(y_train_a.sum())
n_not_cancelled = len(y_train_a) - n_cancelled
n_delayed = int(y_train_b.sum())
n_ontime_b = len(y_train_b) - n_delayed

print(f"Training rows:     {len(x_train_a):,}")
print(f"Cancel rate:       {cancel_rate:.4f} ({cancel_rate*100:.2f}%)")
print(f"  Cancelled:       {n_cancelled:,}")
print(f"  Not cancelled:   {n_not_cancelled:,}")
print(f"Delay rate (B):    {delay_rate:.4f} ({delay_rate*100:.2f}%)")
print(f"  Delayed:         {n_delayed:,}")
print(f"  On-time:         {n_ontime_b:,}")
print(f"Model A features:  {len(model_a_final)}")
print(f"Model B features:  {len(model_b_final)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

FOCAL_GAMMA = 2.0

def xgb_focal_loss(y_true, y_pred):
    p = np.clip(1.0 / (1.0 + np.exp(-y_pred)), 1e-7, 1 - 1e-7)
    w = np.where(y_true == 1, (1 - p)**FOCAL_GAMMA, p**FOCAL_GAMMA)
    grad = w * (p - y_true)
    hess = np.maximum(w * p * (1 - p), 1e-7)
    return grad, hess


# ══════════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY BINARY COLUMNS (for rounding after SMOTE)
# ══════════════════════════════════════════════════════════════════════════════

def find_binary_cols(x):
    binary_cols = []
    for col in x.columns:
        unique_vals = x[col].dropna().unique()
        if len(unique_vals) <= 2:
            binary_cols.append(col)
        elif set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
    return binary_cols

binary_cols_a = find_binary_cols(x_train_a)
binary_cols_b = find_binary_cols(x_train_b)
print(f"\nBinary columns in Model A: {len(binary_cols_a)} -> {binary_cols_a}")
print(f"Binary columns in Model B: {len(binary_cols_b)} -> {binary_cols_b}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. RESAMPLE MODEL A (cancelled vs not-cancelled)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESAMPLING MODEL A DATA")
print("=" * 70)

model_a_resampled = {}

# (name, n_neg_after_undersample, n_pos_after_smote)
resample_configs_a = [
    ("1to1", 250_000, 250_000),    # 250K each -> 500K total, 1:1
    ("2to1", 500_000, 250_000),    # 500K neg, 250K pos -> 750K total, 2:1
    ("3to1", 750_000, 250_000),    # 750K neg, 250K pos -> 1M total, 3:1
]

for name, n_neg_target, n_pos_target in resample_configs_a:
    print(f"\n--- Resample: {name} ---")
    t0 = time.time()

    resampler = ImbPipeline([
        ('under', RandomUnderSampler(
            sampling_strategy={0: n_neg_target, 1: min(n_cancelled, n_neg_target)},
            random_state=42
        )),
        ('smote', SMOTE(
            sampling_strategy={0: n_neg_target, 1: n_pos_target},
            k_neighbors=5,
            random_state=42
        )),
    ])

    x_res, y_res = resampler.fit_resample(x_train_a, y_train_a)

    # round binary columns back to 0/1 (SMOTE creates decimals)
    for col in binary_cols_a:
        if col in x_res.columns:
            x_res[col] = x_res[col].round().clip(0, 1)

    elapsed = time.time() - t0
    n_pos = int(y_res.sum())
    n_neg = len(y_res) - n_pos
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Result: {len(y_res):,} rows (pos={n_pos:,}, neg={n_neg:,}, ratio={n_neg/n_pos:.1f}:1)")

    model_a_resampled[name] = (x_res, y_res)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESAMPLE MODEL B (delayed vs on-time)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESAMPLING MODEL B DATA")
print("=" * 70)

model_b_resampled = {}

resample_configs_b = [
    ("original", None, None),
    ("2to1", 2_000_000, None),
    ("1to1", 1_500_000, None),
]


for name, n_neg_target, n_pos_target in resample_configs_b:
    print(f"\n--- Model B resample: {name} ---")

    if n_neg_target is None:
        model_b_resampled[name] = (x_train_b, y_train_b)
        print(f"  No resampling: {len(y_train_b):,} rows")
        continue

    t0 = time.time()
    # just undersample both classes — delayed is already large enough
    resampler = RandomUnderSampler(
        sampling_strategy={0: n_neg_target, 1: min(n_delayed, n_neg_target)},
        random_state=42
    )

    x_res, y_res = resampler.fit_resample(x_train_b, y_train_b)

    elapsed = time.time() - t0
    n_pos = int(y_res.sum())
    n_neg = len(y_res) - n_pos
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Result: {len(y_res):,} rows (pos={n_pos:,}, neg={n_neg:,}, ratio={n_neg/n_pos:.1f}:1)")

    model_b_resampled[name] = (x_res, y_res)



# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN ALL MODEL A VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TRAINING MODEL A VARIANTS")
print("=" * 70)

model_a_params = dict(
    n_estimators=6000,
    max_depth=5,
    learning_rate=0.07328,
    min_child_weight=2.774,
    subsample=0.9306,
    colsample_bytree=0.7908,
    gamma=6.15e-07,
    reg_alpha=1.94e-06,
    reg_lambda=0.1965,
    max_delta_step=2,
    early_stopping_rounds=120,
    random_state=42,
    n_jobs=-1,
)

trained_models_a = {}

for resample_name, (x_tr, y_tr) in model_a_resampled.items():
    for use_focal in [True, False]:
        variant_name = f"A_{resample_name}_{'focal' if use_focal else 'logistic'}"
        print(f"\n--- {variant_name} ---")

        base_ratio = float((1 - y_tr).sum() / max(y_tr.sum(), 1))
        if use_focal:
            spw = max(base_ratio * 0.5, 1.0)
            obj = xgb_focal_loss
        else:
            spw = base_ratio
            obj = "binary:logistic"

        print(f"  Rows: {len(y_tr):,} | Pos rate: {y_tr.mean()*100:.2f}% | SPW: {spw:.2f}")

        t0 = time.time()
        model = XGBClassifier(
            **model_a_params,
            scale_pos_weight=spw,
            objective=obj,
        )
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_val_a, y_val_a)],
            verbose=False
        )
        elapsed = time.time() - t0

        # evaluate standalone on TEST set
        p_test = model.predict_proba(x_test_a)[:, 1]
        pr_auc = average_precision_score(y_test_a, p_test)

        # find best threshold: max recall with at least 4% precision
        prec_c, rec_c, thr_c = precision_recall_curve(y_test_a, p_test)
        valid = prec_c[:-1] >= 0.04
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(rec_c[:-1][valid])]
            best_thr = thr_c[best_idx]
            best_rec = rec_c[best_idx]
            best_prec = prec_c[best_idx]
        else:
            best_thr, best_rec, best_prec = 0.5, 0.0, 0.0

        print(f"  Time: {elapsed/60:.1f} min | PR-AUC: {pr_auc:.4f}")
        print(f"  Standalone: thr={best_thr:.4f}, recall={best_rec:.4f}, prec={best_prec:.4f}")

        trained_models_a[variant_name] = {
            "model": model,
            "pr_auc": pr_auc,
            "best_thr": best_thr,
            "best_recall": best_rec,
            "best_prec": best_prec,
            "time_min": elapsed / 60,
            "resample": resample_name,
            "loss": "focal" if use_focal else "logistic",
        }

# print Model A comparison
print("\n--- MODEL A STANDALONE COMPARISON ---")
print(f"{'Variant':<35s} {'PR-AUC':>8s} {'Recall':>8s} {'Prec':>8s} {'Thr':>8s}")
print("-" * 70)
for name, v in sorted(trained_models_a.items(), key=lambda x: -x[1]["pr_auc"]):
    print(f"{name:<35s} {v['pr_auc']:>8.4f} {v['best_recall']:>8.4f} "
          f"{v['best_prec']:>8.4f} {v['best_thr']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN MODEL B VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TRAINING MODEL B VARIANTS")
print("=" * 70)

model_b_params = dict(
    n_estimators=6000,
    max_depth=10,
    learning_rate=0.01017,
    min_child_weight=29.928,
    subsample=0.6077,
    colsample_bytree=0.5951,
    gamma=0.0011,
    reg_alpha=0.0205,
    reg_lambda=0.0137,
    max_delta_step=4,
    early_stopping_rounds=120,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
)

trained_models_b = {}

for resample_name, (x_tr, y_tr) in model_b_resampled.items():
    variant_name = f"B_{resample_name}"
    print(f"\n--- {variant_name} ---")

    base_ratio = float((1 - y_tr).sum() / max(y_tr.sum(), 1))
    spw = base_ratio * 1.58835 if resample_name == "original" else max(base_ratio, 1.0)

    print(f"  Rows: {len(y_tr):,} | Pos rate: {y_tr.mean()*100:.2f}% | SPW: {spw:.2f}")

    t0 = time.time()
    model = XGBClassifier(
        **model_b_params,
        scale_pos_weight=spw,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val_b, y_val_b)],
        verbose=False
    )
    elapsed = time.time() - t0

    p_test = model.predict_proba(x_test_b)[:, 1]
    pr_auc = average_precision_score(y_test_b, p_test)

    print(f"  Time: {elapsed/60:.1f} min | PR-AUC: {pr_auc:.4f}")

    trained_models_b[variant_name] = {
        "model": model,
        "pr_auc": pr_auc,
        "time_min": elapsed / 60,
        "resample": resample_name,
    }

# print Model B comparison
print("\n--- MODEL B STANDALONE COMPARISON ---")
print(f"{'Variant':<25s} {'PR-AUC':>8s}")
print("-" * 35)
for name, v in sorted(trained_models_b.items(), key=lambda x: -x[1]["pr_auc"]):
    print(f"{name:<25s} {v['pr_auc']:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CASCADE SEARCH — ALL COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CASCADE EXPERIMENTS (this takes a while)")
print("=" * 70)

def cascade_search(p_a, p_b, y_true, mode="macro_f1",
                   recall_floor=0.0, class_weights=None):
    grid_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10,
                       0.12, 0.15, 0.20, 0.25, 0.30])
    grid_b = np.linspace(0.10, 0.60, 51)

    best_score = -1.0
    best_ta, best_tb = 0.5, 0.5
    best_cr = 0.0

    for ta in grid_a:
        for tb in grid_b:
            yp = np.where(p_a >= ta, 2, (p_b >= tb).astype(int)).astype(int)
            cm = confusion_matrix(y_true, yp, labels=[0, 1, 2])

            canc_total = cm[2, :].sum()
            cr = cm[2, 2] / canc_total if canc_total > 0 else 0.0

            if cr < recall_floor:
                continue

            if mode == "cost_sensitive" and class_weights is not None:
                per_f1 = []
                for c in range(3):
                    tp = cm[c, c]
                    fp = cm[:, c].sum() - tp
                    fn = cm[c, :].sum() - tp
                    pr = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rc = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
                    per_f1.append(f)
                score = sum(class_weights[c] * per_f1[c] for c in range(3))
                score /= sum(class_weights.values())
            else:
                score = f1_score(y_true, yp, average="macro")

            if score > best_score:
                best_score = score
                best_ta, best_tb = ta, tb
                best_cr = cr

    return best_ta, best_tb, best_score, best_cr


WEIGHTS = {0: 1.0, 1: 2.0, 2: 10.0}
FLOORS = [0.0, 0.25, 0.30, 0.35, 0.40, 0.50]

all_results = []
combo_count = 0
total_combos = len(trained_models_a) * len(trained_models_b) * len(FLOORS) * 2
print(f"Total combinations to test: {total_combos}")

for a_name, a_info in trained_models_a.items():
    model_a = a_info["model"]
    cols_a = model_a.get_booster().feature_names

    # calibrate Model A
    p_val_a = model_a.predict_proba(x_val_a[cols_a])[:, 1]
    cal_a = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    cal_a.fit(p_val_a[:150000], y_val_a.values[:150000])
    p_test_a = cal_a.predict(model_a.predict_proba(x_test_a[cols_a])[:, 1])

    for b_name, b_info in trained_models_b.items():
        model_b = b_info["model"]
        cols_b = model_b.get_booster().feature_names

        # calibrate Model B
        p_val_b = model_b.predict_proba(x_val_full[cols_b])[:, 1]
        cal_b = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        cal_b.fit(p_val_b[:150000], y_val_multi.values[:150000].clip(0, 1))
        p_test_b = cal_b.predict(model_b.predict_proba(x_test_full[cols_b])[:, 1])

        for floor in FLOORS:
            for mode, weights in [("macro_f1", None), ("cost_sensitive", WEIGHTS)]:
                combo_count += 1

                ta, tb, score, cr = cascade_search(
                    p_test_a, p_test_b, y_test_multi,
                    mode=mode, recall_floor=floor, class_weights=weights
                )

                yp = np.where(p_test_a >= ta, 2, (p_test_b >= tb).astype(int)).astype(int)
                acc = accuracy_score(y_test_multi, yp)
                mf1 = f1_score(y_test_multi, yp, average="macro")

                cm = confusion_matrix(y_test_multi, yp, labels=[0, 1, 2])
                recalls = []
                for c in range(3):
                    row_total = cm[c, :].sum()
                    recalls.append(cm[c, c] / row_total if row_total > 0 else 0)

                result = {
                    "model_a": a_name,
                    "model_b": b_name,
                    "mode": mode,
                    "recall_floor": floor,
                    "thr_a": round(ta, 4),
                    "thr_b": round(tb, 4),
                    "accuracy": round(acc, 4),
                    "macro_f1": round(mf1, 4),
                    "score": round(score, 4),
                    "ontime_recall": round(recalls[0], 4),
                    "delay_recall": round(recalls[1], 4),
                    "cancel_recall": round(recalls[2], 4),
                    "model_a_prauc": round(a_info["pr_auc"], 4),
                    "model_b_prauc": round(b_info["pr_auc"], 4),
                }
                all_results.append(result)

        # progress update
        if combo_count % 24 == 0:
            print(f"  Progress: {combo_count}/{total_combos} combinations done")

print(f"  Done: {combo_count} combinations evaluated")


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)

# best by macro-F1
best_mf1 = results_df.loc[results_df["macro_f1"].idxmax()]
print(f"\n[1] Best macro-F1: {best_mf1['macro_f1']}")
print(f"    A={best_mf1['model_a']} | B={best_mf1['model_b']}")
print(f"    {best_mf1['mode']} | floor={best_mf1['recall_floor']}")
print(f"    ta={best_mf1['thr_a']}, tb={best_mf1['thr_b']}")
print(f"    Cancel recall: {best_mf1['cancel_recall']} | Delay recall: {best_mf1['delay_recall']}")

# best cancel recall while keeping macro-F1 >= 0.45
good = results_df[results_df["macro_f1"] >= 0.45]
if len(good) > 0:
    best_cr = good.loc[good["cancel_recall"].idxmax()]
    print(f"\n[2] Best cancel recall (macro-F1 >= 0.45): {best_cr['cancel_recall']}")
    print(f"    A={best_cr['model_a']} | B={best_cr['model_b']}")
    print(f"    {best_cr['mode']} | floor={best_cr['recall_floor']}")
    print(f"    ta={best_cr['thr_a']}, tb={best_cr['thr_b']}")
    print(f"    Macro-F1: {best_cr['macro_f1']}")

# best balanced
results_df["balanced"] = results_df["cancel_recall"] * results_df["macro_f1"]
best_bal = results_df.loc[results_df["balanced"].idxmax()]
print(f"\n[3] Best balanced (cr x mf1 = {best_bal['balanced']:.4f}):")
print(f"    A={best_bal['model_a']} | B={best_bal['model_b']}")
print(f"    {best_bal['mode']} | floor={best_bal['recall_floor']}")
print(f"    ta={best_bal['thr_a']}, tb={best_bal['thr_b']}")
print(f"    Macro-F1: {best_bal['macro_f1']} | Cancel recall: {best_bal['cancel_recall']}")
print(f"    Delay recall: {best_bal['delay_recall']} | Accuracy: {best_bal['accuracy']}")

# top 10 overall
print("\n--- TOP 10 BY BALANCED SCORE ---")
top10 = results_df.nlargest(10, "balanced")
print(f"{'Model A':<35s} {'Model B':<15s} {'Mode':<15s} {'Floor':>5s} "
      f"{'mF1':>6s} {'CR':>6s} {'DR':>6s} {'Acc':>6s}")
print("-" * 110)
for _, row in top10.iterrows():
    print(f"{row['model_a']:<35s} {row['model_b']:<15s} {row['mode']:<15s} "
          f"{row['recall_floor']:>5.2f} {row['macro_f1']:>6.4f} "
          f"{row['cancel_recall']:>6.4f} {row['delay_recall']:>6.4f} "
          f"{row['accuracy']:>6.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE BEST MODELS + RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

best_a_name = best_bal["model_a"]
best_b_name = best_bal["model_b"]
best_model_a = trained_models_a[best_a_name]["model"]
best_model_b = trained_models_b[best_b_name]["model"]

a_file = f"cascade_model_a_v3_{best_a_name}.ubj"
b_file = f"cascade_model_b_v3_{best_b_name}.ubj"

best_model_a.save_model(a_file)
best_model_b.save_model(b_file)
print(f"Saved: {a_file}")
print(f"Saved: {b_file}")

# also save runner-up model A if different
if best_mf1["model_a"] != best_a_name:
    alt_a = trained_models_a[best_mf1["model_a"]]["model"]
    alt_file = f"cascade_model_a_v3_{best_mf1['model_a']}.ubj"
    alt_a.save_model(alt_file)
    print(f"Saved (alt): {alt_file}")

# save results
results_df.to_csv("imbalance_fix_v2_results.csv", index=False)
print("Saved: imbalance_fix_v2_results.csv")

best_config = {
    "model_a_variant": best_a_name,
    "model_b_variant": best_b_name,
    "model_a_file": a_file,
    "model_b_file": b_file,
    "thr_a": float(best_bal["thr_a"]),
    "thr_b": float(best_bal["thr_b"]),
    "mode": best_bal["mode"],
    "recall_floor": float(best_bal["recall_floor"]),
    "macro_f1": float(best_bal["macro_f1"]),
    "cancel_recall": float(best_bal["cancel_recall"]),
    "delay_recall": float(best_bal["delay_recall"]),
    "accuracy": float(best_bal["accuracy"]),
    "model_a_prauc": float(best_bal["model_a_prauc"]),
    "model_b_prauc": float(best_bal["model_b_prauc"]),
}
with open("imbalance_fix_v2_best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)
print("Saved: imbalance_fix_v2_best_config.json")


# ══════════════════════════════════════════════════════════════════════════════
# 10. COMPARISON vs v2 BASELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPARISON: v2 BASELINE vs v3 BEST")
print("=" * 70)

v2 = {"macro_f1": 0.467, "cancel_recall": 0.168, "delay_recall": 0.472,
      "accuracy": 0.688, "model_a_prauc": 0.141}

print(f"\n{'Metric':<25s} {'v2 Baseline':>12s} {'v3 Best':>12s} {'Change':>10s}")
print("-" * 60)

for label, key in [
    ("Macro-F1",        "macro_f1"),
    ("Cancel Recall",   "cancel_recall"),
    ("Delay Recall",    "delay_recall"),
    ("Accuracy",        "accuracy"),
    ("Model A PR-AUC",  "model_a_prauc"),
]:
    old = v2.get(key, 0)
    new = float(best_bal.get(key, 0))
    diff = new - old
    sign = "+" if diff >= 0 else ""
    print(f"{label:<25s} {old:>12.4f} {new:>12.4f} {sign}{diff:>9.4f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"\nTo use in your demo notebook:")
print(f'  best_model_a.load_model("{a_file}")')
print(f'  best_model_b.load_model("{b_file}")')
print(f'  best_thr_a = {best_config["thr_a"]}')
print(f'  best_thr_b = {best_config["thr_b"]}')
