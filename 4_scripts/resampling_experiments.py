import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier
from flight_weather_setup import load_data, get_splits

df = load_data()
(X_train_a, y_train_a, X_val_a, y_val_a, X_test_a, y_test_a,
 X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b,
 y_val_true, y_test_true, features) = get_splits(df)

print(f'model A cancel rate: {y_train_a.mean()*100:.2f}%')
print(f'model B delay rate:  {y_train_b.mean()*100:.2f}%')

# model A is way too imbalanced (trying different resampling combos: SMOTE, undersampling)
model_a_configs = [
    ('undersample_1_1', None,
     RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
    ('undersample_1_5', None,
     RandomUnderSampler(sampling_strategy=0.2, random_state=42)),
    ('smote_hybrid',
     SMOTE(sampling_strategy=0.3, random_state=42), # oversampling of cancelled class (up to 30%)
     RandomUnderSampler(sampling_strategy=0.7, random_state=42)),
]

trained_a = {}
for name, over, under in model_a_configs:
    X_res, y_res = X_train_a.copy(), y_train_a.copy()
    if over is not None:
        X_res, y_res = over.fit_resample(X_res, y_res)
    X_res, y_res = under.fit_resample(X_res, y_res)
    print(f'\n{name}: {len(X_res):,} rows, cancel rate: {y_res.mean()*100:.1f}%')

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        objective='binary:logistic', random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_res, y_res, eval_set=[(X_val_a, y_val_a)], verbose=False)
    prauc = average_precision_score(y_val_a, model.predict_proba(X_val_a)[:, 1])
    print(f'  PR-AUC: {prauc:.4f}')
    trained_a[name] = {'model': model, 'prauc': prauc}


# Trying undersampling for model b to help with class imbalance
model_b_configs = [
    ('no_resample',     None),
    ('undersample_1_1', RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
    ('undersample_1_2', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
]

trained_b = {}
for name, sampler in model_b_configs:
    X_res, y_res = X_train_b.copy(), y_train_b.copy()
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_res, y_res)
    print(f'\n{name}: {len(X_res):,} rows, delay rate: {y_res.mean()*100:.1f}%')

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        objective='binary:logistic', random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_res, y_res, eval_set=[(X_val_b, y_val_b)], verbose=False)
    prauc = average_precision_score(y_val_b, model.predict_proba(X_val_b)[:, 1])
    print(f'  PR-AUC: {prauc:.4f}')
    trained_b[name] = {'model': model, 'prauc': prauc}


def cascade_predict(X, model_a, model_b, thresh_a=0.5, thresh_b=0.5):
    prob_cancel = model_a.predict_proba(X)[:, 1]
    pred = np.zeros(len(X), dtype=int)
    cancelled_mask = prob_cancel >= thresh_a
    pred[cancelled_mask] = 2
    not_cancelled = ~cancelled_mask
    if not_cancelled.sum() > 0:
        prob_delay = model_b.predict_proba(X[not_cancelled])[:, 1]
        pred[not_cancelled] = np.where(prob_delay >= thresh_b, 1, 0)
    return pred

# try every combo of model A + model B across threshold pairs
results = []
for a_name, a_info in trained_a.items():
    for b_name, b_info in trained_b.items():
        for ta in np.arange(0.1, 0.6, 0.05):
            for tb in np.arange(0.1, 0.6, 0.05):
                preds = cascade_predict(X_val_a, a_info['model'], b_info['model'], ta, tb)
                mf1 = f1_score(y_val_true, preds, average='macro')
                cancel_recall = f1_score(y_val_true, preds, average=None, labels=[2])[0]
                delay_recall  = f1_score(y_val_true, preds, average=None, labels=[1])[0]
                results.append({
                    'model_a': a_name, 'model_b': b_name,
                    'thresh_a': round(ta, 2), 'thresh_b': round(tb, 2),
                    'macro_f1': round(mf1, 4),
                    'cancel_recall': round(cancel_recall, 4),
                    'delay_recall': round(delay_recall, 4),
                })

results_df = pd.DataFrame(results).sort_values('macro_f1', ascending=False)

print('\ntop 5 by macro F1:')
print(results_df.head())

# best cancel recall without tanking overall F1
print('\nbest cancel recall (macro F1 > 0.35):')
filtered = results_df[results_df['macro_f1'] > 0.35]
if len(filtered) > 0:
    print(filtered.sort_values('cancel_recall', ascending=False).head())