import pandas as pd
import os

df = pd.read_parquet('targets_features_split.parquet')

train = df[df['split'] == 'train']
val = df[df['split'] == 'val']
test = df[df['split'] == 'test']

# sample 150k from train, stratified by target
train_150k = train.groupby('target', group_keys=False).apply(
    lambda x: x.sample(frac=150000/len(train), random_state=42)
)

os.makedirs('splits_150k', exist_ok=True)

train_150k.to_parquet('splits_150k/train.parquet', index=False)
val.to_parquet('splits_150k/val.parquet', index=False)
test.to_parquet('splits_150k/test.parquet', index=False)

print(f'train: {len(train_150k)}')
print(f'val: {len(val)}')
print(f'test: {len(test)}')