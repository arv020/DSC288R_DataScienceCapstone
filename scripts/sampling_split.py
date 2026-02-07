# import gdown
# import pandas as pd
# from pathlib import Path
# import numpy as np
# from sklearn.model_selection import train_test_split

# MODEL_READY_DIR = Path("../data/model_ready")
# MODEL_READY_FILE = MODEL_READY_DIR / "flights_model_ready.parquet"


# # Destination folder
# output_dir = Path("data/sampled_splits")
# output_dir.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

# df = pd.read_parquet(MODEL_READY_FILE)
# df.shape

# state_to_region = {
#     # Northeast
#     "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
#     "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
#     "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",

#     # Midwest
#     "IL": "Midwest", "IN": "Midwest", "MI": "Midwest",
#     "OH": "Midwest", "WI": "Midwest", "IA": "Midwest",
#     "KS": "Midwest", "MN": "Midwest", "MO": "Midwest",
#     "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",

#     # South
#     "DE": "South", "FL": "South", "GA": "South",
#     "MD": "South", "NC": "South", "SC": "South",
#     "VA": "South", "WV": "South", "AL": "South",
#     "KY": "South", "MS": "South", "TN": "South",
#     "AR": "South", "LA": "South", "OK": "South",
#     "TX": "South",

#     # West
#     "AZ": "West", "CO": "West", "ID": "West",
#     "MT": "West", "NV": "West", "NM": "West",
#     "UT": "West", "WY": "West", "AK": "West",
#     "CA": "West", "HI": "West", "OR": "West",
#     "WA": "West"
# }
# df["OriginRegion"] = df["OriginState"].map(state_to_region)

# ## Add additional features/columns here to repopulate train/test set





# # -----------------------------
# # CONFIG
# # -----------------------------
# TOTAL_SAMPLE = 700_000          # target total rows
# MAX_ORIGIN_FRAC = 0.01          # no single origin >5%
# RANDOM_STATE = 42
# MAX_REGION_FRAC = 0.30   # no single region >30% of sample


# # -----------------------------
# # PREP
# # -----------------------------
# df = df.copy()
# df["FlightDate"] = pd.to_datetime(df["FlightDate"])
# df["month"] = df["FlightDate"].dt.to_period("M").astype(str)

# # -----------------------------
# # STEP 1: BALANCE MONTHS
# # -----------------------------
# n_months = df["month"].nunique()
# rows_per_month = TOTAL_SAMPLE // n_months

# month_samples = []
# for m, g in df.groupby("month"):
#     month_samples.append(
#         g.sample(
#             n=min(len(g), rows_per_month),
#             random_state=RANDOM_STATE
#         )
#     )

# month_balanced = pd.concat(month_samples, ignore_index=True)

# # -----------------------------
# # STEP 2: CAP ORIGIN DOMINANCE
# # -----------------------------
# origin_cap = int(len(month_balanced) * MAX_ORIGIN_FRAC)

# origin_balanced = (
#     month_balanced
#     .groupby("Origin", group_keys=False)
#     .apply(
#         lambda x: x.sample(
#             n=min(len(x), origin_cap),
#             random_state=RANDOM_STATE
#         )
#     )
# )

# # -----------------------------
# # STEP 3: CAP REGION DOMINANCE
# # -----------------------------
# region_cap = int(len(origin_balanced) * MAX_REGION_FRAC)

# region_balanced = (
#     origin_balanced
#     .groupby("OriginRegion", group_keys=False)
#     .apply(
#         lambda x: x.sample(
#             n=min(len(x), region_cap),
#             random_state=RANDOM_STATE
#         )
#     )
# )


# # -----------------------------
# # STEP 4: STRATIFY TARGET
# # -----------------------------
# n_classes = origin_balanced["target"].nunique()
# rows_per_class = len(region_balanced) // n_classes

# final_sample = (
#     region_balanced
#     .groupby("target", group_keys=False)
#     .apply(
#         lambda x: x.sample(
#             n=min(len(x), rows_per_class),
#             random_state=RANDOM_STATE
#         )
#     )
# )


# print("Final rows:", len(final_sample))

# print("\nTarget distribution:")
# print(final_sample["target"].value_counts(normalize=True))

# print("\nRegion distribution:")
# print(final_sample["OriginRegion"].value_counts(normalize=True))

# print("\nTop origins:")
# print(final_sample["Origin"].value_counts(normalize=True).head())

# print("\nMonth distribution:")
# print(final_sample["month"].value_counts(normalize=True).sort_index())


# # -----------------------------
# # TRAIN / VAL / TEST SPLIT
# # -----------------------------
# final_sample = final_sample.sort_values("FlightDate")

# n = len(final_sample)
# train_end = int(n * 0.70)
# val_end = int(n * 0.85)

# train_df = final_sample.iloc[:train_end]
# val_df   = final_sample.iloc[train_end:val_end]
# test_df  = final_sample.iloc[val_end:]

# # -----------------------------
# # CHECKS
# # -----------------------------
# print("Train:", train_df.shape)
# print("Val:", val_df.shape)
# print("Test:", test_df.shape)

# print("\nTrain target distribution:")
# print(train_df["target"].value_counts(normalize=True))

# print("\nTest target distribution:")
# print(test_df["target"].value_counts(normalize=True))

# print("\nDate ranges:")
# print("Train:", train_df["FlightDate"].min(), "→", train_df["FlightDate"].max())
# print("Test :", test_df["FlightDate"].min(), "→", test_df["FlightDate"].max())


# train_df.to_parquet(output_dir / "train.parquet", index=False)
# val_df.to_parquet(output_dir / "val.parquet", index=False)
# test_df.to_parquet(output_dir / "test.parquet", index=False)


# Another thing to consider for RF, is to sample data to not train on entire dataset.

# Due to the scale of the dataset (~29M flights), we applied a multi-stage sampling strategy. 

# Data were first sampled evenly across months to retain seasonal patterns. Next, the contribution of any single origin airport was capped at 5% to prevent dominance by major hubs. Finally, the sample was stratified by the target variable (on-time, delayed, cancelled) on a best-effort basis to improve class balance. This approach maintains temporal, geographic, and outcome diversity while producing a manageable dataset for baseline modeling.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# # Destination folder
output_dir = Path("data/sampled_splits")
output_dir.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

def sample_and_split(
    df,
    output_dir="../data/model_ready/sampled_splits",
    total_sample=700_000,
    max_origin_frac=0.05,
    max_region_frac=0.30,
    test_size=0.2,
    val_frac_of_train=0.25,
    random_state=42
):
    """
    Sample a dataframe with balanced months, optional origin/region caps,
    stratified by target, then split into train/val/test and save as Parquet.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must include columns:
        - "FlightDate" (datetime or convertible)
        - "target"
        Optional columns:
        - "Origin"
        - "OriginState"


    Usage Example:
    train_df, val_df, test_df = sample_and_split(
    df,
    total_sample=700_000,
    max_origin_frac=0.05,
    max_region_frac=0.3,
    test_size=0.2,
    val_frac_of_train=0.25,
    output_dir="data/model_ready/sampled_splits",
    random_state=42
)
 
    """
    
    df = df.copy()
    
    # -----------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------
    required_cols = ["FlightDate", "target"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # -----------------------------
    # Ensure FlightDate is datetime
    # -----------------------------
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["month"] = df["FlightDate"].dt.to_period("M").astype(str)
    
    # -----------------------------
    # STEP 1: BALANCE MONTHS
    # -----------------------------
    n_months = df["month"].nunique()
    rows_per_month = total_sample // n_months
    month_samples = []
    for m, g in df.groupby("month"):
        month_samples.append(
            g.sample(n=min(len(g), rows_per_month), random_state=random_state)
        )
    month_balanced = pd.concat(month_samples, ignore_index=True)
    
    # -----------------------------
    # STEP 2: CAP ORIGIN DOMINANCE (optional)
    # -----------------------------
    if "Origin" in month_balanced.columns:
        origin_cap = int(len(month_balanced) * max_origin_frac)
        origin_balanced = (
            month_balanced
            .groupby("Origin", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), origin_cap), random_state=random_state))
        )
    else:
        origin_balanced = month_balanced.copy()
    
    # -----------------------------
    # STEP 2.5: CAP REGION DOMINANCE (optional)
    # -----------------------------
    if "OriginState" in origin_balanced.columns:
        # Map states to regions
        state_to_region = {
            "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
            "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
            "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
            "IL": "Midwest", "IN": "Midwest", "MI": "Midwest",
            "OH": "Midwest", "WI": "Midwest", "IA": "Midwest",
            "KS": "Midwest", "MN": "Midwest", "MO": "Midwest",
            "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
            "DE": "South", "FL": "South", "GA": "South",
            "MD": "South", "NC": "South", "SC": "South",
            "VA": "South", "WV": "South", "AL": "South",
            "KY": "South", "MS": "South", "TN": "South",
            "AR": "South", "LA": "South", "OK": "South",
            "TX": "South",
            "AZ": "West", "CO": "West", "ID": "West",
            "MT": "West", "NV": "West", "NM": "West",
            "UT": "West", "WY": "West", "AK": "West",
            "CA": "West", "HI": "West", "OR": "West",
            "WA": "West"
        }
        origin_balanced["OriginState"] = origin_balanced["OriginState"].str.upper()
        origin_balanced["OriginRegion"] = origin_balanced["OriginState"].map(state_to_region)
        
        region_cap = int(len(origin_balanced) * max_region_frac)
        region_balanced = (
            origin_balanced
            .groupby("OriginRegion", group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), region_cap), random_state=random_state))
        )
    else:
        region_balanced = origin_balanced.copy()
    
    # -----------------------------
    # STEP 3: STRATIFY TARGET
    # -----------------------------
    n_classes = region_balanced["target"].nunique()
    rows_per_class = len(region_balanced) // n_classes
    final_sample = (
        region_balanced
        .groupby("target", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), rows_per_class), random_state=random_state))
    )
    
    # -----------------------------
    # STEP 4: SPLIT TRAIN/VAL/TEST
    # -----------------------------
    train_val_df, test_df = train_test_split(
        final_sample,
        test_size=test_size,
        stratify=final_sample["target"],
        random_state=random_state
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_of_train,
        stratify=train_val_df["target"],
        random_state=random_state
    )
    
    # -----------------------------
    # STEP 5: SAVE FILES
    # -----------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    # -----------------------------
    # STEP 6: PRINT CHECKS
    # -----------------------------
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
    
    print("\nTrain target distribution:")
    print(train_df["target"].value_counts(normalize=True))
    
    print("\nTest target distribution:")
    print(test_df["target"].value_counts(normalize=True))
    
    if "OriginRegion" in final_sample.columns:
        print("\nRegion distribution:")
        print(final_sample["OriginRegion"].value_counts(normalize=True))
    
    print("\nDate ranges:")
    print("Train:", train_df["FlightDate"].min(), "→", train_df["FlightDate"].max())
    print("Test :", test_df["FlightDate"].min(), "→", test_df["FlightDate"].max())
    
    return train_df, val_df, test_df
