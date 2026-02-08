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
    random_state=42,
    target_col="target"
):
    """
    Sample a dataframe with balanced months, optional origin/region caps,
    stratified by target, then split into train/test and return X/y splits.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    
    df = df.copy()
    
    # -----------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------
    required_cols = ["FlightDate", target_col]
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
    # STEP 2: CAP ORIGIN DOMINANCE
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
    # STEP 2.5: CAP REGION DOMINANCE
    # -----------------------------
    if "OriginState" in origin_balanced.columns:
        state_to_region = {
            "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
            "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
            "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
            "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
            "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
            "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
            "DE": "South", "FL": "South", "GA": "South", "MD": "South",
            "NC": "South", "SC": "South", "VA": "South", "WV": "South",
            "AL": "South", "KY": "South", "MS": "South", "TN": "South",
            "AR": "South", "LA": "South", "OK": "South", "TX": "South",
            "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
            "NV": "West", "NM": "West", "UT": "West", "WY": "West",
            "AK": "West", "CA": "West", "HI": "West", "OR": "West",
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
    n_classes = region_balanced[target_col].nunique()
    rows_per_class = len(region_balanced) // n_classes
    final_sample = (
        region_balanced
        .groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), rows_per_class), random_state=random_state))
    )
    
    # -----------------------------
    # STEP 4: SPLIT TRAIN/TEST
    # -----------------------------
    train_df, test_df = train_test_split(
        final_sample,
        test_size=test_size,
        stratify=final_sample[target_col],
        random_state=random_state
    )
    
    # -----------------------------
    # STEP 5: CREATE X/y
    # -----------------------------
    feature_cols = [c for c in final_sample.columns if c != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # -----------------------------
    # STEP 6: SAVE FILES (optional)
    # -----------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    # -----------------------------
    # STEP 7: PRINT CHECKS
    # -----------------------------
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
    
    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True))
    
    if "OriginRegion" in final_sample.columns:
        print("\nRegion distribution:")
        print(final_sample["OriginRegion"].value_counts(normalize=True))
    
    print("\nDate ranges:")
    print("Train:", train_df["FlightDate"].min(), "→", train_df["FlightDate"].max())
    print("Test :", test_df["FlightDate"].min(), "→", test_df["FlightDate"].max())
    
    return X_train, X_test, y_train, y_test
