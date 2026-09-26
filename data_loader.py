"""
Data loader and preprocessing module for Sales Prediction.
Handles sourcing, audit, cleaning, validation, and train/test splitting.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Advertising.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "advertising_cleaned.csv"


def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw Advertising dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate dataset:
    - Drop redundant index columns like 'Unnamed: 0'
    - Verify column names and data types
    - Check missing values and duplicate rows
    """
    df_clean = df.copy()

    # Drop index columns if present
    drop_cols = [c for c in df_clean.columns if "Unnamed" in c or c == ""]
    if drop_cols:
        df_clean.drop(columns=drop_cols, inplace=True)

    # Standardize column names
    expected_cols = ["TV", "Radio", "Newspaper", "Sales"]
    df_clean.columns = [col.strip() for col in df_clean.columns]

    for col in expected_cols:
        if col not in df_clean.columns:
            raise ValueError(f"Missing expected column '{col}' in dataset.")

    # Check and enforce numeric datatypes
    for col in expected_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Audit checks
    null_counts = df_clean.isnull().sum()
    duplicate_count = df_clean.duplicated().sum()

    print(f"Dataset shape: {df_clean.shape}")
    print(f"Missing values:\n{null_counts}")
    print(f"Duplicate rows: {duplicate_count}")

    # Remove any potential duplicate rows
    if duplicate_count > 0:
        df_clean.drop_duplicates(inplace=True)

    return df_clean


def save_processed_data(df: pd.DataFrame, filepath=PROCESSED_DATA_PATH):
    """Save cleaned data to processed folder."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Processed dataset saved to {filepath}")


def load_cleaned_data(filepath=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load or generate cleaned dataset."""
    if not os.path.exists(filepath):
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        save_processed_data(cleaned_df, filepath)
        return cleaned_df
    return pd.read_csv(filepath)


def get_train_test_data(
    filepath=PROCESSED_DATA_PATH,
    feature_cols=None,
    target_col="Sales",
    test_size=0.2,
    random_state=42
):
    """
    Split cleaned data into training and test sets.
    Features: TV, Radio, Newspaper spend (in $1,000s)
    Target: Sales (in 1,000s of units)
    """
    if feature_cols is None:
        feature_cols = ["TV", "Radio", "Newspaper"]

    df = load_cleaned_data(filepath)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Executing Data Loader Pipeline...")
    raw = load_raw_data()
    clean = clean_data(raw)
    save_processed_data(clean)
    X_tr, X_te, y_tr, y_te = get_train_test_data()
    print("Data Loader Pipeline executed successfully.")
