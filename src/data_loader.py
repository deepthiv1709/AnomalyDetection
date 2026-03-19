# src/data_loader.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from given path and perform basic validation.
    """
    # --------------------------
    # Step 1: Load Data
    # --------------------------
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path)

    # --------------------------
    # Step 2: Basic checks
    # --------------------------
    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    print(f"Data loaded successfully from {file_path}")
    print("Shape of dataset:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nStatistical summary:\n", df.describe())

    # --------------------------
    # Step 3: Class Distribution
    # --------------------------
    print("\nClass distribution:\n", df['Class'].value_counts())

    # Linear scale bar chart
    counts = df['Class'].value_counts().sort_index()
    print(counts.values)
    labels = ['Legit (0)', 'Fraud (1)']
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=counts.values)
    plt.title("Class Distribution (linear scale)")
    plt.ylabel("Count")

    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values)*0.01, f"{v:,}", ha='center')
    plt.show()

    # Log scale shows both values clearly while preserving the huge difference
    plt.figure(figsize=(6,6))
    sns.barplot(x=labels, y=counts.values)
    plt.yscale('log')
    plt.title("Class Distribution")
    plt.xlabel("Class (0 = Legit, 1 = Fraud)")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v*1.1, f"{v:,}", ha='center')
    plt.show()

    return df


def validate_data(df: pd.DataFrame):
    """
    Perform basic sanity checks.
    """
    print("\n Running data validation checks...")

    # Missing values
    missing = df.isnull().sum()
    print("\nMissing values:\n", missing[missing > 0])

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

    # Class balance (if exists)
    if "Class" in df.columns:
        print("\nClass distribution:\n", df["Class"].value_counts())

    # ------------------------------------
    # Step 4: Outlier Detection (Baseline)
    # ------------------------------------
    # Calculate z-scores
    z_scores = df.drop(columns=['Class']).apply(zscore)

    # Count potential outliers per column (|z| > 3)
    outlier_counts = (np.abs(z_scores) > 3).sum()

    # Calculate skewness per column
    skewness = df.drop(columns=['Class']).skew()

    # Combine into a single dataframe
    outlier_skew_df = pd.DataFrame({
        'Outliers': outlier_counts,
        'Skewness': skewness
    }).sort_values('Outliers', ascending=False)
    # For skewed data (like Amount in credit card transactions), Z-score underestimates outliers, because extreme values can skew the mean and standard deviation.
    print(outlier_skew_df) 
    # Plotting
    fig, ax1 = plt.subplots(figsize=(14,6))

    sns.barplot(x=outlier_skew_df.index, y=outlier_skew_df['Outliers'], ax=ax1, color='skyblue')
    ax1.set_ylabel('Number of Outliers', color='blue')
    ax1.set_xlabel('Features')
    ax1.set_xticklabels(outlier_skew_df.index, rotation=45)

    # Secondary axis for skewness
    ax2 = ax1.twinx()
    sns.lineplot(x=outlier_skew_df.index, y=outlier_skew_df['Skewness'], ax=ax2, color='red', marker='o')
    ax2.set_ylabel('Skewness', color='red')

    plt.title('Feature Outliers and Skewness')
    plt.show()
    # Many features have extreme outliers, which likely correspond to fraud signals.
    # Skewed distributions, especially for Amount, indicate uneven transaction sizes.
    # This is typical for credit card fraud datasets and influences the choice of anomaly detection methods:
        # Isolation Forest / Autoencoders handle skew and outliers well.
        # Distance-based methods like LOF may require normalization.

    # IQR method
    # Better for skewed data, will flag more points in heavy-tailed distributions.
    print("\nChecking extreme values using IQR:")
    numeric_cols = df.drop(columns=['Class'])
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"{col}: {outliers.shape[0]} potential outliers")

    # Z-score: Only flags extreme deviations from the mean; works well for near-normal features.
    # IQR: Flags points far outside the interquartile range; much more sensitive to skewed and heavy-tailed features like transaction Amount.
    # For credit card fraud, IQR is often more appropriate because the data is highly skewed and contains extreme fraud values.

    # --------------------------
    # Step 5: Correlation Analysis
    # --------------------------
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    print("Validation complete\n")