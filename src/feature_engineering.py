# raw data → meaningful features
import pandas as pd

def create_features(df):
    df["Amount_log"] = df["Amount"].apply(lambda x: 0 if x <= 0 else x)

    df["Amount_bin"] = pd.qcut(df["Amount"], q=10, duplicates="drop")

    return df