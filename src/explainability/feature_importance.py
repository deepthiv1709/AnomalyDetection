import pandas as pd

def get_feature_importance(model, feature_names):
    importance = model.feature_importances_
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)