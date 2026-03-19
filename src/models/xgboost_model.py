from xgboost import XGBClassifier

def train_xgboost(X, y):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X, y)
    return model