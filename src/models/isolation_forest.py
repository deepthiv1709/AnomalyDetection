from sklearn.ensemble import IsolationForest

def train_isolation_forest(X, params):
    model = IsolationForest(**params)
    model.fit(X)
    return model