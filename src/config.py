# easy experimentation
# no hardcoding
# industry standard

class Config:
    DATA_PATH = "C:/AnomalyDetection/data/transactions.csv"
    MODEL_DIR = "models/"
    DATA_DIR = "data/"
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    ISOLATION_FOREST_PARAMS = {
        "n_estimators": 400,
        "contamination": 0.005
    }