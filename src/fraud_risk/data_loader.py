import pandas as pd
from fraud_risk.config import RAW_DATA_FILE

def load_data():
    return pd.read_csv(RAW_DATA_FILE)