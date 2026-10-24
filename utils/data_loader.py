from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def load_data(path=ROOT/'data'/'dataset.csv'):
    df=pd.read_csv(path,parse_dates=['Order_Date'])
    return df.drop_duplicates().copy()
