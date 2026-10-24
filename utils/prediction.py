from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
def load_pipeline():
    return joblib.load(ROOT/'models'/'final_pipeline.pkl')
def predict_profit(pipeline,order_frame):
    return pipeline.predict(order_frame)
