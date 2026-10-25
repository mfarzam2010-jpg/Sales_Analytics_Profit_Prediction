from pathlib import Path

import pandas as pd

from utils.prediction import load_pipeline, predict_profit


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists_and_has_target():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")

    assert len(df) > 100
    assert "Profit" in df.columns


def test_model_file_exists():
    assert (ROOT / "models" / "final_pipeline.pkl").exists()


def test_model_loads_successfully():
    pipeline = load_pipeline()

    assert pipeline is not None
    assert hasattr(pipeline, "predict")


def test_model_prediction():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")

    sample = df.iloc[[0]].copy()
    sample["Order_Date"] = pd.to_datetime(sample["Order_Date"])

    sample["Order_Year"] = sample["Order_Date"].dt.year
    sample["Order_Month"] = sample["Order_Date"].dt.month
    sample["Quarter"] = sample["Order_Date"].dt.quarter

    features = [
        "Customer_Gender",
        "Customer_Age",
        "City",
        "Region",
        "Product_Category",
        "Product_Name",
        "Brand",
        "Quantity",
        "Unit_Price",
        "Discount",
        "Shipping_Cost",
        "Payment_Method",
        "Delivery_Days",
        "Salesperson",
        "Order_Year",
        "Order_Month",
        "Quarter"
    ]

    input_data = sample[features]

    pipeline = load_pipeline()
    prediction = predict_profit(pipeline, input_data)

    assert len(prediction) == 1
    assert pd.notna(prediction[0])
