
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
    pipeline = load_pipeline()

    input_data = pd.DataFrame([{
        "Customer_Gender": "Male",
        "Customer_Age": 30,
        "City": "Karachi",
        "Region": "South",
        "Product_Category": "Electronics",
        "Product_Name": "Test Product",
        "Brand": "Test Brand",
        "Quantity": 2,
        "Unit_Price": 2500.0,
        "Discount": 0.10,
        "Shipping_Cost": 350.0,
        "Payment_Method": "Cash",
        "Delivery_Days": 4,
        "Salesperson": "Test Salesperson",
        "Order_Year": 2026,
        "Order_Month": 10,
        "Quarter": 4
    }])

    prediction = predict_profit(pipeline, input_data)

    assert len(prediction) == 1
    assert pd.notna(prediction[0])
    assert isinstance(float(prediction[0]), float)
