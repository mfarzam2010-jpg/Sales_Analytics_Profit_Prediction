from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_dataset_exists_and_has_target():
    df=pd.read_csv(ROOT/'data'/'dataset.csv')
    assert len(df)>100
    assert 'Profit' in df.columns
def test_model_file_exists():
    assert (ROOT/'models'/'final_pipeline.pkl').exists()
