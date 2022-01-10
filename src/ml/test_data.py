import pytest
import pandas as pd
import numpy as np
from . import data as d


@pytest.fixture
def processed_data():
    df_path = "./data/census_cleaned.csv"
    df = pd.read_csv(df_path)
    X, y, encoder, lb = d.process_data(df,
            categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]
    )
    return X, y, encoder, lb


def test_y_binary(processed_data):
    X, y, encoder, lb = processed_data
    assert set(np.unique(y)) == {0,1}
    

def test_X_one_hot_enc(processed_data):
    X, y, encoder, lb = processed_data
    assert X.dtype == np.float64
