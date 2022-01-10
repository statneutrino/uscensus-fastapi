import pandas as pd
import pytest


@pytest.fixture
def census_data():
    df_path = "./data/census_cleaned.csv"
    df = pd.read_csv(df_path)
    yield df


@pytest.fixture
def processed_data():
    df_path = "./data/census_cleaned.csv"
    df = pd.read_csv(df_path)
    X, y, encoder, lb = tm.process_data(df, 
        label = "salary",
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


def test_import(census_data):
    '''
    Tests data imported correctly
    '''
    df = census_data
    assert type(df) == pd.core.frame.DataFrame
    assert df['salary'].size > 0
    assert df.shape[1] > 0

# def test_feature_engineering(processed_data):
#     X, y, encoder, lb = processed_data
#     X_train, X_test, y_train, y_test = tm.perform_feature_engineering(
#         feature_set = X, 
#         y = y
#     )