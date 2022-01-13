import pandas as pd
import pytest
import ml.process_data as data_proc


@pytest.fixture
def census_data():
    df_path = "./data/census_cleaned.csv"
    df = pd.read_csv(df_path)
    yield df


@pytest.fixture
def processed_data():
    df_path = "./data/census_cleaned.csv"
    df = pd.read_csv(df_path)
    X, y, encoder, lb = data_proc.process_data(df,
                                               label="salary",
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


def test_import_type(census_data):
    '''
    Tests data imported correctly
    '''
    df = census_data
    assert isinstance(df, pd.core.frame.DataFrame)


def test_import_size(census_data):
    '''
    Tests data imported correctly
    '''
    df = census_data
    assert df['salary'].size > 0
    assert df.shape[1] > 0


def test_feature_engineering_shape(processed_data):
    X, y, _, _ = processed_data
    X_train, X_test, y_train, y_test = data_proc.perform_feature_engineering(
        feature_set=X,
        y=y
    )
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_test.size > 0
    assert y_train.size > 0


def test_feature_engineering_size(processed_data, test_size=0.2):
    X, y, _, _ = processed_data
    _, X_test, _, y_test = data_proc.perform_feature_engineering(
        feature_set=X,
        y=y
    )
    assert X_test.shape[0] - \
        1 <= round(test_size * X.shape[0]) <= X_test.shape[0] + 1
    assert y_test.size - \
        1 <= round(test_size * X.shape[0]) <= y_test.size + 1
