from joblib import dump
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def import_data(census_data_path):
    '''
    returns dataframe for the csv found at pth
    input:
            census_data_path: a path to the data in csv format
    output:
            census_df: pandas dataframe containing us census data
    '''
    census_df = pd.read_csv(census_data_path)

    return census_df


def perform_feature_engineering(
        feature_set,
        y,
        test_size=0.2,
        seed=42):
    '''
    Performs simple feature engineering (spliting into training and test sets)
    Features are scaled
    input:
              feature_set: pandas dataframe with no categorical variables
              response: string of response name [optional argument that could
              be used for naming variables or index y column]
              test size: Proportion of hold-out data for test set
              seed: seed for randomizing test set allocation
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Split data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        feature_set, y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test

def process_data(
    X, categorical_features=None, label="salary", training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    if categorical_features == None:
        categorical_features = ["workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

        # Save OneHotEncoder
        # Save LabelBinarizer
        dump(encoder, 'model/OneHotEnc.pkl')
        dump(lb, 'model/LabelBinarizer.pkl')
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            y = None

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
