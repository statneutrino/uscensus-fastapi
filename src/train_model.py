import os
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.data import process_data


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


if __name__ == "__main__":
    CENSUS_DF = import_data("./data/census_cleaned.csv")

    print(CENSUS_DF.columns)

    print("Imported data")

    X, y, encoder, lb = process_data(CENSUS_DF, 
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

    print("Processed data with one hot enconding and label binarizer")

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        feature_set = X, 
        y = y
    )

    print("Dimensions of training data:")
    print(X_train.shape)
    print("Dimensions of test data:")
    print(X_test.shape)



