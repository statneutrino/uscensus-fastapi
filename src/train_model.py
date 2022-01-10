import logging
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


def train_rf(
    X_train,
    X_test,
    y_train,
    y_test,
    seed=42,
    custom_params=None):
    '''
    train, store model results:
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    # Fit random forest classifier and tune parameters using grid search
    # during CV
    rfc = RandomForestClassifier(random_state=seed)
    if custom_params is not None:
        param_grid = custom_params
    else:
        param_grid = {
            'n_estimators': [50],  # 'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [5, 10], # 'max_depth': [5, 10, 100],
            'criterion': ['gini', 'entropy']
        }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    joblib.dump(cv_rfc.best_estimator_, './model/rfc_model.pkl')

    # Create predictions based on best RF model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    print(classification_report(y_train, y_train_preds_rf))
    print(classification_report(y_test, y_test_preds_rf))


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
    print("Training random forest...")
    train_rf(X_train, X_test, y_train, y_test)



