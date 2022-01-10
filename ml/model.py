from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Optional: implement hyperparameter tuning.
def train_rf_model(X_train, y_train, seed=42, custom_params=None, save_model=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
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

    if save_model == True:
        joblib.dump(cv_rfc.best_estimator_, './model/rfc_model.pkl')

    return cv_rfc.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
