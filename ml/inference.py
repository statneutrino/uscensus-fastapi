import joblib
import pandas as pd
from . import process_data as proc_data
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def inference(df, model=None, encoder=None, output="binary", label=None):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    df : pandas data frame
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    lb = joblib.load('./model/LabelBinarizer.pkl')

    if encoder is None:
        encoder = joblib.load('./model/OneHotEnc.pkl')  # Load OneHotEncoder

    if model is None:
        # Load RandomForestClassifier
        model = joblib.load('./model/rfc_model.pkl')
    if label is not None:
        label = "salary"

    processed_data = proc_data.process_data(
        X=df,
        encoder=encoder,
        training=False,
        label=label,
        lb=lb
    )

    pred_y = model.predict(processed_data[0])

    if output == "binary":
        return pred_y
    if output == "string":
        lb = joblib.load('./model/LabelBinarizer.pkl')  # Load LabelBinarizer
        return str(lb.inverse_transform(pred_y)[0])


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
    accuracy = round(accuracy_score(y, preds), 3)
    f1 = round(f1_score(y, preds, zero_division=0), 1)
    precision = round(precision_score(y, preds, zero_division=1), 3)
    recall = round(recall_score(y, preds, zero_division=1), 3)
    try:
        auc = round(roc_auc_score(y, preds), 3)
    except ValueError:
        auc = -1
    return accuracy, precision, recall, f1, auc


def create_slice_metrics_df(cat_feature, df, model=None, lb=None, label=None):
    """
    computes performance on model slices.
    I.e. a function that computes the performance metrics when the value of a
    given feature is held fixed. E.g. for education, it would print out
    the model metrics for each slice of data that has a particular value
    for education.

    Inputs
    ------
    cat_feature : string
        feature in Census data set chosen to compute slices for
    df : data frame
        Cleaned but unpre-processed US Census data
    Returns
    -------
    slice_metrics : pandas Dataframe
        Performance for each slice
    """
    if model is None:
        model = joblib.load('./model/rfc_model.pkl')

    if lb is None:
        lb = joblib.load('./model/LabelBinarizer.pkl')

    # Generate pandas df with slice performance
    slice_metrics = pd.DataFrame(
        columns=(
            'slice',
            'accuracy',
            'precision',
            'recall',
            'f1',
            'auc'))
    for count, slice in enumerate(df[cat_feature].unique()):
        pred_for_slice = inference(
            df[df[cat_feature] == slice],
            model=model,
            encoder=None,
            label=label)
        y = lb.transform(df['salary'][df[cat_feature] == slice]).ravel()
        accuracy, f1, precision, recall, auc = compute_model_metrics(
            y, pred_for_slice)
        slice_metrics.loc[count] = [
            slice, accuracy, f1, precision, recall, auc]

    return slice_metrics
