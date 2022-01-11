import joblib
import pandas as pd
from . import process_data as proc_data
from . import model as mod
from sklearn.metrics import fbeta_score, precision_score, recall_score


def inference(df, model=None, encoder=None, output="binary"):
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
    if encoder == None:
        encoder = joblib.load('./model/OneHotEnc.pkl') # Load OneHotEncoder

    if model == None:
        model = joblib.load('./model/rfc_model.pkl') # Load RandomForestClassifier

    processed_data = proc_data.process_data(
        df,
        encoder = encoder,
        training = False
    )
    pred_y = model.predict(processed_data[0])
    
    if output=="binary":
        return pred_y
    if output=="string":
        lb = joblib.load('./model/LabelBinarizer.pkl') # Load LabelBinarizer
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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_slice_metrics(cat_feature, df, model=None, lb=None):
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
    if model == None:
        model = joblib.load('./model/rfc_model.pkl')

    if lb == None:
        lb = joblib.load('./model/LabelBinarizer.pkl')

    # Generate pandas df with slice performance
    slice_metrics = pd.DataFrame(columns=('slice', 'f1', 'precision', 'recall'))
    for count, slice in enumerate(df[cat_feature].unique()):
        pred_for_slice = inference(model, df[df[cat_feature] == slice], encoder=None)
        y = lb.transform(df['salary'][df[cat_feature] == slice]).ravel()
        fbeta, precision, recall = compute_model_metrics(y, pred_for_slice)
        slice_metrics.loc[count] = [slice, fbeta, precision, recall]

    return slice_metrics
