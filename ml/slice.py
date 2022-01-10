from . import process_data as proc_data
from . import model as mod


def compute_slice_metrics(cat_feature, census_df):
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
    census_df : data frame
        Cleaned but unpre-processed US Census data
    Returns
    -------
    slice_metrics : pandas Dataframe
        Performance for each slice
    """
    for i in unique(