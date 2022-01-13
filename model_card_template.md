# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Pre-trained Random Forest classifier using scikit-learn framework,
on the Census Income Data Set, trained using cross-validation to predict whether salary of an 
individual or range of individuals is under or 
over $50k.

Developed by Alex Spiers as part of Udacity Nanodegree for Machine Learning DevOps Engineer Nanodegree Program

## Intended Use
To be deployed on Heroku as a web app and to be combined with an API, written using FastAPI. 
The API is live in production qithin a full CI/CD
framework. 

On its own, this project can be a portfolio piece but
can also be applied to other projects, e.g., continuous integration,
to flesh them further out.

It is not intended to estimate actual individual's salaries or used to inform pay benchmarking - it is merely used as a portfolio piece to illustrate the author's skills in MLOps, in e.g., continuous integration.

## Training Data
The training data used was the US Census Income Data Set which can be found 
at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income)

80% of the data was used for cross-validation

## Evaluation Data

20% hold-out data was used

## Metrics
Evaluation metrics include:

- Accuracy
- Precision <img src="https://render.githubusercontent.com/render/math?math==\frac{tp}{tp %2Bfp}"> where tp is true positive and fp is false positive
- Recall <img src="https://render.githubusercontent.com/render/math?math==\frac{tp}{tp %2Btn}"> where tp is true positive and tn is true negative
- F1 score which is the harmonic mean of Precision and Recall, or <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{recall^{-1} %2Bprecision}">
- AUC of ROC - ROC curve is the recall as a function of fall-out. AUC is a measure of overall classification performance.

### Model Performance on Test Set

Test set Accuracy score: 0.855
Test set F1 score: 0.807
Test set precision: 0.526
Test set recall: 0.6
Test set AUC: 0.743

See slice_ouput.txt for performance on Education, Sex and Race subcategories

## Ethical Considerations

#### Data:
This dataset has no sensitive data
#### Usage risks and harms
There is the potential to use this data for abuse around hiring or salary benchmarking. We used data sliceing to investigate model bias.

## Caveats and Recommendations
Accuracy is generally quite poor. As a dichotomous prediction (above or below 50k), the classifier has limited use. A better dataset with income
information as continuous variable would be more useful.