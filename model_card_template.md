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

- Precision <img src="https://render.githubusercontent.com/render/math?math==\frac{tp}{tp %2Bfp}"> where tp is true positive and fp is false positive
- Recall <img src="https://render.githubusercontent.com/render/math?math==\frac{tp}{tp %2Btn}"> where tp is true positive and tn is true negative
- F1 score which is the harmonic mean of Precision and Recall, or <img src="https://render.githubusercontent.com/render/math?math=\frac{2}{recall^{-1} %2Bprecision}">
measure disproportionate model performance errors across subgroups. False
Discovery Rate and False Omission Rate, which measure the fraction of negative (not smiling) and positive (smiling) predictions that are incorrectly predicted
to be positive and negative, respectively, are also reported. 

### Model Performance on Test Set



## Ethical Considerations

## Caveats and Recommendations
