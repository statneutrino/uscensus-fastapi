import joblib
import ml.process_data as data_proc
import ml.inference as inference
import ml.model as mod

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


def run_ml_pipeline(data_path, 
                    categorical_features, 
                    label="salary",
                    train_model=True,
                    save_model=True,
                    eval_model=True):
    
    CENSUS_DF = data_proc.import_data(data_path)

    print("Imported data")

    X, y, _, _ = data_proc.process_data(CENSUS_DF,
                                        label=label,
                                        categorical_features=categorical_features
                                        )

    print("Processed data with one hot enconding and label binarizer")

    X_train, X_test, y_train, y_test = data_proc.perform_feature_engineering(
    feature_set=X,
    y=y)

    # Train model option
    if train_model == True:
        fitted_model = mod.train_rf_model(X_train, y_train, save_model=save_model)
    else:
        fitted_model = joblib.load('./model/rfc_model.pkl')

    # Eval model option
    if eval_model == True:
        y_test_preds_rf = fitted_model.predict(X_test)
        accuracy, f1, precision, recall, auc = inference.compute_model_metrics(
            y_test, y_test_preds_rf)
        print("Test set Accuracy score: {}".format(accuracy))
        print("Test set F1 score: {}".format(f1))
        print("Test set precision: {}".format(precision))
        print("Test set recall: {}".format(recall))
        print("Test set AUC: {}".format(auc))

    return fitted_model


def write_slice_printout(slice_list, census_df_path):
    df = data_proc.import_data(census_df_path)
    with open('slice_outputs/slice_output.txt', 'w') as f:
        for slice in slice_list:
            metric_df = inference.create_slice_metrics_df(
                slice, df, label="salary")
            f.write("\n")
            f.write("Estimating performance for {} categories:\n".format(slice))
            for _, row in metric_df.iterrows():
                f.write("\n")
                category, accuracy, f1, precision, recall, auc = row.values
                f.write(
                    "Model metrics for {} {} category:\n".format(
                        category, slice))
                f.write("Test set Accuracy score: {}\n".format(accuracy))
                f.write("Test set F1 score: {}\n".format(f1))
                if f1 == 1:
                    f.write(
                        "F1 is reported as 1.  All predictions and labels are likely negative class\n")
                f.write("Test set precision: {}\n".format(precision))
                if precision == 0:
                    f.write(
                        "Precision is reported as 1.  All predictions and labels are likely negative class\n")
                f.write("Test set recall: {}\n".format(recall))
                if recall == 0:
                    f.write("Recall is reported as 1.  TP + FN is likely 0")
                f.write("Test set AUC: {}\n".format(auc))
                if auc == -1:
                    f.write(
                        "AUC is reported as -1 - only one class has been predicted and therefore AUC cannot be determined\n")


if __name__ == "__main__":
    
    CENSUS_DF_PATH = "./data/census_cleaned.csv"
    
    # Train model, save model, and print performance metrics
    _ = run_ml_pipeline(CENSUS_DF_PATH, 
                        categorical_features, 
                        label="salary",
                        train_model=True,
                        save_model=True,
                        eval_model=True)

    # Produce printout of slice model prediction performance:
    slices_to_test = ['education', 'sex', 'race']

    write_slice_printout(slices_to_test, CENSUS_DF_PATH)

