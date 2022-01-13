import joblib
import ml.process_data as data_proc
import ml.inference as inference

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

if __name__ == "__main__":
    CENSUS_DF = data_proc.import_data("./data/census_cleaned.csv")

    print("Imported data")

    X, y, encoder, lb = data_proc.process_data(CENSUS_DF,
                                               label="salary",
                                               categorical_features=categorical_features
                                               )

    print("Processed data with one hot enconding and label binarizer")

    print(y)

    X_train, X_test, y_train, y_test = data_proc.perform_feature_engineering(
        feature_set=X,
        y=y
    )

    # Train model
    # fitted_model = mod.train_rf_model(X_train, y_train, save_model=True)
    # print("Random forest model fitted")
    # Load model
    fitted_model = joblib.load('./model/rfc_model.pkl')
    print(type(fitted_model))

    # Assess test_performance
    y_test_preds_rf = fitted_model.predict(X_test)

    # Assess overall model performance
    accuracy, f1, precision, recall, auc = inference.compute_model_metrics(y_test, y_test_preds_rf)
    print("Test set Accuracy score: {}".format(accuracy))
    print("Test set F1 score: {}".format(f1))
    print("Test set precision: {}".format(precision))
    print("Test set recall: {}".format(recall))
    print("Test set AUC: {}".format(auc))

    # Test predictions
    encoder = joblib.load('./model/OneHotEnc.pkl')  # Load OneHotEncoder

    processed_data_test = data_proc.process_data(
        CENSUS_DF.sample(5),
        categorical_features=categorical_features,
        encoder=encoder,
        training=False
    )

    predictions = fitted_model.predict(processed_data_test[0])

    #Test model slice function
    # education_metrics = inference.create_slice_metrics_df('education', CENSUS_DF)
    # education_metrics.to_csv(
    #     path_or_buf='./slice_outputs/education.txt',
    #     index=None)

    # Produce printout of slice model prediction performance:
    slices_to_test = ['education', 'sex', 'race']
    
    with open('slice_outputs/slice_output.txt', 'w') as f:
        for slice in slices_to_test:
            metric_df = inference.create_slice_metrics_df(slice, CENSUS_DF)
            f.write("\n")
            f.write("Estimating performance for {} categories:\n".format(slice))
            for _, row in metric_df.iterrows():
                f.write("\n")
                category, accuracy, f1, precision, recall, auc = row.values
                f.write("Model metrics for {} {} category:\n".format(category, slice))
                f.write("Test set Accuracy score: {}\n".format(accuracy))
                f.write("Test set F1 score: {}\n".format(f1))
                if f1 == 1:
                    f.write("F1 is reported as 1.  All predictions and labels are likely negative class\n")
                f.write("Test set precision: {}\n".format(precision))
                if precision == 0:
                    f.write("Precision is reported as 1.  All predictions and labels are likely negative class\n")
                f.write("Test set recall: {}\n".format(recall))
                if recall == 0:
                    f.write("Recall is reported as 1.  TP + FN is likely 0")
                f.write("Test set AUC: {}\n".format(auc))
                if auc == -1:
                    f.write("AUC is reported as -1 - only one class has been predicted and therefore AUC cannot be determined\n")
