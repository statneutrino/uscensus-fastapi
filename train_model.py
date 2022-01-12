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

    # Assess model performance
    test_performance = inference.compute_model_metrics(y_test, y_test_preds_rf)
    print("Test set F1 Score: {}".format(test_performance[0]))
    print("Test set precision: {}".format(test_performance[1]))
    print("Test set recall: {}".format(test_performance[2]))

    # Test predictions
    encoder = joblib.load('./model/OneHotEnc.pkl')  # Load OneHotEncoder

    processed_data_test = data_proc.process_data(
        CENSUS_DF.sample(5),
        categorical_features=categorical_features,
        encoder=encoder,
        training=False
    )

    print(processed_data_test[0].shape)
    predictions = fitted_model.predict(processed_data_test[0])
    print(predictions)

    # Test model slice function
    education_metrics = inference.compute_slice_metrics('education', CENSUS_DF)
    education_metrics.to_csv(
        path_or_buf='./slice_outputs/education.txt',
        index=None)
