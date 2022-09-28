from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from fpd.classifiers.classifier_utils import (
    convert_id_and_class_to_numeric,
    label_encode_keys,
    label_encode_platform,
    split_into_train_and_test,
)
from fpd.dataset import Dataset, TextDataset
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# FROM: https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python
def random_forrest(use_csv: bool = False):
    if use_csv:
        fp = Dataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
        )
    elif use_csv == False:
        td = TextDataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/kht_features.txt"
        )
        td2 = TextDataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/kit_features.txt"
        )
        df = pd.concat([td.to_df(), td2.to_df()])
    # print(df.head())
    # input("Column Names")
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = split_into_train_and_test(df)
    # print(X_train.shape)
    # print(y_train.shape)
    # input()

    assert X_train.shape[0] == y_train.shape[0]
    # Take the 80% portion of the split (X_train and y_train) and split them again by a 70-30 ratio, where the 30% is the validation set
    X_train, X_validate, y_train, y_validate = split_into_train_and_test(
        pd.DataFrame(
            X_train,
            columns=[
                "ID",
                "Platform",
                "Key(s)",
                "Medians",
                "Means",
                "Modes",
                "Standard Deviation",
            ],
        )
    )

    assert X_train.shape[0] == y_train.shape[0]
    X_train = np.delete(X_train, 0, axis=1)
    X_validate = np.delete(X_validate, 0, axis=1)
    X_test = np.delete(X_test, 0, axis=1)
    print(X_train)
    input()
    forest = RandomForestClassifier(
        criterion="gini", n_estimators=5, random_state=1, n_jobs=100
    )
    tuned_parameters = {
        "max_depth": [5, 10, 15, 20],
        "n_estimators": [100, 500, 800],
    }
    clf = GridSearchCV(
        forest,
        tuned_parameters,
        scoring="neg_mean_absolute_error",
        return_train_score=True,
    )
    if use_csv == False:
        # Should be running against validate x and y sets not the test ones, but the size difference makes it so the classifier won't accept it
        print(X_train)
        # input("Pre label encode x_train")
        label_encoder_x_train = label_encode_keys(X_train)
        label_encoder_x_train = label_encode_platform(label_encoder_x_train)
        # label_encoder_x_train = convert_id_and_class_to_numeric(label_encoder_x_train)
        print(label_encoder_x_train)
        # input("Final Encoded x_train")
        label_encoder_x_validate = label_encode_keys(X_validate)
        # print(label_encoder_x_validate)
        # input("Partial Encoded x_validate")
        label_encoder_x_validate = label_encode_platform(label_encoder_x_validate)
        # label_encoder_x_validate = convert_id_and_class_to_numeric(
        #     label_encoder_x_validate
        # )
        # print(label_encoder_x_validate)
        # input("Final Encoded x_validate")
        # Take the id-wise dataframe and use sklearn train-test-split to get the x_test, y_test... etc and combine together
        # ! use gridsearchcv see xgb_regression_age.py:60+
        y_train = y_train.ravel()
        y_train = y_train.astype(np.float)
        # print(y_train)
        # input("Y_Train")
        # sm = SMOTE(random_state=42)
        # label_encoder_x_train, y_train = sm.fit_resample(label_encoder_x_train, y_train)
        # label_encoder_x_validate, y_validate = sm.fit_resample(
        #     label_encoder_x_validate, y_validate
        # )
        label_encoder_x_test = label_encode_keys(X_test)
        label_encoder_x_test = label_encode_platform(label_encoder_x_test)
        # label_encoder_x_test = convert_id_and_class_to_numeric(label_encoder_x_test)

        print(y_train)
        # input("After SMOTE")
        clf.fit(label_encoder_x_train, y_train.ravel())
        y_true, y_pred = np.array(y_validate.ravel(), dtype=float), clf.predict(
            label_encoder_x_validate
        )
        print(y_true)
        input("Y_true")
        print(y_pred)
        input("Y_pred")
        print("Mean absolute error:", metrics.mean_absolute_error(y_true, y_pred))
        y_true, y_pred = np.array(y_test.ravel(), dtype=float), clf.predict(
            label_encoder_x_test
        )
        print(
            "Mean absolute error for test set:",
            metrics.mean_absolute_error(y_true, y_pred),
        )
        # print("Best params", clf.best_params_)
        # print("Results", clf.cv_results_)
        return (
            metrics.mean_absolute_error(y_true, np.array(y_pred, dtype=float)),
            clf.best_params_,
            clf.cv_results_,
        )
