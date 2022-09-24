import xgboost as xgb
import pandas as pd
import numpy as np
from fpd.dataset import Dataset, TextDataset
from fpd.classifiers.classifier_utils import (
    convert_id_and_class_to_numeric,
    label_encode_keys,
    label_encode_platform,
    split_into_train_and_test,
)
from imblearn.over_sampling import SMOTE
from sklearn import metrics


def xgb_classifier(use_csv: bool = False):
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
        # 80% training and 20% test
        X_train, X_test, y_train, y_test = split_into_train_and_test(df, 0.8)
        assert X_train.shape[0] == y_train.shape[0]
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
                "Class",
            ],
        )
    )
    assert X_train.shape[0] == y_train.shape[0]
    # print("X_Test partition:", X_test)
    # input("X_test")
    # print("y_Test partition:", y_test)
    # input("y_test")

    if use_csv == True:
        dtrain = xgb.DMatrix(X_train, label=y_train.values.ravel())
        dtest = xgb.DMatrix(X_validate, label=y_validate.values.ravel())
    elif use_csv == False:
        label_encoder_x_train = label_encode_keys(X_train)
        label_encoder_x_train = label_encode_platform(label_encoder_x_train)
        label_encoder_x_train = convert_id_and_class_to_numeric(label_encoder_x_train)
        print(label_encoder_x_train)
        # input("Final Encoded x_train")
        label_encoder_x_validate = label_encode_keys(X_validate)
        print(label_encoder_x_validate)
        # input("Partial Encoded x_validate")
        label_encoder_x_validate = label_encode_platform(label_encoder_x_validate)
        label_encoder_x_validate = convert_id_and_class_to_numeric(
            label_encoder_x_validate
        )
        y_train = y_train.ravel()
        y_train = y_train.astype(np.float)
        y_validate = y_validate.ravel()
        y_validate = y_validate.astype(np.float)
        print(y_train)
        input("DType")
        sm = SMOTE(random_state=42)
        label_encoder_x_train, y_train = sm.fit_resample(label_encoder_x_train, y_train)
        label_encoder_x_validate, y_validate = sm.fit_resample(
            label_encoder_x_validate, y_validate
        )
        dtrain = xgb.DMatrix(label_encoder_x_train, label=y_train)
        dtest = xgb.DMatrix(label_encoder_x_validate, label=y_validate)
    param = {
        "max_depth": 3,  # the maximum depth of each tree
        "eta": 0.3,  # the training step for each iteration
        "objective": "multi:softprob",  # error evaluation for multiclass training
        "num_class": 3,  # the number of classes that exist in this dataset
    }
    num_round = 20  # the number of training iterations
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    best_preds = [str(i) for i in best_preds]
    y_validate = y_validate.ravel()
    y_validate = y_validate.astype(str)
    print(y_validate)
    input("Stringify")
    print(
        "Precision:", metrics.precision_score(y_validate, best_preds, average="macro")
    )
    print(
        "XGBoost Classifier Accuracy: %.3f" % metrics.accuracy_score(y_test, best_preds)
    )
