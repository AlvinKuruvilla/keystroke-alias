from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from fpd.dataset import (
    Dataset,
    TextDataset,
    get_target_from_df,
    split_into_train_and_test,
)
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode_platform(array):
    keys = []
    for row in array:
        print("Platform:", row[0])
        keys.append(row[0])
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(keys)
    label_encoded_keys = label_encoder.transform(keys)
    count = 0
    for row in array:
        row[0] = label_encoded_keys[count]
        count += 1
    return array


def label_encode_keys(array):
    keys = []
    for row in array:
        print(row)
        keys.append(row[1])
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(keys)
    label_encoded_keys = label_encoder.transform(keys)
    count = 0
    for row in array:
        row[1] = label_encoded_keys[count]
        count += 1
    return array


# NOTE: Takes WAY too long to run (7+ hours) and still not running to completion
def create_svm(use_csv: bool = False):
    if use_csv:
        fp = Dataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
        )
    elif use_csv == False:
        fp = TextDataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/keystroke_features.txt"
        )
    print(fp.target())
    X_train, X_test, y_train, y_test = train_test_split(
        fp.as_numpy_array(), fp.target(), test_size=0.3, random_state=109
    )  # 70% training and 30% test
    print(X_train)
    input()
    print(X_test)
    input()
    print(y_train)
    input()
    print(y_test)
    input()
    clf = svm.SVC(kernel="linear")  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)
    print("HERE")
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


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
    y = get_target_from_df(df)
    print(y)
    input("Stratify")
    X_train, X_test, y_train, y_test = split_into_train_and_test(df)
    # 70% training and 30% test
    print(X_train)
    input()
    print(X_test)
    input()
    print(y_train)
    input()
    print(y_test)
    input()
    forest = RandomForestClassifier(
        criterion="gini", n_estimators=5, random_state=1, n_jobs=100
    )
    if use_csv == False:
        label_encoder_x_train = label_encode_keys(X_train)
        label_encoder_x_train = label_encode_platform(label_encoder_x_train)
        print(label_encoder_x_train)
        input("Final Encoded x_train")
        label_encoder_x_test = label_encode_keys(X_test)
        print(label_encoder_x_test)
        input("Partial Encoded x_test")
        label_encoder_x_test = label_encode_platform(label_encoder_x_test)
        print(label_encoder_x_test)
        input("Final Encoded x_test")

        forest.fit(label_encoder_x_train, y_train.ravel())
        y_pred = forest.predict(label_encoder_x_test)
        print("Random Forrest Accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))


def xgb_classifier(use_csv: bool = False):
    if use_csv:
        fp = Dataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
        )
    elif use_csv == False:
        fp = TextDataset(
            "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/keystroke_features.txt"
        )
    y = fp.target()
    X_train, X_test, y_train, y_test = train_test_split(
        fp.as_numpy_array(),
        fp.target(),
        test_size=0.3,
        random_state=1,
        stratify=y,
    )  # 70% training and 30% test
    print("X_Test partition:", X_test)
    input("X_test")
    print("y_Test partition:", y_test)
    input("y_test")

    if use_csv == True:
        dtrain = xgb.DMatrix(X_train, label=y_train.values.ravel())
        dtest = xgb.DMatrix(X_test, label=y_test.values.ravel())
    elif use_csv == False:
        label_encoder_x_train = label_encode_keys(X_train)
        print(label_encoder_x_train)
        input("Encoded x_train")

        label_encoder_x_test = label_encode_keys(X_test)
        dtrain = xgb.DMatrix(label_encoder_x_train, label=y_train)
        dtest = xgb.DMatrix(label_encoder_x_test, label=y_test)
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
    print("Precision:", metrics.precision_score(y_test, best_preds, average="macro"))
    print(
        "XGBoost Classifier Accuracy: %.3f" % metrics.accuracy_score(y_test, best_preds)
    )
