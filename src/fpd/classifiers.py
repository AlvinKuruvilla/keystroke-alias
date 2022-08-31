from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from fpd.dataset import Dataset
import xgboost as xgb
import numpy as np

# NOTE: Takes WAY too long to run (7+ hours) and still not running to completion
def create_svm():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
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
def random_forrest():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
    )
    y = fp.target()
    X_train, X_test, y_train, y_test = train_test_split(
        fp.as_numpy_array(),
        fp.target(),
        test_size=0.3,
        random_state=1,
        stratify=y,
    )  # 70% training and 30% test
    # print(X_train)
    # input()
    # print(X_test)
    # input()
    # print(y_train)
    # input()
    # print(y_test)
    # input()
    forest = RandomForestClassifier(
        criterion="gini", n_estimators=5, random_state=1, n_jobs=100
    )
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print("Random Forrest Accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))


def xgb_classifier():
    fp = Dataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/all_features.csv"
    )
    y = fp.target()
    X_train, X_test, y_train, y_test = train_test_split(
        fp.as_numpy_array(),
        fp.target(),
        test_size=0.3,
        random_state=1,
        stratify=y,
    )  # 70% training and 30% test
    # print(X_train)
    # input()
    # print(X_test)
    # input()
    # print(y_train)
    # input()
    # print(y_test)
    # input()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
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
    print("Precision:", metrics.precision_score(y_test, best_preds, average="macro"))
    print(
        "XGBoost Classifier Accuracy: %.3f" % metrics.accuracy_score(y_test, best_preds)
    )
