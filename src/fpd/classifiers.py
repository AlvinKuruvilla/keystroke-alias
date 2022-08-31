from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from fpd.dataset import Dataset


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
    print(X_train)
    input()
    print(X_test)
    input()
    print(y_train)
    input()
    print(y_test)
    input()
    forest = RandomForestClassifier(
        criterion="gini", n_estimators=5, random_state=1, n_jobs=2
    )
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print("Accuracy: %.3f" % metrics.accuracy_score(y_test, y_pred))
