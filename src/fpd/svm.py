import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path)

    def path(self):
        return self.dataset_path

    def get_data(self):
        return self.data

    def as_numpy_array(self):
        return self.data.to_numpy()

    def feature_names(self):
        # The last column denotes whether or not the user is a fake profile or not so we want to ignore
        return list(self.get_data().columns[:-1])

    def target_names(self):
        return ["Fake Profile", "Genuine Profile"]

    def target(self):
        return np.array(list(self.get_data().iloc[:, len(self.feature_names())]))


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
