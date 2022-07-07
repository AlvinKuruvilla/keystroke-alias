import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE


def compare_classification(label_name, feature_type, top_n_features, model):
    """Function to process the data, apply oversampling techniques (SMOTE) and run the classification model specified using GridSearchCV
    Input:  label_name: The task to be performed (Gender, Major/Minor, Typing Style)
            feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
            top_n_features: Thu number of features to be selected using Mutual Info criterion
            model: The ML model to train and evaluate
    Output: accuracy scores, best hyperparameters of the gridsearch run
    """
    # Creating class label vector using metadata
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if label_name == "Typing Style":
        for i in range(117):
            if Y_vector[i] == "a":
                Y_vector[i] = 0
            if Y_vector[i] == "b":
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if label_name == "Major/Minor":
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if type(Y_vector[i]) == float:
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if label_name == "Gender" or label_name == "Ethnicity":
        for i in range(116):
            if Y_values[i] == "M" or Y_values[i] == "Asian":
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype("int")

    X_matrix = feature_type

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)

    X_matrix_new = SelectKBest(mutual_info_classif, k=top_n_features).fit_transform(
        X_matrix, Y_vector
    )
    X_matrix_new, Y_vector = SMOTE(kind="svm").fit_sample(X_matrix_new, Y_vector)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix_new, Y_vector, test_size=0.3, random_state=0
    )

    if model == "SVM":
        # Set the parameters by cross-validation
        tuned_parameters = [
            {
                "kernel": ["rbf"],
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                "C": [0.1, 1, 10, 100, 1000, 10000],
            },
            {
                "kernel": ["poly"],
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                "C": [0.1, 1, 10, 100, 1000, 10000],
            },
            {
                "kernel": ["sigmoid"],
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                "C": [0.1, 1, 10, 100, 1000, 10000],
            },
            {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000, 10000]},
        ]

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring="accuracy", return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "DTree":
        tuned_parameters = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_leaf_nodes": list(range(2, 100)),
            "min_samples_split": [2, 3, 4, 6, 8, 10],
        }
        clf = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            tuned_parameters,
            scoring="accuracy",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "RForest":
        tuned_parameters = {
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "max_features": ["auto", "sqrt"],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [200, 400, 600, 800, 1000],
        }
        clf = GridSearchCV(
            RandomForestClassifier(),
            tuned_parameters,
            scoring="accuracy",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "XGBoost":
        tuned_parameters = {
            "min_child_weight": [1, 5, 10],
            "gamma": [0.5, 1, 5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [3, 4, 5],
        }
        clf = GridSearchCV(
            xgb.XGBClassifier(),
            tuned_parameters,
            scoring="accuracy",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "ABoost":
        tuned_parameters = {
            "n_estimators": [10, 50, 100, 200, 500, 1000],
            "learning_rate": [0.00001, 0.001, 0.01, 0.1],
            "algorithm": ["SAMME", "SAMME.R"],
        }
        clf = GridSearchCV(
            AdaBoostClassifier(),
            tuned_parameters,
            scoring="accuracy",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "MLP":
        tuned_parameters = {
            "hidden_layer_sizes": [(10,), (50,), (70,), (90,), (100,), (120,), (150,)],
            "activation": ["tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.0001, 0.01, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [100, 1000],
        }
        clf = GridSearchCV(
            MLPClassifier(),
            tuned_parameters,
            scoring="accuracy",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "NB":
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), None, None
