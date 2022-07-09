import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def compare_regression(label_name, feature_type, top_n_features, model):
    """Function to process the data and run the regression model specified using GridSearchCV
    Input:  label_name: The task to be performed (Age, Height)
            feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
            top_n_features: Thu number of features to be selected using Mutual Info criterion
            model: The ML model to train and evaluate
    Output: accuracy scores, best hyperparameters of the gridsearch run
    """
    # Creating class label vector using metadata
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()

    Y_vector = np.asarray(Y_values)

    Y_vector = Y_vector[:-1]
    Y_vector = Y_vector.astype("int")

    X_matrix = feature_type

    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)

    np.random.seed(0)
    X_matrix_new = SelectKBest(mutual_info_classif, k=top_n_features).fit_transform(
        X_matrix, Y_vector
    )

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix_new, Y_vector, test_size=0.3, random_state=0
    )

    if model == "SVM":
        # Set the parameters by cross-validation
        tuned_parameters = [
            {
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
                "C": [0.1, 1, 10, 100, 1000],
            },
            {
                "kernel": ["poly"],
                "gamma": ["scale", "auto"],
                "C": [0.1, 1, 10, 100, 1000],
            },
            {
                "kernel": ["sigmoid"],
                "gamma": ["scale", "auto"],
                "C": [0.1, 1, 10, 100, 1000],
            },
            {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
        ]

        clf = GridSearchCV(
            SVR(),
            tuned_parameters,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "Lasso":
        # Set the parameters by cross-validation
        tuned_parameters = {
            "alpha": [0.2, 0.4, 0.6, 0.8, 1],
            "selection": ["cyclic", "random"],
        }

        clf = GridSearchCV(
            Lasso(),
            tuned_parameters,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "Ridge":
        # Set the parameters by cross-validation
        tuned_parameters = {
            "alpha": [25, 10, 4, 2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
        }

        clf = GridSearchCV(
            Ridge(),
            tuned_parameters,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "KNN":
        # Set the parameters by cross-validation
        tuned_parameters = {
            "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20],
        }

        clf = GridSearchCV(
            KNeighborsRegressor(),
            tuned_parameters,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "XGB":
        # Set the parameters by cross-validation
        tuned_parameters = {
            "objective": ["reg:linear"],
            "learning_rate": [0.01, 0.1, 0.001],  # so called `eta` value
            "max_depth": [5, 10, 15, 20],
            "min_child_weight": [4, 8],
            "silent": [1],
            "subsample": [0.7],
            "colsample_bytree": [0.5, 0.7, 1.0],
            "n_estimators": [100, 500, 800],
        }

        clf = GridSearchCV(
            XGBRegressor(),
            tuned_parameters,
            scoring="neg_mean_absolute_error",
            return_train_score=True,
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_


def regression_results(problem, feature_type, model):
    num_features = []
    mae = []
    hyper = []
    val_score = []
    for i in range(5, 105, 5):
        res, setup, val = compare_regression(problem, feature_type, i, model)
        num_features.append(i)
        mae.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(mae)
    print(hyper)
    # print(val_score)


class_problems = ["Age", "Height"]
models = ["SVM", "KNN", "XGBoost"]

for model in models:
    print(
        "###########################################################################################"
    )
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        regression_results(class_problem, get_desktop_features(), model)
        print("Phone")
        regression_results(class_problem, get_phone_features(), model)
        print("Tablet")
        regression_results(class_problem, get_tablet_features(), model)
        print("Combined")
        regression_results(class_problem, get_combined_features(), model)
        print()
        print(
            "-----------------------------------------------------------------------------------------"
        )
