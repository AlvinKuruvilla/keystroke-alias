import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from features.feature_lists import get_combined_features, get_desktop_features


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


# function to call the compare_regression function for the specified model, feature_type and task
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


def run_age_xgb_regression():
    class_problems = ["Age"]

    models = ["XGBoost"]

    for model in models:
        print(
            "###########################################################################################"
        )
        print(model)
        for class_problem in class_problems:
            print(class_problem)
            print("Desktop")
            regression_results(class_problem, get_desktop_features(), model)
            print("Combined")
            regression_results(class_problem, get_combined_features(), model)
            print()
            print(
                "-----------------------------------------------------------------------------------------"
            )
