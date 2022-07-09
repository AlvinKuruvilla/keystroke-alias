import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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
