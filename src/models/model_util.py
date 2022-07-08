import numpy as np
from sklearn import preprocessing
import pandas as pd
from imblearn.over_sampling import SMOTE

from features.feature_lists import get_combined_features


# Create the appropriate train-test splits for free text classification tasks to align with the ML models
def get_train_test_splits(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if label_name == "Typing Style":
        for i in range(117):
            if Y_vector[i] == "a":
                Y_vector[i] = 0
            elif Y_vector[i] == "b":
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

    # uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features()
    # X_matrix = get_desktop_features()
    # X_matrix = get_phone_features()
    # X_matrix = get_tablet_features()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)
    X_matrix_new, Y_vector = SMOTE(kind="svm").fit_sample(X_matrix_new, Y_vector)
    return X_matrix_new, Y_vector
