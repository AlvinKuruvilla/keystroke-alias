from sklearn import preprocessing
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from core.features.feature_processor import get_combined_features


def get_train_test_splits(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if label_name == "Gender" or label_name == "Ethnicity":
        for i in range(60):
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

    print(X_matrix_new)
    # NOTE: It looks like the "kind" argument has been removed from SMOTE
    # NOTE: It looks like if  SMOTE is imported like how we did it we need to change fit_sample to fit_resample
    X_matrix_new, Y_vector = SMOTE().fit_resample(X_matrix_new, Y_vector)
    return X_matrix_new, Y_vector
