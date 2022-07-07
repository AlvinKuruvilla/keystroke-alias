from tqdm import tqdm
import numpy as np
import pickle


def get_features(features):
    """Input: All feature dictionary Output: Feature matrix with unique columns"""
    feature_set = []
    for key1 in features:
        for key2 in features[key1]:
            feature_set.append(key2)

    # Getting unique columns by removing repeated keys
    unique_feature_set = set(feature_set)
    unique_feature_set = list(unique_feature_set)

    size = len(unique_feature_set)
    rows, cols = (len(features), len(unique_feature_set))
    feature_vector = [[0 for x in range(len(cols))] for x in range(rows)]

    # Updating feature matrix based on present features in dictionary
    for key1 in tqdm(features):
        for key2 in features[key1]:
            for j in range(len(unique_feature_set)):
                if unique_feature_set[j] == key2:
                    temp = abs(np.median(features[key1][key2]))
                    feature_vector[(key1) - 1][j] = int(temp)
                    break
                else:
                    feature_vector[(key1) - 1][j] = 0
    return feature_vector


top_feature_KIT_Desktop_F1 = [
    "in",
    "nSPACE",
    "et",
    "lSPACE",
    "oSPACE",
    "ca",
    "iSPACE",
    "pl",
    "ve",
    "ha",
    "ne",
    "da",
    "he",
    "wi",
]
top_feature_KIT_Tablet_F1 = ["me", "is", "ne", "ha"]
top_feature_KIT_Tablet_F2 = ["ha", "is", "me"]
top_feature_KIT_Tablet_F3 = ["ne"]
top_feature_KIT_Phone_F1 = ["BACKSPACEBACKSPACE"]
top_feature_KIT_Phone_F2 = ["BACKSPACEBACKSPACE"]
top_feature_KIT_Phone_F4 = ["BACKSPACEBACKSPACE"]


def top_feature_KIT(pickle_file, top_feature):
    kit_feature_dictionary = pickle.load(open(pickle_file, "rb"))
    selected_top_feature = [[0 for x in range(len(top_feature))] for x in range(116)]
    for key1 in kit_feature_dictionary:
        if key1 == 117:
            break
        for i in range(len(top_feature)):
            for key2 in kit_feature_dictionary[key1]:
                if str(top_feature[i]) == str(key2):
                    selected_top_feature[key1 - 1][i] = np.median(
                        kit_feature_dictionary[key1][key2]
                    )
                    break
    return selected_top_feature


def concatenated_feature_matrix_KIT():
    feature_KIT_Desktop_F1 = top_feature_KIT(
        "desktop_kit_feature_f1_dictionary.pickle", top_feature_KIT_Desktop_F1
    )
    feature_KIT_Tablet_F1 = top_feature_KIT(
        "tablet_kit_feature_f1_dictionary.pickle", top_feature_KIT_Tablet_F1
    )
    feature_KIT_Tablet_F2 = top_feature_KIT(
        "tablet_kit_feature_f2_dictionary.pickle", top_feature_KIT_Tablet_F2
    )
    feature_KIT_Tablet_F3 = top_feature_KIT(
        "tablet_kit_feature_f3_dictionary.pickle", top_feature_KIT_Tablet_F3
    )
    feature_KIT_Phone_F1 = top_feature_KIT(
        "phone_kit_feature_f1_dictionary.pickle", top_feature_KIT_Phone_F1
    )
    feature_KIT_Phone_F2 = top_feature_KIT(
        "phone_kit_feature_f2_dictionary.pickle", top_feature_KIT_Phone_F2
    )
    feature_KIT_Phone_F4 = top_feature_KIT(
        "phone_kit_feature_f4_dictionary.pickle", top_feature_KIT_Phone_F4
    )
    return np.concatenate(
        (
            np.array(feature_KIT_Desktop_F1),
            np.array(feature_KIT_Tablet_F1),
            np.array(feature_KIT_Tablet_F2),
            np.array(feature_KIT_Tablet_F3),
            np.array(feature_KIT_Phone_F1),
            np.array(feature_KIT_Phone_F2),
            np.array(feature_KIT_Phone_F4),
        ),
        axis=1,
    )
