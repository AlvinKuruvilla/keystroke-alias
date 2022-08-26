import numpy as np
import pickle
from tqdm import tqdm
from core.features.util import remove_outliers
from core.features.constants import *


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
    return np.concatenate((np.array(feature_KIT_Desktop_F1)), axis=1)


# Calculated median of required features
def top_feature_advanced_word(pickle_file, feature_dict):
    temp_features = pickle.load(open(pickle_file, "rb"))
    selected_top_feature = []
    top_feature_list = [[0 for x in range(len(temp_features))] for x in range(116)]
    # for key in feature_dict:
    for key1 in temp_features:
        if key1 == 117:
            break
        temp = []
        for key2 in temp_features[key1]:
            for key in feature_dict:
                if key2 == key:
                    for i in feature_dict[key]:
                        # Removing outliers
                        temp_without_outlier = remove_outliers(
                            np.array(temp_features[key1][key2])[:, i]
                        )

                        # Median feature
                        # temp.append(np.median(temp_without_outlier))

                        """# Mean feature
                        temp.append(np.mean(temp_without_outlier))

                        #IQR feature
                        a = np.asarray(temp_without_outlier)
                        upper_quartile = np.percentile(a, 75)
                        lower_quartile = np.percentile(a, 25)
                        IQR = (upper_quartile - lower_quartile) * 1.5
                        temp.append(IQR)
                        temp.append(upper_quartile)
                        temp.append(lower_quartile)

                        # Kurtosis feature
                        temp.append(kurtosis(temp_without_outlier))

                        # Skew features
                        temp.append(skew(temp_without_outlier))"""

                        # Mean-Median
                        temp.append(
                            np.mean(temp_without_outlier)
                            - np.median(temp_without_outlier)
                        )
                    break
            top_feature_list[key1 - 1] = list(temp)
    return top_feature_list


# Combines data from multiple devices (desktop, phone, tablet)
def combine_top_advanced_word():
    desktop_features = top_feature_advanced_word(
        "desktop_advanced_word_feature_dictionary.pickle",
        top_feature_advanced_word_Desktop_map,
    )

    return np.concatenate((np.array(desktop_features)), axis=1)


def combine_top_advanced_word_desktop_only():
    desktop_features = top_feature_advanced_word(
        "desktop_advanced_word_feature_dictionary.pickle",
        top_feature_advanced_word_Desktop_only_map,
    )
    return np.array(desktop_features)


# return all the desktop features for free text
def get_desktop_features():
    desktop_features_KHT = top_feature_KIT(
        "desktop_kht_feature_dictionary.pickle", feature_list_Desktop_KHT
    )
    desktop_features_KIT_1 = top_feature_KIT(
        "desktop_kit_feature_f1_dictionary.pickle", feature_list_Desktop_KIT_1
    )
    desktop_features_KIT_2 = top_feature_KIT(
        "desktop_kit_feature_f2_dictionary.pickle", feature_list_Desktop_KIT_2
    )
    desktop_features_KIT_3 = top_feature_KIT(
        "desktop_kit_feature_f3_dictionary.pickle", feature_list_Desktop_KIT_3
    )
    desktop_features_KIT_4 = top_feature_KIT(
        "desktop_kit_feature_f4_dictionary.pickle", feature_list_Desktop_KIT_4
    )

    desktop_features_advanced = top_feature_advanced_word(
        "desktop_advanced_word_feature_dictionary.pickle",
        feature_dict_advanced_word_Desktop,
    )
    return np.concatenate(
        (
            np.array(desktop_features_KHT),
            np.array(desktop_features_advanced),
            np.array(desktop_features_KIT_1),
            np.array(desktop_features_KIT_2),
            np.array(desktop_features_KIT_3),
            np.array(desktop_features_KIT_4),
        ),
        axis=1,
    )


def get_combined_features():
    desktop_features_combined = get_desktop_features()
    return np.concatenate((np.array(desktop_features_combined)), axis=1)
