import numpy as np
import pickle
import math
from .fe_util import remove_outliers
from .processor import top_feature_KIT


# TODO: How are these generated?
feature_list_Desktop_KHT = [
    "v",
    "n",
    "s",
    "h",
    "BACKSPACE",
    "r",
    "u",
    "m",
    "i",
    "p",
    "t",
    "d",
    "o",
    "SPACE",
    "l",
    "q",
    "e",
    ".",
    "c",
    "w",
    "f",
    "a",
]
feature_list_Desktop_KIT_1 = [
    "ir",
    "ot",
    "ve",
    "no",
    "ha",
    "he",
    "to",
    "rd",
    "hSPACE",
    "is",
    "co",
    "un",
    "pSPACE",
    "di",
    "fi",
    "ni",
    "pl",
    "ne",
    "fSPACE",
    "hi",
    "tSPACE",
    "es",
    "eSPACE",
    "on",
    "dSPACE",
    "oSPACE",
    "rl",
    "me",
    "le",
    "wi",
    "if",
    "nc",
    "lSPACE",
    "SPACEs",
    "SPACEa",
    "nt",
    "pe",
    "or",
    "iSPACE",
    "wo",
    "SPACEt",
    "ca",
    "nd",
    "ce",
    "ly",
    "ov",
    "te",
    "ed",
    "th",
    "tw",
    "ff",
    "in",
    "aSPACE",
    "SPACEl",
    "li",
    "se",
    "it",
    "la",
    "SPACEn",
    "of",
    "ul",
    "ec",
    "ct",
    "yp",
    "et",
    "SPACEc",
    "io",
    "sa",
    "ts",
    "re",
    "SPACEf",
    "ti",
    "il",
    "ap",
    "ue",
    "ySPACE",
    "ds",
    "SPACEh",
    "el",
    "da",
    "SPACEo",
    "ol",
    "ll",
    "ef",
    "SPACEp",
    "ta",
    "st",
    "am",
    "SPACEw",
    "ar",
    "rs",
    "en",
    "er",
    "sSPACE",
    "qu",
    "s.",
    "at",
    ".SPACE",
    "nSPACE",
    "SPACEu",
    "SPACEm",
    "as",
    "SPACEi",
    "e.",
    "SPACEd",
    "si",
    "ee",
    "ss",
]

feature_list_Desktop_KIT_2 = [
    "rd",
    "rl",
    "to",
    "un",
    "ap",
    "ot",
    "el",
    "ol",
    "ir",
    "ef",
    "rs",
    "ov",
    "yp",
    "ly",
    "nSPACE",
    "SPACEm",
    "ue",
    "if",
    "no",
    "nt",
    "ve",
    "ed",
    "SPACEs",
    "iSPACE",
    "es",
    "co",
    "ee",
    "SPACEn",
    "wi",
    "st",
    "ni",
    "it",
    "ct",
    "nc",
    "et",
    "hi",
    "ce",
    "pSPACE",
    "oSPACE",
    "hSPACE",
    "SPACEf",
    "pe",
    "sa",
    "se",
    "re",
    "of",
    "ts",
    "SPACEu",
    "di",
    "la",
    "ff",
    "si",
    "aSPACE",
    "ta",
    "ti",
    "en",
    "ySPACE",
    "pl",
    "ss",
    "da",
    "fSPACE",
    "li",
    "tw",
    "lSPACE",
    "th",
    "SPACEc",
    "il",
    "ne",
    "is",
    "te",
    "or",
    "le",
    "ul",
    "SPACEp",
    "as",
    "SPACEl",
    "s.",
    ".SPACE",
    "in",
    "me",
    "ds",
    "dSPACE",
    "SPACEa",
    "er",
    "nd",
    "SPACEt",
    "fi",
    "he",
    "ha",
    "qu",
    "ll",
    "ec",
    "am",
    "SPACEd",
    "on",
    "ca",
    "at",
    "sSPACE",
    "SPACEi",
    "SPACEo",
    "wo",
    "tSPACE",
    "SPACEh",
    "io",
    "eSPACE",
    "e.",
    "SPACEw",
    "ar",
]

feature_list_Desktop_KIT_3 = [
    "un",
    "to",
    "li",
    "rl",
    "es",
    "nt",
    "ce",
    "hSPACE",
    "th",
    "iSPACE",
    "rd",
    "fi",
    "fSPACE",
    "el",
    "pl",
    "si",
    "ef",
    "en",
    "ff",
    "SPACEs",
    "oSPACE",
    "ed",
    "co",
    "ov",
    "ap",
    "ni",
    "if",
    "dSPACE",
    "il",
    "ll",
    "SPACEn",
    "sa",
    "ds",
    "ot",
    "as",
    "ir",
    "et",
    "pSPACE",
    "SPACEc",
    "tw",
    "ca",
    "ve",
    "he",
    "it",
    "lSPACE",
    "SPACEt",
    "ar",
    "SPACEf",
    "in",
    "ue",
    "am",
    "ct",
    "la",
    "no",
    "wi",
    "nSPACE",
    "ti",
    "ne",
    "ec",
    "SPACEp",
    "ta",
    "is",
    "st",
    "rs",
    "pe",
    "ss",
    "or",
    "re",
    "hi",
    "io",
    "ts",
    "SPACEw",
    "se",
    "aSPACE",
    "ySPACE",
    "ha",
    "nc",
    "ol",
    "da",
    "of",
    "SPACEu",
    "le",
    "ly",
    "SPACEm",
    "SPACEl",
    "on",
    "at",
    "s.",
    "sSPACE",
    "yp",
    "di",
    "SPACEa",
    "te",
    "ee",
    "SPACEd",
    "eSPACE",
    "er",
    "SPACEi",
    "ul",
    ".SPACE",
    "me",
    "qu",
    "e.",
    "SPACEo",
    "wo",
    "SPACEh",
    "nd",
    "tSPACE",
]

feature_list_Desktop_KIT_4 = [
    "un",
    "pl",
    "rl",
    "rd",
    "ol",
    "in",
    "as",
    "ds",
    "en",
    "me",
    "ef",
    "to",
    "iSPACE",
    "ll",
    "ir",
    "nt",
    "fSPACE",
    "ot",
    "at",
    "ar",
    "ee",
    "if",
    "am",
    "fi",
    "el",
    "SPACEm",
    "co",
    "ed",
    "ap",
    "SPACEu",
    "di",
    "aSPACE",
    "hSPACE",
    "ly",
    "hi",
    "ov",
    "it",
    "SPACEn",
    "ct",
    "SPACEi",
    "ff",
    "ca",
    "tw",
    "ce",
    "SPACEf",
    "es",
    "wo",
    "ni",
    "si",
    "eSPACE",
    "et",
    "ti",
    "th",
    "nc",
    "ySPACE",
    "da",
    "SPACEs",
    "qu",
    "pe",
    "er",
    "SPACEl",
    "or",
    "oSPACE",
    "nSPACE",
    "li",
    "SPACEc",
    "ul",
    "pSPACE",
    "st",
    "of",
    "il",
    "wi",
    "on",
    "SPACEp",
    "la",
    "yp",
    "s.",
    "SPACEt",
    "sSPACE",
    "rs",
    "dSPACE",
    "he",
    "sa",
    "ec",
    "te",
    "lSPACE",
    "ta",
    "ha",
    "SPACEw",
    "nd",
    "ss",
    "ue",
    "ts",
    "no",
    "ne",
    "ve",
    "io",
    "is",
    "SPACEd",
    ".SPACE",
    "SPACEa",
    "e.",
    "SPACEo",
    "se",
    "SPACEh",
    "tSPACE",
    "le",
    "re",
]
feature_dict_advanced_word_Desktop = {
    "if": [2, 1, 3, 4, 6, 0, 13, 15, 10, 12, 7, 9],
    "this": [6, 1, 3, 2, 8, 10, 4, 5, 12, 13, 15, 0, 7, 9, 14, 11],
    "have": [6, 13, 3, 1, 0, 7, 4, 8, 15, 5, 12, 10, 9, 14, 2, 11],
    "me": [4, 6, 0, 13, 15, 2, 7, 9, 10, 12, 1, 3],
    "with": [6, 13, 9, 0, 12, 4, 10, 15, 7, 1, 5, 11, 8, 2, 14, 3],
    "to": [4, 6, 0, 13, 15, 10, 12, 7, 9, 1, 3, 2],
    "sentences": [13, 6, 5, 8, 11, 14, 3, 4, 10, 0, 1, 7, 15, 2, 12, 9],
    "not": [1, 2, 3, 4, 6, 10, 12, 13, 15, 8, 11, 0, 7, 9, 5, 14],
    "type": [1, 3, 11, 13, 5, 14, 4, 10, 2, 8, 12, 7, 0, 6, 15, 9],
    "words": [11, 14, 8, 5, 1, 6, 9, 4, 13, 15, 3, 7, 0, 12, 10, 2],
    "will": [1, 11, 5, 3, 15, 2, 13, 14, 12, 4, 9, 10, 8, 0, 6, 7],
    "carefully": [1, 4, 3, 10, 0, 15, 14, 7, 13, 5, 8, 11, 9, 12, 6, 2],
    "different": [6, 3, 8, 15, 9, 1, 12, 5, 14, 4, 11, 7, 13, 10, 0, 2],
    "two": [1, 14, 11, 3, 8, 5, 13, 15, 7, 9, 10, 12, 4, 6, 0, 2],
    "see": [14, 8, 13, 15, 3, 0, 1, 10, 12, 5, 7, 9, 4, 6, 2, 11],
    "first": [1, 5, 8, 14, 6, 4, 10, 3, 0, 15, 12, 7, 2, 9, 13, 11],
    "sample": [4, 5, 8, 1, 14, 13, 15, 10, 3, 0, 11, 2, 7, 6, 12, 9],
    "sets": [6, 3, 1, 9, 10, 12, 7, 5, 4, 13, 8, 15, 0, 11, 14, 2],
    "that": [6, 1, 3, 7, 0, 8, 10, 2, 9, 14, 5, 12, 4, 13, 15, 11],
    "overlap": [1, 10, 14, 15, 6, 3, 8, 9, 11, 2, 0, 7, 5, 13, 12, 4],
    "collection": [15, 6, 10, 1, 12, 14, 7, 8, 13, 4, 11, 0, 9, 3, 5, 2],
    "is": [4, 6, 2, 10, 12, 1, 3, 7, 9, 0, 13, 15],
    "there": [1, 3, 2, 6, 4, 8, 5, 11, 15, 10, 14, 7, 9, 13, 0, 12],
    "data": [1, 3, 8, 2, 4, 5, 7, 9, 13, 0, 6, 10, 11, 12, 15, 14],
    "of": [1, 3, 4, 6, 10, 12, 0, 13, 15, 7, 9, 2],
    "test": [13, 11, 0, 14, 15, 5, 3, 1, 7, 4, 8, 6, 2, 10, 12, 9],
    "are": [2, 5, 13, 15, 4, 6, 8, 3, 10, 12, 0, 11, 1, 14, 7, 9],
    "lines": [6, 11, 5, 15, 4, 13, 10, 3, 8, 14, 12, 1, 0, 9, 2, 7],
    "the": [1, 4, 6, 2, 8, 10, 12, 7, 9, 3, 13, 15, 0, 11, 5, 14],
    "in": [1, 3, 4, 6, 0, 13, 15, 7, 9, 10, 12, 2],
    "selected": [13, 0, 15, 5, 1, 6, 9, 3, 14, 12, 10, 8, 7, 11, 4, 2],
    "a": [0, 1, 3],
    "i": [0, 1, 3],
}

word_feature_id_mapping = {
    1: "wht",
    2: "avg_kht",
    3: "std_kht",
    4: "median_kht",
    5: "avg_f1",
    6: "std_f1",
    7: "median_f1",
    8: "avg_f2",
    9: "std_f2",
    10: "median_f2",
    11: "avg_f3",
    12: "std_f3",
    13: "median_f3",
    14: "avg_f4",
    15: "std_f4",
    16: "median_f4",
}
top_feature_advanced_word_Desktop_only_map = {}
top_feature_advanced_word_Desktop_only_map["if"] = [2, 1, 3, 4, 6, 0, 13]
top_feature_advanced_word_Desktop_only_map["this"] = [6, 1, 3]
top_feature_advanced_word_Desktop_only_map["have"] = [6]
top_feature_advanced_word_Desktop_only_map["me"] = [4, 6]
top_feature_advanced_word_Desktop_only_map["with"] = [6]
top_feature_advanced_word_Desktop_only_map["to"] = [4, 6]
top_feature_advanced_word_Desktop_only_map["sentences"] = [13, 6]
top_feature_advanced_word_Desktop_only_map["not"] = [1]
top_feature_advanced_word_Desktop_only_map["type"] = [1]
top_feature_advanced_word_Desktop_only_map["words"] = [11, 8, 14]
top_feature_advanced_word_Desktop_only_map["will"] = [1]
top_feature_advanced_word_Desktop_only_map["carefully"] = [1]


# List of top 25 features and their equivalent position
top_feature_advanced_word_Desktop_map = {}
top_feature_advanced_word_Desktop_map["selected"] = [0, 7, 10]
top_feature_advanced_word_Desktop_map["me"] = [4, 6]
top_feature_advanced_word_Desktop_map["if"] = [4, 6]
top_feature_advanced_word_Desktop_map["that"] = [4]
top_feature_advanced_word_Desktop_map["sample"] = [4]
top_feature_advanced_word_Desktop_map["test"] = [4]
top_feature_advanced_word_Desktop_map["have"] = [4]
top_feature_advanced_word_Desktop_map["data"] = [1, 4]
top_feature_advanced_word_Desktop_map["with"] = [13]
top_feature_advanced_word_Desktop_map["will"] = [13]

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
    return np.concatenate(
        (np.array(desktop_features),),
        axis=1,
    )


def combine_top_advanced_word_desktop_only():
    desktop_features = top_feature_advanced_word(
        "desktop_advanced_word_feature_dictionary.pickle",
        top_feature_advanced_word_Desktop_only_map,
    )
    return np.array(desktop_features)


# utilities to get the most commonly occurring words and corresponding features for all users
word_feature_id_mapping = {
    1: "wht",
    2: "avg_kht",
    3: "std_kht",
    4: "median_kht",
    5: "avg_f1",
    6: "std_f1",
    7: "median_f1",
    8: "avg_f2",
    9: "std_f2",
    10: "median_f2",
    11: "avg_f3",
    12: "std_f3",
    13: "median_f3",
    14: "avg_f4",
    15: "std_f4",
    16: "median_f4",
}


def get_key_array_for_user(user_key, user_dict):
    key_array_lengths = []
    for key in user_dict[user_key]:
        key_array_lengths.append((len(user_dict[user_key][key]), key))
    final_keys_array = []
    for key in key_array_lengths:
        final_keys_array.append(key[1])
    return final_keys_array


def get_top_word_keys(device):
    advanced_word_feat_dict = pickle.load(
        open(device + "_advanced_word_feature_dictionary.pickle", "rb")
    )
    final_key_set = get_key_array_for_user(1, advanced_word_feat_dict)
    for user_id in advanced_word_feat_dict:
        user_key_array = get_key_array_for_user(user_id, advanced_word_feat_dict)
        final_key_set = set(user_key_array).intersection(final_key_set)
    return final_key_set


def get_advanced_word_values_given_user_and_key(
    advanced_word_feat_dict, user_key, key, word_feat_id
):
    try:
        temp = list(
            np.asarray(advanced_word_feat_dict[user_key + 1][key])[:, word_feat_id]
        )
        for k in temp:
            if math.isnan(k):
                return int(0)
        return abs(
            np.median(
                np.asarray(advanced_word_feat_dict[user_key + 1][key])[:, word_feat_id]
            )
        )
    except KeyError as e:
        return None


def get_advanced_word_features(device):
    """Input: All feature dictionary Output: Feature matrix with unique columns"""
    features = pickle.load(
        open(device + "_advanced_word_feature_dictionary.pickle", "rb")
    )

    # Getting unique columns by removing repeated keys
    feature_set = list(get_top_word_keys(device))
    val = []
    rows, cols = (len(features), len(feature_set))
    feature_vector = [[0 for x in range(cols * 16)] for x in range(rows)]

    for i, key in enumerate(feature_set):
        for j in range(16):
            val.append(get_advanced_word_values_given_user_and_key(features, i, key, j))
        feature_vector[i] = val

    return feature_vector


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
    # TODO: The pickle files for F3 and F4 are not generating properly yet
    desktop_features_KIT_3 = top_feature_KIT(
        "desktop_kit_feature_f3_dictionary.pickle", feature_list_Desktop_KIT_3
    )
    desktop_features_KIT_4 = top_feature_KIT(
        "desktop_kit_feature_f4_dictionary.pickle", feature_list_Desktop_KIT_4
    )

    # desktop_features_advanced = top_feature_advanced_word(
    #     "desktop_advanced_word_feature_dictionary.pickle",
    #     feature_dict_advanced_word_Desktop,
    # )
    return np.concatenate(
        (
            np.array(desktop_features_KHT),
            # np.array(desktop_features_advanced),
            np.array(desktop_features_KIT_1),
            np.array(desktop_features_KIT_2),
            np.array(desktop_features_KIT_3),
            np.array(desktop_features_KIT_4),
        ),
        axis=1,
    )


def get_combined_features():
    desktop_features_combined = get_desktop_features()
    return np.concatenate((np.array(desktop_features_combined),), axis=1)
