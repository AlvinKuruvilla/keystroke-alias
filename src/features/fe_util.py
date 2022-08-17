from re import L
import numpy as np
import pickle

# Remove outlier points in the distribution using the 1.5IQR rule
def remove_outliers(x):
    a = np.asarray(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            result.append(y)
    return result


def event_to_int(event: str) -> int:
    if event == "P":
        return 1
    elif event == "R":
        return 0


# Make sure that the direction of the event is an integer rather than "P" or "R"
def conform_to_int(data):
    result = []
    for row_idx in data:
        result.append([event_to_int(row_idx[0]), row_idx[1], row_idx[2]])
    return result


def pickle_all_feature_data(
    desktop_kit_features_f1,
    desktop_kit_features_f2,
    desktop_kit_features_f3,
    desktop_kit_features_f4,
    desktop_kht_features,
):
    with open("kht.pickle", "wb") as handle:
        pickle.dump(desktop_kht_features, handle)
    with open("kit_f1.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f1, handle)
    with open("kit_f2.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f2, handle)
    with open("kit_f3.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f3, handle)
    with open("kit_f4.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f4, handle)


def load_feature_file(feature_file_path: str):
    with open(feature_file_path, "rb") as handle:
        b = pickle.load(handle)
    return b
