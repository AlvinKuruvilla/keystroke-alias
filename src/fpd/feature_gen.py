import itertools
import numpy as np
import statistics
import pandas as pd
from custom.features.kht import get_KHT_features
from custom.features.kit import (
    get_KIT_features_F1,
    get_KIT_features_F2,
    get_KIT_features_F3,
    get_KIT_features_F4,
    get_dataframe_KIT,
)


def kht_for_file(path: str):
    data_frame = pd.read_csv(path)
    user_data = data_frame.values
    user_feat_dict = get_KHT_features(user_data)
    return user_feat_dict


def kit_for_file(path: str, kit_feature_index: int):
    data_frame = pd.read_csv(path)
    data_frame = get_dataframe_KIT(data_frame.values)
    user_data = data_frame.values
    assert 1 <= kit_feature_index <= 4
    if kit_feature_index == 1:
        return get_KIT_features_F1(user_data)
    elif kit_feature_index == 2:
        return get_KIT_features_F2(user_data)
    elif kit_feature_index == 3:
        return get_KIT_features_F3(user_data)
    elif kit_feature_index == 4:
        return get_KIT_features_F4(user_data)


def feature_average(feature_dict):
    timings = list(feature_dict.values())
    combined = list(itertools.chain.from_iterable(timings))
    return np.mean(combined)


def feature_median(feature_dict):
    timings = list(feature_dict.values())
    combined = list(itertools.chain.from_iterable(timings))
    return np.median(combined)


def feature_mode(feature_dict):
    timings = list(feature_dict.values())
    combined = list(itertools.chain.from_iterable(timings))
    return statistics.mode(combined)


def feature_standard_deviation(feature_dict):
    timings = list(feature_dict.values())
    combined = list(itertools.chain.from_iterable(timings))
    return np.std(combined)
