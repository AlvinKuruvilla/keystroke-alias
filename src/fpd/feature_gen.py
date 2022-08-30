import itertools
import os
import pickle
import csv
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm
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


def all_kht_features(path: str):
    kht = []
    feature_dict = kht_for_file(path)
    kht.append(feature_average(feature_dict))
    kht.append(feature_median(feature_dict))
    kht.append(feature_mode(feature_dict))
    kht.append(feature_standard_deviation(feature_dict))
    return kht


def all_kit_features(path: str):
    kit = []
    i = 1
    while i <= 4:
        feature_dict = kit_for_file(path, i)
        kit.append(feature_average(feature_dict))
        kit.append(feature_median(feature_dict))
        kit.append(feature_mode(feature_dict))
        kit.append(feature_standard_deviation(feature_dict))
        i += 1
    return kit


def generate_features_file(directory: str):
    user_files = os.listdir(directory)
    header = [
        "ID",
        "KHT_Mean",
        "KHT_Median",
        "KHT_Mode",
        "KHT_Stdev",
        "KIT_F1_Mean",
        "KIT_F1_Median",
        "KIT_F1_Mode",
        "KIT_F1_Stdev",
        "KIT_F2_Mean",
        "KIT_F2_Median",
        "KIT_F2_Mode",
        "KIT_F2_Stdev",
        "KIT_F3_Mean",
        "KIT_F3_Median",
        "KIT_F3_Mode",
        "KIT_F3_Stdev",
        "KIT_F4_Mean",
        "KIT_F4_Median",
        "KIT_F4_Mode",
        "KIT_F4_Stdev",
    ]
    all_features = []
    user_count = 1
    with open("all_features.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        user_files = os.listdir(directory)
        for i in tqdm(range(len(user_files))):
            user_file = user_files[i]
            if ".csv" in user_file and not user_file.startswith("."):
                path = directory + user_file
                print(path)
                kht = all_kht_features(path)
                kit = all_kit_features(path)
                all_features.append(user_count)
                all_features = all_features + kht + kit
                print(all_features)
                writer.writerow(all_features)
                user_count += 1
                all_features.clear()
