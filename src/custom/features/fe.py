import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from custom.features.kht import get_KHT_features
from custom.features.kit import (
    get_dataframe_KIT,
    get_KIT_features_F2,
    get_KIT_features_F1,
    get_KIT_features_F3,
    get_KIT_features_F4,
)

from custom.features.word_level import get_advanced_word_features


def get_all_users_features_KIT(directory):
    users_feat_dict_f1 = {}
    users_feat_dict_f2 = {}
    users_feat_dict_f3 = {}
    users_feat_dict_f4 = {}
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            data_frame = pd.read_csv(directory + user_file)
            # print("Read:", data_frame)
            data_frame = get_dataframe_KIT(data_frame.values)
            # print("DataFrame:", data_frame)
            user_data = data_frame.values
            # print("User Data:", user_data)

            user_feat_dict_f1 = get_KIT_features_F1(user_data)
            users_feat_dict_f1[i + 1] = user_feat_dict_f1

            user_feat_dict_f2 = get_KIT_features_F2(user_data)
            users_feat_dict_f2[i + 1] = user_feat_dict_f2

            user_feat_dict_f3 = get_KIT_features_F3(user_data)
            users_feat_dict_f3[i + 1] = user_feat_dict_f3

            user_feat_dict_f4 = get_KIT_features_F4(user_data)
            users_feat_dict_f4[i + 1] = user_feat_dict_f4

    return (
        users_feat_dict_f1,
        users_feat_dict_f2,
        users_feat_dict_f3,
        users_feat_dict_f4,
    )


def get_all_users_features_KHT(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            print(user_file)
            data_frame = pd.read_csv(directory + user_file)
            user_data = data_frame.values
            user_feat_dict = get_KHT_features(user_data)
            users_feat_dict[i + 1] = user_feat_dict
    return users_feat_dict


def get_all_users_features_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory + user_file)
        user_data = data_frame.values
        user_feat_dict = get_word_features(user_data)
        users_feat_dict[i + 1] = user_feat_dict

    return users_feat_dict


def get_all_users_features_advanced_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory + user_file)
        user_data = data_frame.values
        processed_data = get_dataframe_KIT(user_data)
        processed_data = np.c_[np.arange(len(processed_data)), processed_data]
        processed_data = processed_data[np.argsort(processed_data[:, 2])]
        user_feat_dict = get_advanced_word_features(processed_data)
        users_feat_dict[i + 1] = user_feat_dict

    return users_feat_dict
