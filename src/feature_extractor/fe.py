import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from .kit import (
    get_KIT_features_F1_from_file,
    get_dataframe_KIT,
    get_KIT_features_F2,
    get_KIT_features_F1,
    get_KIT_features_F4,
)

# FIXME:Where is get_word_features
from .word_level import get_advanced_word_features

# FIXME We have to get the F3 and F4 features but currently they cause a crash
def get_all_users_features_KIT(directory):
    users_feat_dict_f1 = {}
    users_feat_dict_f2 = {}
    # users_feat_dict_f3 = {}
    # users_feat_dict_f4 = {}
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory + user_file)
        data_frame = get_dataframe_KIT(data_frame.values)
        user_data = data_frame.values

        user_feat_dict_f1 = get_KIT_features_F1(user_data)
        users_feat_dict_f1[i + 1] = user_feat_dict_f1

        user_feat_dict_f2 = get_KIT_features_F2(user_data)
        users_feat_dict_f2[i + 1] = user_feat_dict_f2

        # user_feat_dict_f3 = get_KIT_features_F1_from_file(user_data)
        # users_feat_dict_f3[i + 1] = user_feat_dict_f3

        # user_feat_dict_f4 = get_KIT_features_F4(user_data)
        # users_feat_dict_f4[i + 1] = user_feat_dict_f4

    return (
        users_feat_dict_f1,
        users_feat_dict_f2,
        # users_feat_dict_f3,
        # users_feat_dict_f4,
    )


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
