import os
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


def path_to_platform(path: str):
    filename = os.path.basename(path)
    return filename.split("_")[0]


def kht_from_dataframe(df: pd.DataFrame):
    user_data = df.values
    user_feat_dict = get_KHT_features(user_data)
    return user_feat_dict


def kit_from_dataframe(df: pd.DataFrame, kit_feature_index: int):
    df = get_dataframe_KIT(df.values)
    user_data = df.values
    assert 1 <= kit_feature_index <= 4
    if kit_feature_index == 1:
        return get_KIT_features_F1(user_data)
    elif kit_feature_index == 2:
        return get_KIT_features_F2(user_data)
    elif kit_feature_index == 3:
        return get_KIT_features_F3(user_data)
    elif kit_feature_index == 4:
        return get_KIT_features_F4(user_data)


def split_into_four(df: pd.DataFrame):
    ret = []
    i = 4
    # df.index represents the number of rows in the dataframe
    for j in range(0, len(df.index), 4):
        ret.append(df.iloc[j:i])
        i += 4
    return ret


def merge_dataframes(df_list):
    return pd.concat(df_list, axis=0, ignore_index=True)


# Handles strings like "<0>""
def remove_invalid_keystrokes(data):
    for i in range(0, len(data)):
        df = data[i]
        for row in df.itertuples():
            # print(row[2])
            if row[2] == "<0>":
                # print("HERE")
                num = int(row.Index)
                # print(num)
                rem = df.drop(index=num)
                data[i] = rem
    # After removing the weird values the size of each dataframe element is
    # smaller so we need to coalesce. Re-partitioning will be the job of
    # subsequent methods that use the return value of this method
    return merge_dataframes(data)


def dictionary_to_flat_list(d):
    res = []
    temp = []
    for key, value_list in d.items():
        temp.append(key)
        temp.extend(list(value_list))
        res.append(temp)
        temp = []
    return res


def make_features_file(directory: str):
    user_id = 1
    # classification refers to being genuine (0) or a fake profile (1)
    classification = 0
    header = ["ID, Platform, Char, Timing, Class"]
    user_files = os.listdir(directory)
    with open("keystroke_features.txt", "w") as f:
        f.write(str(header))
        f.write("\n")
        for i in tqdm(range(len(user_files))):
            user_file = user_files[i]
            if ".csv" in user_file and not user_file.startswith("."):
                print(user_file)
                path = directory + user_file
                df = pd.read_csv(path, header=None)
                data = split_into_four(df)
                df = remove_invalid_keystrokes(data)
                user_data_kht = kht_from_dataframe(df)
                user_data_kit_f1 = kit_from_dataframe(df, 1)
                user_data_kit_f2 = kit_from_dataframe(df, 2)
                user_data_kit_f3 = kit_from_dataframe(df, 3)
                user_data_kit_f4 = kit_from_dataframe(df, 4)

                kht_data_row = dictionary_to_flat_list(user_data_kht)
                # print(kht_data_row)
                kit_data_row_f1 = dictionary_to_flat_list(user_data_kit_f1)
                kit_data_row_f2 = dictionary_to_flat_list(user_data_kit_f2)
                kit_data_row_f3 = dictionary_to_flat_list(user_data_kit_f3)
                kit_data_row_f4 = dictionary_to_flat_list(user_data_kit_f4)

                if (
                    path_to_platform(path).upper() == "F"
                    or path_to_platform(path).upper() == "I"
                ):
                    classification = 0
                    classification = str(classification)
                elif path_to_platform(path).upper() == "T":
                    classification = 1
                    classification = str(classification)

                for sub_list in kht_data_row:
                    f.write(str(user_id) + " ")
                    f.write(path_to_platform(path).upper())
                    print(sub_list)
                    # input()
                    f.write(str(sub_list) + " ")
                    f.write(classification)
                    f.write("\n")
                for sub_list in kit_data_row_f1:
                    f.write(str(user_id) + " ")
                    f.write(path_to_platform(path).upper())
                    f.write(str(sub_list) + " ")
                    f.write(classification)
                    f.write("\n")

                for sub_list in kit_data_row_f2:
                    f.write(str(user_id) + " ")
                    f.write(path_to_platform(path).upper())
                    f.write(str(sub_list) + " ")
                    f.write(classification)
                    f.write("\n")

                for sub_list in kit_data_row_f3:
                    f.write(str(user_id) + " ")
                    f.write(path_to_platform(path).upper())
                    f.write(str(sub_list) + " ")
                    f.write(classification)
                    f.write("\n")
                for sub_list in kit_data_row_f4:
                    f.write(str(user_id) + " ")
                    f.write(path_to_platform(path).upper())
                    f.write(str(sub_list) + " ")
                    f.write(classification)
                    f.write("\n")
                user_id += 1


# [id, key, timings]
# 1, [A, 00000,1111,222,333,44], 0
