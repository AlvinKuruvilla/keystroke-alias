import numpy as np
import pandas as pd

# get KIT feature based on current key and timing values
def get_timings_KIT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for i, (key, timing) in enumerate(keys_in_pipeline):
        if search_key == key:
            mask[i] = 0
            non_zero_indices = np.nonzero(mask)

            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []

            return keys_in_pipeline, timing, search_key_timing
    return keys_in_pipeline, None, None


# function to get KIT data frame with key, press_time, release_time for a given user
def get_dataframe_KIT(data):
    """Input: data  Output: Dataframe with (key, press_time, release_time)"""
    feature_dictionary = {}
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_start, curr_end = get_timings_KIT(
                keys_in_pipeline, curr_key, curr_timing
            )
            if curr_start is None:
                continue
            else:
                result_key.append(curr_key)
                press.append(curr_start)
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(
        list(zip(result_key, press, release)),
        columns=["Key", "Press_Time", "Release_Time"],
    )
    return resultant_data_frame


# function to get Flight1 KIT feature dictionary for a given user
def get_KIT_features_F1(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][1]

        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight2 KIT feature dictionary for a given user
def get_KIT_features_F2(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][1]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F3(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][2]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F4(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][2]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary
