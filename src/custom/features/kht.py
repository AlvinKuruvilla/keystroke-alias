import numpy as np
from custom.features.fe_util import conform_to_int, event_to_int

# get KHT feature based on current key and timing values
def get_KHT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    #! FIX: There is something with the values used in the subtraction to find the kht. The values used are not found in the csv used for the calculation
    #! Sometimes the subtraction result is also wrong as well
    for i, row in enumerate(keys_in_pipeline):

        key = row[1]
        timing = row[2]
        print("Key:", key)
        input("KEY")
        print("Timing:", timing)
        input("TIMING")
        if search_key == key:
            mask[i] = 0
            kht = int(float(search_key_timing)) - int(float(timing))
            print(search_key_timing, int(timing))
            input("TIMING")
            print(kht)
            input("KHT")
            non_zero_indices = np.nonzero(mask)
            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []
            print(
                key,
                int(float(search_key_timing)),
                int(float(timing)),
                int(float(search_key_timing)) - int(float(timing)),
            )
            input("KHT")
            return keys_in_pipeline, kht

    return keys_in_pipeline, None


# function to get KHT feature dictionary for a given user
def get_KHT_features(data):
    feature_dictionary = {}
    keys_in_pipeline = []

    for row_idx in range(len(data)):
        keys_in_pipeline = list(data)
        # print("keys_in_pipeline:", keys_in_pipeline)
        curr_key = data[row_idx][1]
        print("curr_key:", curr_key)
        print("Action:", data[row_idx][0])
        curr_direction = event_to_int(data[row_idx][0])
        curr_timing = data[row_idx][2]
        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_kht = get_KHT(
                conform_to_int(keys_in_pipeline), curr_key, curr_timing
            )
            if curr_kht is None:
                continue
            else:
                if curr_key in list(feature_dictionary.keys()):
                    feature_dictionary[curr_key].append(curr_kht)
                else:
                    feature_dictionary[curr_key] = []
                    feature_dictionary[curr_key].append(curr_kht)

    return feature_dictionary
