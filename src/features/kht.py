import numpy as np

# get KHT feature based on current key and timing values
def get_KHT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)

    for i, (key, timing) in enumerate(keys_in_pipeline):
        if search_key == key:
            mask[i] = 0
            kht = int(float(search_key_timing)) - int(float(timing))
            non_zero_indices = np.nonzero(mask)
            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []
            return keys_in_pipeline, kht

    return keys_in_pipeline, None


# function to get KHT feature dictionary for a given user
def get_KHT_features(data):
    feature_dictionary = {}
    keys_in_pipeline = []

    for row_idx in range(len(data)):
        keys_in_pipeline = list(data)
        curr_key = data[row_idx][0]
        curr_direction = data[row_idx][1]
        curr_timing = data[row_idx][2]

        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_kht = get_KHT(
                keys_in_pipeline, curr_key, curr_timing
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
