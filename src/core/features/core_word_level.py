import numpy as np

""" function to return word level statistics for each extracted word.
These features are as follows:
1) Word hold time
2) Average, Standard Deviation and Median of all key hold times in the word
3) Average, Standard Deviation and Median of all flight 1 features for all digraphs in the word
4) Average, Standard Deviation and Median of all flight 2 features for all digraphs in the word
5) Average, Standard Deviation and Median of all flight 3 features for all digraphs in the word
6) Average, Standard Deviation and Median of all flight 4 features for all digraphs in the word
"""


def get_advanced_word_level_features(words_in_pipeline):
    def get_word_hold(words_in_pipeline):
        return int(float(words_in_pipeline[-1][2])) - int(
            float(words_in_pipeline[0][1])
        )

    def get_avg_std_median_key_hold(words_in_pipeline):
        key_holds = []
        for _, press, release in words_in_pipeline:
            key_holds.append(int(float(release)) - int(float(press)))
        return np.mean(key_holds), np.std(key_holds), np.median(key_holds)

    def get_avg_std_median_flights(words_in_pipeline):
        flights_1 = []
        flights_2 = []
        flights_3 = []
        flights_4 = []
        for i in range(len(words_in_pipeline) - 1):
            k1_r = words_in_pipeline[i][2]
            k1_p = words_in_pipeline[i][1]
            k2_r = words_in_pipeline[i + 1][2]
            k2_p = words_in_pipeline[i + 1][1]
            flights_1.append(int(float(k2_p)) - int(float(k1_r)))
            flights_2.append(int(float(k2_r)) - int(float(k1_r)))
            flights_3.append(int(float(k2_p)) - int(float(k1_p)))
            flights_4.append(int(float(k2_r)) - int(float(k1_p)))
        return (
            np.mean(flights_1),
            np.std(flights_1),
            np.median(flights_1),
            np.mean(flights_2),
            np.std(flights_2),
            np.median(flights_2),
            np.mean(flights_3),
            np.std(flights_3),
            np.median(flights_3),
            np.mean(flights_4),
            np.std(flights_4),
            np.median(flights_4),
        )

    wh = get_word_hold(words_in_pipeline)
    avg_kh, std_kh, median_kh = get_avg_std_median_key_hold(words_in_pipeline)
    (
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ) = get_avg_std_median_flights(words_in_pipeline)
    return [
        wh,
        avg_kh,
        std_kh,
        median_kh,
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ]


# function to get the advanced word level features of every user
def get_advanced_word_features(processed_data):
    words_in_pipeline = []
    feature_dictionary = {}

    ignore_keys = ["LCTRL", "RSHIFT", "TAB", "DOWN"]
    delimiter_keys = ["SPACE", ".", ",", "RETURN"]

    for row_idx in range(len(processed_data)):
        curr_key = processed_data[row_idx][1]
        curr_press = processed_data[row_idx][2]
        curr_release = processed_data[row_idx][3]

        if curr_key in ignore_keys:
            continue

        if curr_key in delimiter_keys:
            if len(words_in_pipeline) > 0:
                advanced_word_features = get_advanced_word_level_features(
                    words_in_pipeline
                )
                key_word = ""
                for char, _, _ in words_in_pipeline:
                    key_word = key_word + str(char)

                if key_word in list(feature_dictionary.keys()):
                    feature_dictionary[key_word].append(advanced_word_features)
                else:
                    feature_dictionary[key_word] = []
                    feature_dictionary[key_word].append(advanced_word_features)
            words_in_pipeline = []
            continue

        if curr_key == "BACKSPACE":
            words_in_pipeline = words_in_pipeline[:-1]
            continue

        words_in_pipeline.append([curr_key, curr_press, curr_release])

    return feature_dictionary
