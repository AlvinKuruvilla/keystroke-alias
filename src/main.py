import os
import pickle
import pandas as pd
from rich.traceback import install

from data_processor.aggregator import (
    Platform,
    average_keystroke_counts,
    max_keystroke_count,
    min_keystroke_count,
    stdev_keystroke_count,
)
from fpd.dataset import TextDataset
from fpd.classifiers import xgb_classifier
from fpd.feature_gen import make_kht_features_file, make_kit_features_file
from parser.sentence_parser import SentenceParser

install()
from custom.features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
)


def collect_stats():
    for platform in Platform:
        print("Average Keystrokes for", platform, average_keystroke_counts(platform))
        print("Standard Deviation for", platform, stdev_keystroke_count(platform))
        print("Max Keystrokes for", platform, max_keystroke_count(platform))
        print("Min Keystrokes for", platform, min_keystroke_count(platform))


def pickle_features():
    dir_name = "data/"
    dir_path = os.path.join(os.getcwd(), dir_name, "km/")
    selected_profile_path = os.path.join(dir_path)
    desktop_kht_features = get_all_users_features_KHT(selected_profile_path)
    with open("desktop_kht_feature_dictionary.pickle", "wb") as handle:
        pickle.dump(desktop_kht_features, handle)
    (
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
    ) = get_all_users_features_KIT(selected_profile_path)
    with open("desktop_kit_feature_f1_dictionary.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f1, handle)
    with open("desktop_kit_feature_f2_dictionary.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f2, handle)
    with open("desktop_kit_feature_f3_dictionary.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f3, handle)
    with open("desktop_kit_feature_f4_dictionary.pickle", "wb") as handle:
        pickle.dump(desktop_kit_features_f4, handle)


def run_classifiers(use_csv: bool = False):
    xgb_classifier(use_csv)
    # bagged_decision_tree_classifier(use_csv)
    # adaboost(use_csv)
    # FIX: NOT WORKING
    # voting_ensemble(use_csv)
    # random_forrest(use_csv)


if __name__ == "__main__":
    # TODO: Pickle dump the desktop pickle files and add function for advanced word pickle file
    dir_name = "data/"
    dir_path = os.path.join(os.getcwd(), dir_name, "km/")
    selected_profile_path = os.path.join(dir_path)

    td = TextDataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/keystroke_features.txt"
    )

    sp = SentenceParser(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/data/km/f_18_fpd1.csv"
    )
    # print(sp.capital_letters_count_feature())
    # df = pd.read_csv(
    #     "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/data/km/f_17_fpd1.csv",
    #     header=None,
    # )
    td = TextDataset(
        "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-alias/kht_features.txt"
    )
    print(td.get_platform_series())
