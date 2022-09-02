import os
import pandas as pd
import pickle
from rich.traceback import install
from core.tasks.cnn_gender_task import train_model_cnn
from custom.features.fe_util import load_feature_file, pickle_all_feature_data

from core.tasks.xgb_regression_age import run_age_xgb_regression

# from custom.models.rnn_gender import train_model
from core.tasks.rnn_gender_task import train_model
from fpd.ensemble import adaboost, bagged_decision_tree_classifier, voting_ensemble
from fpd.feature_gen import (
    generate_features_file,
    kht_for_file,
    kht_from_dataframe,
    kit_from_dataframe,
    make_features_file,
    remove_invalid_keystrokes,
    split_into_four,
)
from fpd.classifiers import create_svm, random_forrest, xgb_classifier

install()
from custom.features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
)


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


if __name__ == "__main__":
    # TODO: Pickle dump the desktop pickle files and add function for advanced word pickle file
    dir_name = "data/"
    dir_path = os.path.join(os.getcwd(), dir_name, "km/")
    selected_profile_path = os.path.join(dir_path)

    # run_age_xgb_regression()
    # train_model("Gender")
    # train_model_cnn("Gender")
    # generate_features_file(selected_profile_path)

    # xgb_classifier()
    # bagged_decision_tree_classifier()
    # adaboost()
    # voting_ensemble()
    # random_forrest()
    # xgb_classifier()
    make_features_file(selected_profile_path)
