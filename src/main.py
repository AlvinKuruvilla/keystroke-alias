import os
import pickle
from rich.traceback import install

from models.free_text_classification import run_free_text_xgboost_gender_ml_model

install()
from features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
    get_all_users_features_advanced_word,
)

if __name__ == "__main__":
    dir_name = "data"
    dir_path = os.path.join(os.getcwd(), dir_name)
    for file in os.listdir(os.path.join(dir_path, "Desktop/")):
        print(file)
    selected_profile_path = os.path.join(dir_path, "Desktop/")
    (
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
    ) = get_all_users_features_KIT(selected_profile_path)

    pickle.dump(
        desktop_kit_features_f1, open("desktop_kit_feature_f1_dictionary.pickle", "wb")
    )
    pickle.dump(
        desktop_kit_features_f2, open("desktop_kit_feature_f2_dictionary.pickle", "wb")
    )
    pickle.dump(
        desktop_kit_features_f3, open("desktop_kit_feature_f3_dictionary.pickle", "wb")
    )
    pickle.dump(
        desktop_kit_features_f4, open("desktop_kit_feature_f4_dictionary.pickle", "wb")
    )

    desktop_kht_features = get_all_users_features_KHT(selected_profile_path)

    pickle.dump(
        desktop_kht_features, open("desktop_kht_feature_dictionary.pickle", "wb")
    )
    desktop_advanced_word_features = get_all_users_features_advanced_word(
        selected_profile_path
    )

    pickle.dump(
        desktop_advanced_word_features,
        open("desktop_advanced_word_feature_dictionary.pickle", "wb"),
    )
    # FIXME: ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
    run_free_text_xgboost_gender_ml_model()
