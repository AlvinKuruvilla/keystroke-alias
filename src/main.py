import os
from rich.traceback import install
from features.fe_util import pickle_all_feature_data

from models.regression_age_task import run_age_xgb_regression
from models.rnn_gender import train_model


install()
from features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
)

if __name__ == "__main__":
    dir_name = "data/"
    dir_path = os.path.join(os.getcwd(), dir_name, "km/")
    selected_profile_path = os.path.join(dir_path)
    desktop_kht_features = get_all_users_features_KHT(selected_profile_path)
    # print(desktop_kht_features)
    print("BEFORE MODEL:", len(desktop_kht_features))
    (
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
    ) = get_all_users_features_KIT(selected_profile_path)
    pickle_all_feature_data(
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
        desktop_kht_features,
    )
    run_age_xgb_regression(
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
        desktop_kht_features,
    )
    train_model(
        "Gender",
        desktop_kit_features_f1,
        desktop_kit_features_f2,
        desktop_kit_features_f3,
        desktop_kit_features_f4,
        desktop_kht_features,
    )
