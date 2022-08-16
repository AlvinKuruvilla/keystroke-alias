import os
import pickle
from rich.traceback import install

from models.regression_age_task import run_age_xgb_regression
from models.rnn_gender import train_model


install()
from features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
    get_all_users_features_advanced_word,
)

if __name__ == "__main__":
    dir_name = "data/"
    dir_path = os.path.join(os.getcwd(), dir_name, "km/")
    selected_profile_path = os.path.join(dir_path)
    desktop_kht_features = get_all_users_features_KHT(selected_profile_path)
    print(desktop_kht_features)
