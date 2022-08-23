import os
from rich.traceback import install
from features.fe_util import load_feature_file, pickle_all_feature_data

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
    run_age_xgb_regression()
    # train_model("Gender")
