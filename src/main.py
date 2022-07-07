import os
import pickle
from features.fe import (
    get_all_users_features_KHT,
    get_all_users_features_KIT,
    get_all_users_features_advanced_word,
)

if __name__ == "__main__":
    dir_name = "data"
    dir_path = os.path.join(os.getcwd(), dir_name)
    selected_profile_path = os.path.join(dir_path, "Desktop/")
    desktop_kht_features = get_all_users_features_KHT(selected_profile_path)

    pickle.dump(
        desktop_kht_features, open("desktop_kht_feature_dictionary.pickle", "wb")
    )

    desktop_kht_features = get_all_users_features_KIT(selected_profile_path)

pickle.dump(desktop_kht_features, open("desktop_kht_feature_dictionary.pickle", "wb"))
