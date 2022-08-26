import numpy as np
from custom.features.fe_util import load_feature_file
from custom.features.processor import top_feature_KIT


# return all the desktop features for free text
def get_desktop_features():
    # print(load_feature_file("kit_f1.pickle"))
    # input("pickle")
    desktop_kht_features = top_feature_KIT(load_feature_file("kht.pickle"))
    desktop_kit_features_f1 = top_feature_KIT(load_feature_file("kit_f1.pickle"))
    desktop_kit_features_f2 = top_feature_KIT(load_feature_file("kit_f2.pickle"))
    desktop_kit_features_f3 = top_feature_KIT(load_feature_file("kit_f3.pickle"))
    desktop_kit_features_f4 = top_feature_KIT(load_feature_file("kit_f4.pickle"))

    # print(type(desktop_kht_features))
    # input()
    # return np.array(desktop_kht_features)
    return np.concatenate(
        (
            np.array(desktop_kht_features),
            np.array(desktop_kit_features_f1),
            np.array(desktop_kit_features_f2),
            np.array(desktop_kit_features_f3),
            np.array(desktop_kit_features_f4),
        ),
        axis=1,
    )


def get_combined_features():
    desktop_features_combined = get_desktop_features()
    return np.concatenate((np.array(desktop_features_combined),), axis=1)
