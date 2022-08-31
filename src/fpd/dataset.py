import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path)

    def path(self):
        return self.dataset_path

    def get_data(self):
        return self.data

    def as_numpy_array(self):
        return self.data.to_numpy()

    def feature_names(self):
        # The last column denotes whether or not the user is a fake profile or not so we want to ignore
        return list(self.get_data().columns[:-1])

    def target_names(self):
        return ["Fake Profile", "Genuine Profile"]

    def target(self):
        return np.array(list(self.get_data().iloc[:, len(self.feature_names())]))
