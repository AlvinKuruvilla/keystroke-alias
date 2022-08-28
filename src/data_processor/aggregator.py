import os
import pandas as pd


class KeystrokeFile:
    def __init__(self, path: str):
        self.path = path

    def get_path(self):
        return self.path

    def get_monographs(self):
        data = self.data()
        try:
            re = data.drop("Press or Release", axis=1)
            return re
        except KeyError:
            re = data.drop(data.columns[[0]], axis=1)
            return re

    def data(self):
        df = pd.read_csv(self.path, header=None)
        return df

    def all_keys(self):
        data = self.data()
        try:
            return list(data["Key"])
        except KeyError:
            return list(data.iloc[:, 1])

    def digraphs(self):
        i = 0
        data = self.all_keys()
        print(data)
        # input()
        digraphs = []
        while i < len(data):
            if i + 1 >= len(data):
                return digraphs
            digraph = []
            digraph.append(data[i])
            digraph.append(data[i + 1])
            digraphs.append(digraph)
            i += 1
        return digraphs


if __name__ == "__main__":
    kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "subset", "f_4_fpd1.csv"))
    digraphs = kf.digraphs()
    print(digraphs)
