import os
import pandas as pd
import collections
from tqdm import tqdm


def flatten(l):
    return [item for sublist in l for item in sublist]


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
        # print(data)
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

    def collapse_into_strings(self, digraphs):
        ret = []
        for digraph in digraphs:
            string = digraph[0] + digraph[1]
            ret.append(string)
        return ret

    def digraph_frequency(self, digraphs):
        digraph_strs = self.collapse_into_strings(digraphs)
        frequency = collections.Counter(digraph_strs)
        return dict(frequency)

    def n_most_common_digraphs(self, digraphs_frequencies_dict, n: int):
        ret = []
        sorted_frequencies = sorted(
            digraphs_frequencies_dict.items(), key=lambda x: x[1]
        )
        if n <= 0:
            raise ValueError("N must be greater than 0")
        d = list(sorted_frequencies)[-n:]
        for digraph_set in d:
            ret.append(digraph_set[0])
        return ret


def all_common_digraphs(n: int):
    p = os.path.join(os.getcwd(), "data", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    all = []
    for file in tqdm(onlyfiles):
        if file.endswith(".csv"):
            kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
            digraphs = kf.digraphs()
            all.append(digraphs)
    flat = flatten(all)
    frequencies = kf.digraph_frequency(flat)
    return kf.n_most_common_digraphs(frequencies, n)


if __name__ == "__main__":
    print(all_common_digraphs(30))
