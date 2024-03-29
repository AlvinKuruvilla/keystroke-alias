import os
import enum
import pandas as pd
import collections
from tqdm import tqdm
from rich.traceback import install
import statistics
import matplotlib.pyplot as plt

from fpd.dataset import TextDataset

install()


class Platform(enum.Enum):
    FACEBOOK = 0
    INSTAGRAM = 1
    TWITTER = 2


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

    def keystroke_count(self):
        # NOTE: A "keystroke" in this context only refers to the press events within the file
        data = self.data()
        count = 0
        try:
            re = data.drop("Time", axis=1)
        except KeyError:
            re = data.drop(data.columns[[2]], axis=1)
        action = list(re.iloc[:, 0])
        keys = list(re.iloc[:, 1])
        assert len(keys) == len(action)
        for index in range(len(keys)):
            if action[index] == "P":
                count += 1
        return count


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


def average_keystroke_counts(platform: Platform):
    p = os.path.join(os.getcwd(), "data", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    count = 0
    file_count = 0
    for file in onlyfiles:
        if platform == Platform.FACEBOOK:
            if file.endswith(".csv") and file.startswith("f_"):
                file_count += 1
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                count += kf.keystroke_count()
        elif platform == Platform.INSTAGRAM:
            if file.endswith(".csv") and file.startswith("i_"):
                file_count += 1
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                count += kf.keystroke_count()
        elif platform == Platform.TWITTER:
            if file.endswith(".csv") and file.startswith("t_"):
                file_count += 1
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                count += kf.keystroke_count()
    return count // file_count


def stdev_keystroke_count(platform: Platform):
    p = os.path.join(os.getcwd(), "data", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    counts = []
    for file in onlyfiles:
        if platform == Platform.FACEBOOK:
            if file.endswith(".csv") and file.startswith("f_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        elif platform == Platform.INSTAGRAM:
            if file.endswith(".csv") and file.startswith("i_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        if platform == Platform.TWITTER:
            if file.endswith(".csv") and file.startswith("t_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
    return statistics.stdev(counts)


def min_keystroke_count(platform: Platform):
    p = os.path.join(os.getcwd(), "data", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    counts = []
    for file in onlyfiles:
        if platform == Platform.FACEBOOK:
            if file.endswith(".csv") and file.startswith("f_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        elif platform == Platform.INSTAGRAM:
            if file.endswith(".csv") and file.startswith("i_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        if platform == Platform.TWITTER:
            if file.endswith(".csv") and file.startswith("t_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
    return min(counts)


def max_keystroke_count(platform: Platform):
    p = os.path.join(os.getcwd(), "data", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    counts = []
    for file in onlyfiles:
        if platform == Platform.FACEBOOK:
            if file.endswith(".csv") and file.startswith("f_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        elif platform == Platform.INSTAGRAM:
            if file.endswith(".csv") and file.startswith("i_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
        if platform == Platform.TWITTER:
            if file.endswith(".csv") and file.startswith("t_"):
                kf = KeystrokeFile(os.path.join(os.getcwd(), "data", "km", file))
                counts.append(kf.keystroke_count())
    return max(counts)


def find_letters(keys):
    letters = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    matches = []
    for key in keys:
        clean = str(key).lower().strip()[1:-1]
        if clean in letters:
            matches.append(clean)
    print(matches)
    return matches


def find_special_characters(keys):
    punctuations = [
        "Key.space",
        "Key.cmd",
        "Key.tab",
        "Key.enter",
        "Key.ctrl",
        "Keys.caps_lock",
        "Key.backspace",
        "Key.esc",
        "Key.shift",
        "Key.shift_r",
        ",",
        ".",
        ";",
        "!" "?",
    ]

    matches = []
    for key in keys:
        clean = str(key).strip()[1:-1]
        print(clean)
        if clean in punctuations:
            matches.append(clean)
    # print(matches)
    return matches


def letter_frequency_graph(td: TextDataset):
    keys = td.get_key_series().tolist()
    matches = find_letters(keys)
    counter = collections.Counter(matches)
    plt.bar(counter.keys(), counter.values())
    plt.show()


def special_character_frequency_graph(td: TextDataset):
    keys = td.get_key_series().tolist()
    matches = find_special_characters(keys)
    counter = collections.Counter(matches)
    plt.bar(counter.keys(), counter.values())
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(matches) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.show()
