import pandas as pd
import numpy as np
import statistics


class Dataset:
    # TODO: Make the methods more flexible
    # A wrapper class to represent the ideal csv dataset for our use-case with nicely organized columns
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path)

    def path(self):
        return self.dataset_path

    def get_data(self):
        df = self.data
        df = df.drop(columns=["ID"])
        return df.drop(columns=["Target"])

    def as_numpy_array(self):
        return self.get_data().to_numpy()

    def feature_names(self):
        # The last column denotes whether or not the user is a fake profile or not so we want to ignore
        return list(self.get_data().columns[:-1])

    def target_names(self):
        return ["Fake Profile", "Genuine Profile"]

    def target(self):
        return np.array(list(self.get_data().iloc[:, len(self.feature_names())]))


class TextDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def path(self):
        return self.dataset_path

    def get_data(self):
        data = self.to_df()
        return data.drop(columns=["ID", "Class"])

    def get_raw_data(self):
        return self.to_df()

    def as_numpy_array(self):
        return self.get_data().to_numpy(dtype=object)

    def target_names(self):
        return ["Fake Profile", "Genuine Profile"]

    def target(self):
        return np.array(self.get_raw_data().iloc[:, -1:])

    def feature_names(self):
        # The last column denotes whether or not the user is a fake profile or not so we want to ignore
        return list(self.get_data().columns[:-1])

    def to_df(self):
        ids = self.get_id_series()
        keys = self.get_key_series()
        medians = self.get_statistical_median_series()
        means = self.get_statistical_mean_series()
        modes = self.get_statistical_mode_series()
        stdevs = self.get_statistical_stdev_series()
        classes = self.get_class_series()
        assert (
            ids.size
            == keys.size
            == medians.size
            == means.size
            == modes.size
            == stdevs.size
            == classes.size
        )
        return pd.concat([ids, keys, medians, means, modes, stdevs, classes], axis=1)

    def parse_brackets(self):
        ret = []
        with open(self.path(), "r") as f:
            next(f)
            for line in f:
                beginning_bracket_index = line.find("[") + 1
                ending = line.find("]")
                temp = line[beginning_bracket_index:ending]
                ret.append(temp.strip().split(","))
        return ret

    def get_key_series(self):
        ret = []
        rows = self.parse_brackets()
        for row in rows:
            # print(row)
            key = row[0]
            key = key.replace('"', "")
            ret.append(key)
        return pd.Series(ret, name="Key(s)")

    def get_id_series(self):
        ret = []
        with open(self.path(), "r") as f:
            next(f)
            for line in f:
                ret.append(line.strip().split(" ")[0])
        return pd.Series(ret, name="ID")

    def get_timings_series(self):
        ret = []
        rows = self.parse_brackets()
        for row in rows:
            # NOTE: I am wary about doing this because we might end
            # up replacing keys that were meant to have quotes around like ` or " them so we will have to carefully look at the results
            clean = [i.replace('"', "") for i in row]

            clean = clean[1:]
            ret.append(clean)
        return pd.Series(ret, name="Timings")

    def get_class_series(self):
        ret = []
        with open(self.path(), "r") as f:
            next(f)
            for line in f:
                ret.append(line.strip().split(" ")[-1])
        return pd.Series(ret, name="Class")

    def get_statistical_median_series(self):
        # FIXME: There is a lot of parsing issues towards the end of the
        # Series... we need to revisit how the timings are being parsed
        timings = self.get_timings_series()
        hold = []
        medians = []
        for values in timings:
            # print(values)
            for value in values:
                try:
                    clean = int(value.replace(" ", ""))
                except ValueError:
                    # print("Invalid value found: %s" % value)
                    clean = 0
                hold.append(clean)
            medians.append(np.median(np.array(hold)))
            hold = []
        return pd.Series(medians, name="Medians")

    def get_statistical_mean_series(self):
        # FIXME: There is a lot of parsing issues towards the end of the
        # Series... we need to revisit how the timings are being parsed
        timings = self.get_timings_series()
        hold = []
        means = []
        for values in timings:
            # print(values)
            for value in values:
                try:
                    clean = int(value.replace(" ", ""))
                except ValueError:
                    # print("Invalid value found: %s" % value)
                    clean = 0
                hold.append(clean)
            means.append(np.median(np.array(hold)))
            hold = []
        return pd.Series(means, name="Means")

    def get_statistical_mode_series(self):
        # FIXME: There is a lot of parsing issues towards the end of the
        # Series... we need to revisit how the timings are being parsed
        timings = self.get_timings_series()
        hold = []
        modes = []
        for values in timings:
            # print(values)
            for value in values:
                try:
                    clean = int(value.replace(" ", ""))
                except ValueError:
                    # print("Invalid value found: %s" % value)
                    clean = 0
                hold.append(clean)
            try:
                modes.append(statistics.mode(np.array(hold)))
            except statistics.StatisticsError:
                # print("ERROR: No unique mode found")
                modes.append(0)
            hold = []
        return pd.Series(modes, name="Modes")

    def get_statistical_stdev_series(self):
        # FIXME: There is a lot of parsing issues towards the end of the
        # Series... we need to revisit how the timings are being parsed
        timings = self.get_timings_series()
        hold = []
        stdevs = []
        for values in timings:
            # print(values)
            for value in values:
                try:
                    clean = int(value.replace(" ", ""))
                except ValueError:
                    # print("Invalid value found: %s" % value)
                    clean = 0
                hold.append(clean)
            try:
                stdevs.append(statistics.stdev(hold))
            except statistics.StatisticsError:
                # print("ERROR: Less than 2 points to make a standard deviation variance")
                stdevs.append(0)
            hold = []
        return pd.Series(stdevs, name="Standard Deviation")
