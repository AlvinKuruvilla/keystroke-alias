import os
import csv
import pandas as pd

from parser_util import path_leaf


class KM_Parser:
    def __init__(
        self,
        _generated_bbmas_dir: str = os.path.join(os.getcwd(), "gen", "km"),
    ):
        self.bbmas_dir = _generated_bbmas_dir
        pass

    def get_bbmas_dir(self):
        return self.bbmas_dir

    def set_bbmas_dir(self, new_dir):
        self.bbmas_dir = new_dir

    def get_bbmas_file_path(self, km_file_path: str):
        km_filename = path_leaf(km_file_path)
        bbmas_file_path = os.path.join(self.get_bbmas_dir(), km_filename)
        return bbmas_file_path

    def create_bbmas_file(self, km_file_path: str):
        df = pd.read_csv(km_file_path)
        bbmas_file_path = self.get_bbmas_file_path(km_file_path)
        print(bbmas_file_path)
        input()
        with open(self.get_bbmas_file_path(km_file_path), "w+") as f:
            rows = zip(
                KM_Parser.make_id_column(df),
                KM_Parser.get_key_column(df),
                KM_Parser.get_action_column(df),
                KM_Parser.get_time_column(df),
            )
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def get_action_column(df: pd.DataFrame):
        new_actions = []
        actions = df["Press or Release"].tolist()
        for action in actions:
            if action == "P":
                new_actions.append("0")
            if action == "R":
                new_actions.append("1")
        return new_actions

    @staticmethod
    def get_key_column(df: pd.DataFrame):
        keys = df["Key"].tolist()
        return keys

    @staticmethod
    def get_time_column(df: pd.DataFrame):
        times = df["Time"].tolist()
        return times

    @staticmethod
    def make_id_column(df: pd.DataFrame):
        id = 0
        ids = []
        row_count = df.shape[0]
        for _ in range(row_count):
            ids.append(id)
            id += 1
        return ids


if __name__ == "__main__":
    parser = KM_Parser()
    df = pd.read_csv("/Users/alvinkuruvilla/Dev/typing-data-extractor/testdata/456.csv")
    parser.create_bbmas_file(
        "/Users/alvinkuruvilla/Dev/typing-data-extractor/testdata/456.csv"
    )
