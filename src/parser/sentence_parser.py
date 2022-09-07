import pandas as pd
from fpd.feature_gen import split_into_four, merge_dataframes

# Handles strings like "<0>""
def remove_invalid_keystrokes(data):
    for i in range(0, len(data)):
        df = data[i]
        for row in df.itertuples():
            # print(row[2])
            if row[2] == "<0>":
                # print("HERE")
                num = int(row.Index)
                # print(num)
                rem = df.drop(index=num)
                data[i] = rem
    # After removing the weird values the size of each dataframe element is
    # smaller so we need to coalesce. Re-partitioning will be the job of
    # subsequent methods that use the return value of this method
    return merge_dataframes(data)


class SentenceParser:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def path(self):
        return self.csv_file_path

    def as_df(self):
        return pd.read_csv(self.path(), header=None)

    def letters(self):
        return remove_invalid_keystrokes(split_into_four(self.as_df()))

    def make_sentences(self):
        filtered = []
        ignorable = ["Key.cmd", "Key.tab", "Key.shift", "Key.shift_r"]
        keys = self.letters().iloc[:, 1]
        action = self.letters().iloc[:, 0]
        assert action.size == keys.size
        for i in range(keys.size):
            if action[i] == "P":
                filtered.append(keys[i])
        sentence = ""
        next = 0
        for i in range(len(filtered)):
            if i + 1 < len(filtered) and i - 1 >= 0:
                key = filtered[i]
                next = (next + 1) % len(filtered)
                if not key in ignorable:
                    if key == "Key.enter":
                        sentence += "\n"
                        continue
                    elif key == "Key.space":
                        sentence += " "
                        continue
                    elif key == "Key.backspace" or key == "Key.ctrl":
                        continue
                    elif filtered[i - 1] == "Key.ctrl":
                        # Technically when the participant hits ctrl c, they are ending the current session and
                        # starting a new one so that should probably be thought of a as a distinct sentence
                        sentence += "\n"
                        continue
                    elif filtered[next] == "Key.backspace":
                        continue
                    sentence += key.strip()
        return sentence
